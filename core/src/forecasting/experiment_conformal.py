import os
import json
import time

import numpy as np
import torch
import torch.nn as nn

from .data_provider import get_forecasting_loader_and_dataset

import models
from common.experiment import Experiment, EarlyStopping

class ConformalExperiment(Experiment):

    def __init__(self, configs):
        super().__init__(configs)
        self._build_dataloaders()
        self._build_model()
        self.alpha = configs.error_rate
        self.horizon= configs.pred_len

    def _build_model(self):
        self.model = getattr(models, self.configs.model)(self.configs)

        if self.configs.use_multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.configs.gpus)
        
        self.model.to(self.device)

    def _build_dataloaders(self):
        self.train_loader, self.train_set = get_forecasting_loader_and_dataset(self.configs, split='train', shuffle=True)
        self.val_loader, self.val_set = get_forecasting_loader_and_dataset(self.configs, split='val', shuffle=False)
        self.test_loader, self.test_set = get_forecasting_loader_and_dataset(self.configs, split='test', shuffle=False)

    def _build_criterion(self):
        if self.configs.criterion == 'MSE':
            criterion = nn.MSELoss()
        return criterion

    def train(self):
        criterion = self._build_criterion()
        optimizer, scheduler = self._build_optimizer()
        early_stopping = EarlyStopping(self.configs.patience)

        global_step = 0
        for epoch in range(self.configs.train_epochs):
            self.model.train()

            train_loss = []
            epoch_start = time.time()
            for batch in self.train_loader:
                optimizer.zero_grad()
                batch_pred, batch_true = self._forward_step(batch)
                loss = criterion(batch_pred, batch_true)

                loss.backward()
                optimizer.step()
                global_step += 1

                lr = optimizer.param_groups[0]['lr']
                self.writer.add_scalar('B.LR', lr, global_step)

                scheduler.step()

                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            val_loss, val_metrics = self.evaluate(self.val_loader, criterion)
            test_loss, test_metrics = self.evaluate(self.test_loader, criterion)

            val_mse, val_mae = val_metrics['MSE'], val_metrics['MAE']
            test_mse, test_mae = test_metrics['MSE'], test_metrics['MAE']
            print(f'Epoch {epoch + 1} | Time cost {time.time() - epoch_start:.2f}s')
            print(f'{"Train loss":<10} {train_loss:.4f} | {"Val loss":<8} {val_loss:.4f} | Test loss {test_loss:.4f}')
            print(f'{"Val MSE":<10} {val_mse:.4f} | {"Val MAE":<8} {val_mae:.4f} |')
            print(f'{"Test MSE":<10} {test_mse:.4f} | {"Test MAE":<8} {test_mae:.4f} |')

            early_stopping(val_loss)
            if early_stopping.save_model:
                self._save_checkpoint()
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.configs.lr_scheduler == 'reduce_on_plateau':
                scheduler.step(val_loss)
            elif self.configs.lr_scheduler != 'one_cycle':
                scheduler.step()

            self.writer.add_scalar('A.Loss/a.Train', train_loss, global_step)
            self.writer.add_scalar('A.Loss/b.Val', val_loss, global_step)
            self.writer.add_scalar('A.Loss/c.Test', test_loss, global_step)

            self.writer.add_scalar('C.MSE/a.Val', val_metrics['MSE'], global_step)
            self.writer.add_scalar('C.MSE/b.Test', test_metrics['MSE'], global_step)
            self.writer.add_scalar('D.MAE/a.Val', val_metrics['MAE'], global_step)
            self.writer.add_scalar('D.MAE/b.Test', test_metrics['MAE'], global_step)

        self._load_best_checkpoint()
        return self.model
    
    @torch.no_grad()
    def evaluate(self, eval_loader, criterion=None, load_best=False, save_result=False):
        if load_best:
            self._load_best_checkpoint()

        self.model.eval()

        eval_loss, pred, true = [], [], []
        for batch in eval_loader:
            batch_pred, batch_true = self._forward_step(batch)
            if criterion is not None:
                loss = criterion(batch_pred, batch_true)
                eval_loss.append(loss.item())

            pred.append(batch_pred.cpu().numpy())
            true.append(batch_true.cpu().numpy())

        if criterion is not None:
            eval_loss = np.mean(eval_loss)
        else:
            eval_loss = None

        pred = np.concatenate(pred)
        true = np.concatenate(true)

        eval_mse = np.mean((pred -  true) ** 2).item()
        eval_mae = np.mean(np.abs(pred - true)).item()
        eval_metrics = {'MSE': eval_mse, 'MAE': eval_mae}

        if save_result:
            self._save_results(eval_metrics)
        return eval_loss, eval_metrics
   
    @torch.no_grad() 
    def calibrate(self, calib_loader, calib_set, load_best=False, save_result=True):        
        if load_best:
            self._load_best_checkpoint()

        self.model.eval()
        n_calibration = len(calib_set)
        calibration_scores = []
        for batch in calib_loader:
            batch_pred, batch_true = self._forward_step(batch)
            score = self.nonconformity(batch_pred, batch_true)
            calibration_scores.append(score)

        # [output_size, horizon, n_samples]
        self.calibration_scores = torch.vstack(calibration_scores).transpose(0, 2)

        # [horizon, output_size]
        q = min((n_calibration + 1.0) * (1 - self.alpha) / n_calibration, 1)
        corrected_q = min((n_calibration + 1.0) * (1 - self.alpha / self.horizon) / n_calibration, 1)

        self.critical_calibration_scores = self.get_critical_scores(calibration_scores=self.calibration_scores, q=q)
        self.corrected_critical_calibration_scores = self.get_critical_scores(
            calibration_scores=self.calibration_scores, q=corrected_q
        )
        if save_result:
            self._save_calib_scores(self.critical_calibration_scores.cpu().detach(), "score.pt")
            self._save_calib_scores(self.corrected_critical_calibration_scores.cpu().detach(), "score_corrected.pt")

        return self.critical_calibration_scores, self.corrected_critical_calibration_scores

    def get_critical_scores(self, calibration_scores, q):
        return torch.tensor(
            [
                [
                    torch.quantile(position_calibration_scores, q=q)
                    for position_calibration_scores in feature_calibration_scores
                ]
                for feature_calibration_scores in calibration_scores
            ]
        ).T

    def nonconformity(self, pred, true):
        return torch.nn.functional.l1_loss(pred, true, reduction="none")

    def _forward_step(self, batch):
        batch = [tensor.to(self.device) for tensor in batch]
        batch_x, batch_x_stamp, batch_y, batch_y_stamp = batch

        batch_x_dec = batch_y.clone()
        batch_x_dec[:, -self.configs.pred_len:, :] = 0

        batch_pred = self.model(batch_x, batch_x_stamp, batch_x_dec, batch_y_stamp).unsqueeze(2)

        batch_pred = batch_pred[:, -self.configs.pred_len:, :]
        batch_true = batch_y[:, -self.configs.pred_len:, :]
        return batch_pred, batch_true
    
    def _save_results(self, metrics):
        res_path = os.path.join(self.exp_dir, 'result.json')
        with open(res_path, 'w') as fout:
            json.dump(metrics, fout, indent=4)

    def _save_calib_scores(self, score, name):
        score_path = os.path.join(self.exp_dir, name)
        torch.save(score, score_path)
