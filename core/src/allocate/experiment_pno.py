import os, sys
import json
import time

import numpy as np
import pyepo
import pyepo.metric
import torch
import torch.nn as nn
import torch.optim as optim

from .data_provider import get_allocating_loader_and_dataset
# sys.path.append(os.path.realpath('.'))
import models
from models.Allocate import AllocateModel
from common.experiment import Experiment, EarlyStopping

class PnOExperiment(Experiment):
    def __init__(self, configs):
        super().__init__(configs)
        self.alpha = configs.error_rate
        self.horizon= configs.pred_len
        self._build_forecast_model() # Maybe we should load the pretrained model in the first step
        self._init_constraint()
        self._build_allocate_model()
        self._build_dataloaders()
        if self.configs.data_path != "exchange_rate_600s.csv":
            self._cal_constraint(self.val_loader, self.val_set)
            self._build_allocate_model()

    def _build_dataloaders(self):
        self.train_loader, self.train_set = get_allocating_loader_and_dataset(self.configs, self.allocate_model, split='train', shuffle=True)
        self.val_loader, self.val_set = get_allocating_loader_and_dataset(self.configs, self.allocate_model, split='val', shuffle=False)
        self.test_loader, self.test_set = get_allocating_loader_and_dataset(self.configs, self.allocate_model, split='test', shuffle=False)

    def _build_criterion(self):
        if self.configs.criterion == "SPO+":
            criteria = pyepo.func.SPOPlus(self.allocate_model, processes=self.configs.num_workers) # Maybe the loss function can be added to the MSE (as an ablation)
        elif self.configs.criterion == "MSE+SPO+":
            loss_spo = pyepo.func.SPOPlus(self.allocate_model, processes=self.configs.num_workers)
            loss_mse = nn.MSELoss()
            criteria = lambda y_pred, y, action, value: loss_spo(y_pred, y, action, value) + self.configs.criterion_weight * loss_mse(y_pred, y)
        return criteria

    def _build_forecast_model(self):
        self.forecast_model = getattr(models, self.configs.model)(self.configs)
        
        if self.configs.use_multi_gpu:
            self.forecast_model = nn.DataParallel(self.forecast_model, device_ids=self.configs.gpus)

        self.forecast_model.to(self.device)

        self.model = self.forecast_model # Make it compatible with common experiment

    def _build_allocate_model(self):
        self.allocate_model = AllocateModel(self.constraint, self.configs.uncertainty_quantile)

    def nonconformity(self, pred, true):
        return torch.nn.functional.l1_loss(pred, true, reduction="none")
    
    def _init_constraint(self):
        self.constraint = torch.ones(self.configs.n_vars, self.configs.pred_len) * 100000

    @torch.no_grad()
    def _cal_constraint(self, calib_loader, calib_set):
        self.forecast_model.eval()
        n_calibration = len(calib_set)
        calibration_scores = []
        for batch in calib_loader:
            batch = [tensor.to(self.device) for tensor in batch]
            batch_x, batch_y, optimal_action, optimal_value = batch
            batch_y_pred = self.forecast_model(batch_x)
            score = self.nonconformity(batch_y_pred, batch_y)
            calibration_scores.append(score)

        # [output_size, horizon, n_samples]
        self.calibration_scores = torch.vstack(calibration_scores).transpose(0, 1)

        # [horizon, output_size]
        q = min((n_calibration + 1.0) * (1 - self.alpha) / n_calibration, 1)
        corrected_q = min((n_calibration + 1.0) * (1 - self.alpha / self.horizon) / n_calibration, 1)

        self.critical_calibration_scores = self.get_critical_scores(calibration_scores=self.calibration_scores, q=q)
        self.corrected_critical_calibration_scores = self.get_critical_scores(
            calibration_scores=self.calibration_scores, q=corrected_q
        )

        self.constraint = self.critical_calibration_scores

    def get_critical_scores(self, calibration_scores, q):
        """
        Computes critical calibration scores from scores in the calibration set.

        Args:
            calibration_scores: calibration scores for each example in the
                calibration set.
            q: target quantile for which to return the calibration score

        Returns:
            critical calibration scores for each target horizon
        """

        return torch.tensor(
            [
                [
                    torch.quantile(position_calibration_scores, q=q)
                    for position_calibration_scores in feature_calibration_scores
                ]
                for feature_calibration_scores in calibration_scores
            ]
        ).T
    
    @torch.no_grad()
    def evaluate_forecast(self, eval_loader, criterion=None, load_best=False):
        if load_best:
            self._load_best_checkpoint()
        
        self.forecast_model.eval()

        eval_loss, pred, true = [], [], []
        for batch in eval_loader:
            batch = [tensor.to(self.device) for tensor in batch]
            batch_x, batch_y, action, optimal_value = batch
            batch_y_pred = self.forecast_model(batch_x)
            if criterion is not None:
                loss = criterion(batch_y_pred, batch_y)
                eval_loss.append(loss.item())
            
            pred.append(batch_y_pred.cpu().numpy())
            true.append(batch_y.cpu().numpy())
        
        if criterion is not None:
            eval_loss = np.mean(eval_loss)
        else:
            eval_loss = None

        pred = np.concatenate(pred)
        true = np.concatenate(true)

        eval_mse = np.mean((pred -  true) ** 2).item()
        eval_mae = np.mean(np.abs(pred - true)).item()
        eval_metrics = {'MSE': eval_mse, 'MAE': eval_mae}
        
        return eval_loss, eval_metrics

    @torch.no_grad()
    def _cal_regret(self, allocate_model, y_pred, y):
        optimal_cost = np.min(y)
        # opt sol for pred cost
        allocate_model.setObj(y_pred)
        sol, _ = allocate_model.solve()
        # obj with true cost
        sol_cost = np.dot(sol, y)
        if allocate_model.modelSense == pyepo.EPO.MINIMIZE:
            loss = sol_cost - optimal_cost
        if allocate_model.modelSense == pyepo.EPO.MAXIMIZE:
            loss = optimal_cost - sol_cost
        return loss

    @torch.no_grad()
    def evaluate_regret(self, eval_loader, scaler=None):
        self.forecast_model.eval()
        loss = 0
        rel_loss = 0
        trial = 0

        for batch in eval_loader:
            batch = [tensor.to(self.device) for tensor in batch]
            batch_x, batch_y, optimal_action, optimal_value = batch

            # Predict
            with torch.no_grad():
                batch_y_pred = self.forecast_model(batch_x).to("cpu").detach().numpy()
                batch_y = batch_y.to("cpu").detach().numpy()
                optimal_value = optimal_value.to("cpu").detach().numpy()

            # Different from pyepo.metric.regret
            # Inverse Transform data
            if scaler is not None:
                batch_y_pred = scaler.inverse_transform(batch_y_pred)
                batch_y = scaler.inverse_transform(batch_y)
                optimal_value = scaler.inverse_transform(optimal_value)

            # Solve
            for j in range(batch_y_pred.shape[0]):
                # accumulate loss
                this_regret = self._cal_regret(self.allocate_model, batch_y_pred[j], batch_y[j])
                loss += this_regret
                rel_loss += (this_regret / np.min(batch_y[j]))

            trial += batch_x.shape[0]

        result = {
            'abs_regret': loss / trial,
            'rel_regret': rel_loss / trial
        }

        return result
        
    @torch.no_grad()
    def evaluate(self, eval_loader, scaler, criterion=None, load_best=False, save_result=False):
        eval_regret = self.evaluate_regret(eval_loader, scaler)
        _, eval_metrics = self.evaluate_forecast(eval_loader, criterion, load_best)
        eval_metrics['Regret'] = eval_regret['abs_regret']
        eval_metrics['Rel Regret'] = eval_regret['rel_regret']

        if save_result:
            self._save_results(eval_metrics)

        return eval_metrics

    def allocate(self):
        criterion = self._build_criterion()
        optimizer, scheduler = self._build_optimizer()
        early_stopping = EarlyStopping(self.configs.patience)

        global_step = 0
        for epoch in range(self.configs.train_epochs):
            self.forecast_model.train()

            train_loss = []
            epoch_start = time.time()
            for batch in self.train_loader:
                optimizer.zero_grad()
                batch = [tensor.to(self.device) for tensor in batch]
                batch_x, batch_y, optimal_action, optimal_value = batch
                batch_y_pred = self.forecast_model(batch_x)

                loss = criterion(batch_y_pred, batch_y, optimal_action, optimal_value)

                loss.backward()
                optimizer.step()

                global_step += 1

                lr = optimizer.param_groups[0]['lr']
                self.writer.add_scalar('B.LR', lr, global_step)

                scheduler.step()
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            print(f"Epoch: {epoch + 1} || Training Time: {time.time() - epoch_start:.2f}s")
            print(f"Epoch: {epoch + 1} || Training Loss: {train_loss:.10f}")

            # Update the constraints and rebuild the dataset and allocation model
            self._cal_constraint(self.val_loader, self.val_set)
            self._build_allocate_model()

            val_start = time.time()
            val_metrics = self.evaluate(self.val_loader, self.val_set.scaler)
            val_end = time.time()
            test_metrics = self.evaluate(self.test_loader, self.test_set.scaler)

            print(f"Epoch: {epoch + 1} Val  || Time: {val_end - val_start:.2f}")
            print(f"Epoch: {epoch + 1} Val  || Regret: {val_metrics['Regret']:.8f} || Rel Regret: {val_metrics['Rel Regret']:.8f} || MSE: {val_metrics['MSE']:.4f} || MAE: {val_metrics['MAE']:.4f}")
            print(f"Epoch: {epoch + 1} Test || Regret: {test_metrics['Regret']:.8f} || Rel Regret: {test_metrics['Rel Regret']:.8f} || MSE: {test_metrics['MSE']:.4f} || MAE: {test_metrics['MAE']:.4f}")


            if self.configs.train_epochs > 50:
                early_stopping(val_metrics['Regret'])
                if early_stopping.save_model:
                    self._save_checkpoint()
                    self._save_calib_scores(self.constraint.cpu().detach(), "constraint.pt")
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                self._save_checkpoint()
                self._save_calib_scores(self.constraint.cpu().detach(), "constraint.pt")

            self.writer.add_scalar('A.Loss/a.Train', train_loss, global_step)
            self.writer.add_scalar('E.Regret/a.Val', val_metrics['Regret'], global_step)
            self.writer.add_scalar('E.Regret/b.Test', test_metrics['Regret'], global_step)
            self.writer.add_scalar('C.MSE/a.Val', val_metrics['MSE'], global_step)
            self.writer.add_scalar('C.MSE/b.Test', test_metrics['MSE'], global_step)
            self.writer.add_scalar('D.MAE/a.Val', val_metrics['MAE'], global_step)
            self.writer.add_scalar('D.MAE/b.Test', test_metrics['MAE'], global_step)

        self._load_best_checkpoint()
        return self.model

    def _save_results(self, metrics):
        res_path = os.path.join(self.exp_dir, 'result.json')
        with open(res_path, 'w') as fout:
            json.dump(metrics, fout, indent=4)
    
    def _save_calib_scores(self, score, name):
        score_path = os.path.join(self.exp_dir, name)
        torch.save(score, score_path)