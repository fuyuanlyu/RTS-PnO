import os
import json
import time

import numpy as np
import torch
import torch.nn as nn

from .data_provider import get_forecasting_loader_and_dataset

import models
from common.experiment import Experiment, EarlyStopping

class HeuristicAllocateExperiment(Experiment):

    def __init__(self, configs):
        super().__init__(configs)
        self.prev_exp_dir = os.path.join('output', configs.prev_exp_id)
        self._build_dataloaders()
        self._build_model()
        self._load_constraint()
        self.alpha = configs.error_rate
        self.horizon= configs.pred_len
        self.risk = configs.risk
        self.topk = configs.topk

    def _build_model(self):
        self.model = getattr(models, self.configs.model)(self.configs)

        if self.configs.use_multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.configs.gpus)
        
        self.model.to(self.device)
        self.load_checkpoint(model_path=os.path.join(self.prev_exp_dir, 'model.pt'))

    def _build_dataloaders(self):
        self.test_loader, self.test_set = get_forecasting_loader_and_dataset(self.configs, split='test', shuffle=False)

    def _build_criterion(self):
        if self.configs.criterion == 'MSE':
            criterion = nn.MSELoss()
        return criterion

    def _load_constraint(self):
        constraint_path = os.path.join(self.prev_exp_dir, 'score.pt')
        self.pretrained_constraint = torch.load(constraint_path).squeeze().unsqueeze(0).to("cpu").detach().numpy()

    def evaluate(self, eval_loader, scaler):
        result = self.evaluate_regret(eval_loader, scaler)
        return result

    def _cal_regret(self, y_pred, y_true, k=3):
        rank_idx = np.argpartition(y_pred, k)
        selected_value = y_true[rank_idx[:k]]
        cost = np.mean(selected_value)
        optimal = np.min(y_true)
        return cost - optimal

    @torch.no_grad()
    def evaluate_regret(self, eval_loader, scaler=None):
        self.model.eval()
        loss = 0
        rel_loss = 0
        trial = 0

        for batch in eval_loader:
            batch = [tensor.to(self.device) for tensor in batch]
            batch_pred, batch_true = self._forward_step(batch)
            batch_pred = batch_pred.squeeze().to("cpu").detach().numpy()
            batch_true = batch_true.squeeze().to("cpu").detach().numpy()
            if self.risk:
                batch_pred = batch_pred + self.pretrained_constraint
            else:
                batch_pred = batch_pred

            if scaler is not None:
                batch_pred = scaler.inverse_transform(batch_pred)
                batch_true = scaler.inverse_transform(batch_true)

            for j in range(batch_true.shape[0]):
                this_regret = self._cal_regret(batch_pred[j], batch_true[j], self.topk)
                loss += this_regret
                rel_loss += (this_regret / np.min(batch_true[j]))
            
            trial += batch_true.shape[0]

        result = {
            'abs_regret': loss / trial,
            'rel_regret': rel_loss / trial
        }

        return result
    
    def calibrate(self, calib_loader, calib_set, load_best=False):        
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
        self._save_calib_scores(self.critical_calibration_scores.cpu().detach(), "score.pt")
        self._save_calib_scores(self.corrected_critical_calibration_scores.cpu().detach(), "score_corrected.pt")

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
