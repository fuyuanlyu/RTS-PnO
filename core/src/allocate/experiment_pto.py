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
from common.experiment import Experiment

class PtOExperiment(Experiment):
    def __init__(self, configs):
        super().__init__(configs)
        self.prev_exp_dir = os.path.join('output', configs.prev_exp_id)
        self._build_forecast_model() # Maybe we should load the pretrained model in the first step
        self._load_constraint()
        self._build_allocate_model()
        self._build_dataloaders()
    
    def _build_dataloaders(self):
        self.test_loader, self.test_set = get_allocating_loader_and_dataset(self.configs, self.allocate_model, split='test', shuffle=False)

    def _build_forecast_model(self):
        self.forecast_model = getattr(models, self.configs.model)(self.configs)
        
        if self.configs.use_multi_gpu:
            self.forecast_model = nn.DataParallel(self.forecast_model, device_ids=self.configs.gpus)

        self.forecast_model.to(self.device)

        self.model = self.forecast_model # Make it compatible with common experiment
        if self.configs.load_prev_weights:
            self.load_checkpoint(model_path=os.path.join(self.prev_exp_dir, 'model.pt'))

    def _build_allocate_model(self):
        self.allocate_model = AllocateModel(self.pretrained_constraint, self.configs.uncertainty_quantile)

    def _load_constraint(self):
        constraint_path = os.path.join(self.prev_exp_dir, 'score.pt')
        self.pretrained_constraint = torch.transpose(torch.load(constraint_path), 0, 1)
       
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
        total_regret = 0
        abs_regrets = []
        total_rel_regret = 0
        rel_regrets = []
        optsum = 0
        trial = 0

        for batch in eval_loader:
            batch = [tensor.to(self.device) for tensor in batch]
            batch_x, batch_y, optimal_action, optimal_value = batch

            # Predict
            with torch.no_grad():
                batch_y_pred = self.forecast_model(batch_x).to("cpu").detach().numpy()
                batch_y = batch_y.to("cpu").detach().numpy()
                optimal_value = optimal_value.to("cpu").detach().numpy()

            # Different from pyepo
            # Inverse Transform data
            if scaler is not None:
                batch_y_pred = scaler.inverse_transform(batch_y_pred)
                batch_y = scaler.inverse_transform(batch_y)
                optimal_value = scaler.inverse_transform(optimal_value)

            # Solve
            for j in range(batch_y_pred.shape[0]):
                this_regret = self._cal_regret(self.allocate_model, batch_y_pred[j], batch_y[j])
                total_regret += this_regret
                abs_regrets.append(this_regret)
                this_rel_regret = this_regret / np.min(batch_y[j])
                total_rel_regret += this_rel_regret
                rel_regrets.append(this_rel_regret)

            optsum += abs(optimal_value).sum().item()
            trial += batch_x.shape[0]

        return total_regret / trial, abs_regrets, total_rel_regret / trial, rel_regrets
    
    @torch.no_grad()
    def evaluate(self, eval_loader, scaler, criterion=None, save_result=False):
        eval_regret, eval_regrets, eval_rel_regret, eval_rel_regrets = self.evaluate_regret(eval_loader, scaler)
        _, eval_metrics = self.evaluate_forecast(eval_loader, criterion)
        eval_metrics['Regret'] = eval_regret
        eval_metrics['Regrets'] = eval_regrets
        eval_metrics['Rel Regret'] = eval_rel_regret
        eval_metrics['Rel Regrets'] = eval_rel_regrets

        if save_result:
            self._save_results(eval_metrics)

        return eval_metrics
    
    def _save_results(self, metrics):
        res_path = os.path.join(self.exp_dir, 'result.json')
        with open(res_path, 'w') as fout:
            json.dump(metrics, fout, indent=4)
