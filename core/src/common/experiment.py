import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class Experiment:

    def __init__(self, configs):
        self.configs = configs
        self.device = f'cuda:{self.configs.gpu}'
        self.exp_dir = os.path.join('output', configs.exp_id)
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))

    def load_checkpoint(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

    def _build_model(self):
        raise NotImplementedError

    def _build_dataloaders(self):
        raise NotImplementedError

    def _build_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.configs.lr)
        if self.configs.lr_scheduler == 'fixed':
            scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
        elif self.configs.lr_scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.configs.gamma)
        elif self.configs.lr_scheduler == 'one_cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.configs.lr,
                epochs=self.configs.train_epochs, steps_per_epoch=len(self.train_loader),
                pct_start=self.configs.pct_start
            )
        elif self.configs.lr_scheduler == 'fixed_then_exponential':
            lr_lambda = lambda epoch: 1 if epoch < 4 else 0.9 ** (epoch - 3)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.configs.lr_scheduler == 'step_then_fixed':
            lr_lambda = lambda epoch: 0.5 ** (epoch // 2) if epoch < 11 else 1
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.configs.lr_scheduler == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return optimizer, scheduler

    def _save_checkpoint(self):
        model_path = os.path.join(self.exp_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path) 

    def _load_best_checkpoint(self):
        model_path = os.path.join(self.exp_dir, 'model.pt')
        self.load_checkpoint(model_path)


class EarlyStopping:

    def __init__(self, patience):
        self.patience = patience

        self.counter = None
        self.early_stop = None
        self.save_model = None
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        if self.val_loss_min == np.inf or val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss
            self.save_model = True
            self.counter = 0
        else:
            self.save_model = False
            self.counter += 1
            print(f'Validation loss didn\'t decrease ({self.counter} out of {self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True
