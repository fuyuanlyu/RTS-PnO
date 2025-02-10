import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyepo.data.dataset import optDataset
from torch.utils.data import DataLoader

def get_allocating_loader_and_dataset(configs, optmodel, split, shuffle, interval_override=None):
    dataset = Allocatedataset(
        configs=configs, 
        optmodel=optmodel, 
        interval=configs.interval if interval_override is None else interval_override,
        split=split
    )
    loader = DataLoader(
        dataset.dataset, 
        batch_size=configs.batch_size, 
        shuffle=shuffle,
        num_workers=configs.num_workers
    )
    return loader, dataset

class Allocatedataset:
    def __init__(self, configs, optmodel, interval, split='train', date_col=['fdate', 'sec_in_fdate']):
        self.data_path = os.path.join(configs.root_path, configs.data_path)
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.interval = interval
        self.split = split
        
        self.task_type = configs.task_type
        self.univariate = configs.univariate

        self.date_col = configs.date_col
        self.standardize = configs.standardize

        self._load_data()

        data_x, data_y = [], []
        length = len(self)
        print(split, ":", length)
        for index in range(length):
            x, y = self._extract(index)
            data_x.append(x.squeeze())
            data_y.append(y.squeeze())

        data_x, data_y = np.array(data_x), np.array(data_y)
        self.dataset = optDataset(optmodel, data_x, data_y)

    def _load_data(self):
        df_raw = pd.read_csv(self.data_path)
        assert self.date_col in df_raw.columns
        assert (self.task_type == 'M' and self.univariate is None) or \
               (self.task_type == 'U' and self.univariate in df_raw.columns)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test

        train_end = num_train
        val_end = train_end + num_val
        test_end = val_end + num_test
        start, end = {
            'train': (0, train_end),
            'val': (train_end - self.seq_len, val_end),
            'test': (val_end - self.seq_len, test_end)
        }[self.split]

        # self.data = df_raw.drop(['fdate', 'sec_in_fdate', 'transact_time'], axis=1)
        df_data = df_raw[[self.univariate]]

        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(df_data[:num_train].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[self.date_col]
        stamp = df_stamp

        self.data = data[start:end]
        self.stamp = stamp[start:end]
        self.num_vars = self.data.shape[1]

    def __len__(self):
        return (len(self.data)-self.seq_len-self.pred_len)//self.interval + 1
    
    def _extract(self, index):
        if index < 0 or index >= len(self):
            raise IndexError('negative index or index out of range')

        x_start = index * self.interval
        x_end = x_start + self.seq_len
        x_data = self.data[x_start:x_end].astype(np.float32)
        
        y_start = x_end
        y_end = y_start + self.pred_len
        y_data = self.data[y_start:y_end].astype(np.float32)

        return x_data, y_data
    
    def inverse_transform(self, data):
        if not self.standardize:
            warnings.warn('Dataset not standardized, returning as is')
            return data

        return self.scaler.inverse_transform(data)
