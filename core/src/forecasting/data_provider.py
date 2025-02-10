import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


def load_forecasting_dataset(configs, split, shuffle, interval_override=None):
    dataset = ForecastingDataset(
        os.path.join(configs.root_path, configs.data_path),
        [configs.seq_len, configs.label_len, configs.pred_len],
        configs.interval if interval_override is None else interval_override,
        split,
        task_type=configs.task_type,
        univariate=configs.univariate,
        date_col=configs.date_col,
        standardize=configs.standardize,
        enc_freq=configs.enc_freq,
    )
    loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=shuffle,
        num_workers=configs.num_workers
    )
    return loader

def get_forecasting_loader_and_dataset(configs, split, shuffle, interval_override=None):
    dataset = ForecastingDataset(
        os.path.join(configs.root_path, configs.data_path),
        [configs.seq_len, configs.label_len, configs.pred_len],
        configs.interval if interval_override is None else interval_override,
        split,
        task_type=configs.task_type,
        univariate=configs.univariate,
        date_col=configs.date_col,
        standardize=configs.standardize,
        enc_freq=configs.enc_freq,
    )
    loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=shuffle,
        num_workers=configs.num_workers
    )
    return loader, dataset

class ForecastingDataset(Dataset):

    def __init__(self, data_path, data_size, interval, split, task_type,
                 univariate=None, date_col='date', standardize=False, enc_freq=None):
        assert len(data_size) == 3
        assert split in {'train', 'val', 'test'}
        assert task_type in {'U', 'M'}
        assert isinstance(enc_freq, str) or enc_freq is None

        self.data_path = data_path
        self.seq_len, self.label_len, self.pred_len = data_size
        self.interval = interval
        self.split = split

        self.task_type = task_type
        self.univariate = univariate

        self.date_col = date_col
        self.standardize = standardize
        self.enc_freq = enc_freq

        self._load_data()

    def _load_data(self):
        df_raw = pd.read_csv(self.data_path)
        assert self.date_col in df_raw.columns
        assert (self.task_type == 'M' and self.univariate is None) or \
               (self.task_type == 'U' and self.univariate in df_raw.columns)


        if 'ETTh' in self.data_path:
            num_train = 12 * 30 * 24
            num_val = 4 * 30 * 24
            num_test = 4 * 30 * 24
        elif 'ETTm' in self.data_path:
            num_train = 12 * 30 * 24 * 4
            num_val = 4 * 30 * 24 * 4
            num_test = 4 * 30 * 24 * 4
        else:
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

        if self.task_type == 'M':
            df_data = df_raw.drop(self.date_col, axis=1)
        elif self.task_type == 'U':
            df_data = df_raw[[self.univariate]]

        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(df_data[:num_train].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.date_col]].astype(str).apply(pd.to_datetime)

        df_stamp['year'] = df_stamp[self.date_col].apply(lambda row: row.year)
        df_stamp['month'] = df_stamp[self.date_col].apply(lambda row: row.month)
        df_stamp['day'] = df_stamp[self.date_col].apply(lambda row: row.day)
        df_stamp['weekday'] = df_stamp[self.date_col].apply(lambda row: row.weekday())
        df_stamp['hour'] = df_stamp[self.date_col].apply(lambda row: row.hour)
        df_stamp['minute'] = df_stamp[self.date_col].apply(lambda row: row.minute)

        stamp = df_stamp.drop([self.date_col], axis=1).values

        self.data = data[start:end]
        self.stamp = stamp[start:end]
        self.num_vars = self.data.shape[1]

    def __len__(self):
        return (len(self.data)-self.seq_len-self.pred_len)//self.interval + 1

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError('negative index or index out of range')

        x_start = index * self.interval
        x_end = x_start + self.seq_len
        x_data = self.data[x_start:x_end].astype(np.float32)

        # `y_data` contains the last `self.label_len` tokens of `x_data` as start tokens
        # This feature is only used by encoder-decoder models
        # The actual prediction sequence should be sliced from `y_data` by [-self.pred_len:]
        y_start = x_end - self.label_len
        y_end = y_start + self.label_len + self.pred_len
        y_data = self.data[y_start:y_end].astype(np.float32)

        x_stamp = self.stamp[x_start:x_end].astype(np.float32)
        y_stamp = self.stamp[y_start:y_end].astype(np.float32)
        return x_data, x_stamp, y_data, y_stamp

    def inverse_transform(self, data):
        if not self.standardize:
            warnings.warn('Dataset not standardized, returning as is')
            return data

        return self.scaler.inverse_transform(data)
