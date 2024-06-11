import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
#from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import random

warnings.filterwarnings('ignore')

class Dataset_SiteA(Dataset):
    def __init__(self, root_path='./dataset',
                 # data_path='SiteB_MayToOct.csv',
                 data_path='SiteA_Jan22toJun22.csv',
                 flag='train',
                 scale=True,
                 size=(12, 6, 12),
                 r_temp_split=(0.8, 0.2, 1.0),
                 # r_spa_split=(0.7, 0.1, 0.2),
                 features='M',
                 pretrain=False,
                 **kwargs):

        self.flag = flag

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path

        seq_len, label_len, pred_len = size

        self.r_temp_split = r_temp_split
        # self.r_spa_split = r_spa_split

        self.features = features

        self.historical_window = seq_len
        self.forecasting_horizon = pred_len
        self.label_len = label_len
        self._load_data()


    def _split(self, data, axis=1, r_trn=.8, r_val=.2, r_tst=1., seed=2024, replace=False, shuffle=False):
        '''
            axis: 0 for temporal split, 1 for spatial split
            return: index of train, val, test
        '''
        n_dp = data.shape[axis]
        random.Random(seed).seed(seed)
        if not replace:
            idx = np.arange(n_dp)
            if shuffle:
                np.random.shuffle(idx)
            i_trn = idx[:int(n_dp * r_trn)]
            i_val = idx[int(n_dp * r_trn):int(n_dp * (r_trn + r_val))]
            # i_tst = idx[int(n_dp * (r_trn + r_val)):int(n_dp * (r_trn + r_val + r_tst))]
            i_tst = idx
        else:
            i_trn = np.random.choice(n_dp, size=int(n_dp * r_trn), replace=False)
            i_val = np.random.choice(n_dp, size=int(n_dp * r_val), replace=False)
            i_tst = np.random.choice(n_dp, size=int(n_dp * r_tst), replace=False)
        return i_trn, i_val, i_tst


    def _load_data(self):

        df = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col=0)
        df_stamp = pd.DataFrame(pd.to_datetime(df.index), columns=['TimeStamp'])
        self.scaler = StandardScaler()
        sdf = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)

        df_stamp['month'] = df_stamp.TimeStamp.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.TimeStamp.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.TimeStamp.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.TimeStamp.apply(lambda row: row.hour, 1)
        self.data_stamp = df_stamp.drop(columns=['TimeStamp'])

        temp_idx_trn, temp_idx_val, temp_idx_tst = self._split(sdf, axis=0,
                                                               r_trn=self.r_temp_split[0],
                                                               r_val=self.r_temp_split[1],
                                                               r_tst=self.r_temp_split[-1])

        self.trn = sdf.iloc[temp_idx_trn, :]
        self.val = sdf.iloc[temp_idx_val, :]
        self.tst = sdf.iloc[temp_idx_tst, :]

    def _get_item(self, data, index):
        data = np.float32(data)
        if self.features == 'M':
            s_begin = index
            s_end = s_begin + self.historical_window
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.forecasting_horizon

            seq_x = data[s_begin:s_end]
            seq_y = data[r_begin:r_end]

            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        elif self.features == 'S':
            # # print(data.shape)
            s_begin = index // data.shape[1]
            s_end = s_begin + self.historical_window
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.forecasting_horizon
            # print(index, s_begin, s_end)

            seq_x = data[s_begin:s_end, index % data.shape[1]]
            seq_y = data[r_begin:r_end, index % data.shape[1]]
            # print(seq_x.shape, seq_y.shape)

            seq_x = seq_x[:, np.newaxis]
            seq_y = seq_y[:, np.newaxis]

            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        else:
            raise NotImplementedError('Only support M and S features')

        return np.float32(seq_x), np.float32(seq_y), np.float32(seq_x_mark), np.float32(seq_y_mark)


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def _return_psample(self, data, index):
        # print(self.flag, data.shape, index)
        sample = data.T.iloc[index, :]
        return np.float32(sample)

    def _get_all_samples(self):
        if self.flag == 'train':
            return np.float32(self.trn.T)
        elif self.flag == 'val':
            return np.float32(self.val.T)
        else:
            return np.float32(self.tst.T)

    def __getitem__(self, index):
        if self.flag == 'train':
            return self._get_item(self.trn, index)
        elif self.flag == 'val':
            return self._get_item(self.val, index)
        else:
            return self._get_item(self.tst, index)
    def __len__(self):
        if self.features == 'M':
            if self.flag == 'train':
                return self.trn.shape[0] - self.historical_window - self.forecasting_horizon + 1
            if self.flag == 'val':
                return self.val.shape[0] - self.historical_window - self.forecasting_horizon + 1
            else:
                return self.tst.shape[0] - self.historical_window - self.forecasting_horizon + 1
        if self.features == 'S':
            if self.flag == 'train':
                return (self.trn.shape[0] - self.historical_window - self.forecasting_horizon + 1) * self.trn.shape[1]
            if self.flag == 'val':
                return (self.val.shape[0] - self.historical_window - self.forecasting_horizon + 1) * self.val.shape[1]
            else:
                return (self.tst.shape[0] - self.historical_window - self.forecasting_horizon + 1) * self.tst.shape[1]
class Dataset_SiteB(Dataset):
    def __init__(self, root_path='./dataset',
                 # data_path='SiteB_MayToOct.csv',
                 data_path='SiteB_Jan22toJun22.csv',
                 flag='train',
                 scale=True,
                 size=(12, 6, 12),
                 r_temp_split=(0.8, 0.2, 1.0),
                 # r_spa_split=(0.7, 0.1, 0.2),
                 features='M',
                 pretrain=False,
                 **kwargs):

        self.flag = flag

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path

        seq_len, label_len, pred_len = size

        self.r_temp_split = r_temp_split
        # self.r_spa_split = r_spa_split

        self.features = features

        self.historical_window = seq_len
        self.forecasting_horizon = pred_len
        self.label_len = label_len
        self._load_data()


    def _split(self, data, axis=1, r_trn=.8, r_val=.2, r_tst=1., seed=2024, replace=False, shuffle=False):
        '''
            axis: 0 for temporal split, 1 for spatial split
            return: index of train, val, test
        '''
        n_dp = data.shape[axis]
        random.Random(seed).seed(seed)
        if not replace:
            idx = np.arange(n_dp)
            if shuffle:
                np.random.shuffle(idx)
            i_trn = idx[:int(n_dp * r_trn)]
            i_val = idx[int(n_dp * r_trn):int(n_dp * (r_trn + r_val))]
            # i_tst = idx[int(n_dp * (r_trn + r_val)):int(n_dp * (r_trn + r_val + r_tst))]
            i_tst = idx
        else:
            i_trn = np.random.choice(n_dp, size=int(n_dp * r_trn), replace=False)
            i_val = np.random.choice(n_dp, size=int(n_dp * r_val), replace=False)
            i_tst = np.random.choice(n_dp, size=int(n_dp * r_tst), replace=False)
        return i_trn, i_val, i_tst


    def _load_data(self):

        df = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col=0)
        df_stamp = pd.DataFrame(pd.to_datetime(df.index), columns=['TimeStamp'])
        self.scaler = StandardScaler()
        sdf = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)

        df_stamp['month'] = df_stamp.TimeStamp.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.TimeStamp.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.TimeStamp.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.TimeStamp.apply(lambda row: row.hour, 1)
        self.data_stamp = df_stamp.drop(columns=['TimeStamp'])

        temp_idx_trn, temp_idx_val, temp_idx_tst = self._split(sdf, axis=0,
                                                               r_trn=self.r_temp_split[0],
                                                               r_val=self.r_temp_split[1],
                                                               r_tst=self.r_temp_split[-1])

        self.trn = sdf.iloc[temp_idx_trn, :]
        self.val = sdf.iloc[temp_idx_val, :]
        self.tst = sdf.iloc[temp_idx_tst, :]

    def _get_item(self, data, index):
        data = np.float32(data)
        if self.features == 'M':
            s_begin = index
            s_end = s_begin + self.historical_window
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.forecasting_horizon

            seq_x = data[s_begin:s_end]
            seq_y = data[r_begin:r_end]

            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        elif self.features == 'S':
            # # print(data.shape)
            s_begin = index // data.shape[1]
            s_end = s_begin + self.historical_window
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.forecasting_horizon
            # print(index, s_begin, s_end)

            seq_x = data[s_begin:s_end, index % data.shape[1]]
            seq_y = data[r_begin:r_end, index % data.shape[1]]
            # print(seq_x.shape, seq_y.shape)

            seq_x = seq_x[:, np.newaxis]
            seq_y = seq_y[:, np.newaxis]

            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        else:
            raise NotImplementedError('Only support M and S features')

        return np.float32(seq_x), np.float32(seq_y), np.float32(seq_x_mark), np.float32(seq_y_mark)


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def _return_psample(self, data, index):
        # print(self.flag, data.shape, index)
        sample = data.T.iloc[index, :]
        return np.float32(sample)

    def _get_all_samples(self):
        if self.flag == 'train':
            return np.float32(self.trn.T)
        elif self.flag == 'val':
            return np.float32(self.val.T)
        else:
            return np.float32(self.tst.T)

    def __getitem__(self, index):
        if self.flag == 'train':
            return self._get_item(self.trn, index)
        elif self.flag == 'val':
            return self._get_item(self.val, index)
        else:
            return self._get_item(self.tst, index)
    def __len__(self):
        if self.features == 'M':
            if self.flag == 'train':
                return self.trn.shape[0] - self.historical_window - self.forecasting_horizon + 1
            if self.flag == 'val':
                return self.val.shape[0] - self.historical_window - self.forecasting_horizon + 1
            else:
                return self.tst.shape[0] - self.historical_window - self.forecasting_horizon + 1
        if self.features == 'S':
            if self.flag == 'train':
                return (self.trn.shape[0] - self.historical_window - self.forecasting_horizon + 1) * self.trn.shape[1]
            if self.flag == 'val':
                return (self.val.shape[0] - self.historical_window - self.forecasting_horizon + 1) * self.val.shape[1]
            else:
                return (self.tst.shape[0] - self.historical_window - self.forecasting_horizon + 1) * self.tst.shape[1]

class Dataset_SiteC(Dataset):
    def __init__(self, root_path='./dataset',
                 # data_path='SiteB_MayToOct.csv',
                 data_path='SiteC_Jan22toJun22.csv',
                 flag='train',
                 scale=True,
                 size=(12, 6, 12),
                 r_temp_split=(0.8, 0.2, 1.0),
                 # r_spa_split=(0.7, 0.1, 0.2),
                 features='M',
                 pretrain=False,
                 **kwargs):

        self.flag = flag

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path

        seq_len, label_len, pred_len = size

        self.r_temp_split = r_temp_split
        # self.r_spa_split = r_spa_split

        self.features = features

        self.historical_window = seq_len
        self.forecasting_horizon = pred_len
        self.label_len = label_len
        self._load_data()


    def _split(self, data, axis=1, r_trn=.8, r_val=.2, r_tst=1., seed=2024, replace=False, shuffle=False):
        '''
            axis: 0 for temporal split, 1 for spatial split
            return: index of train, val, test
        '''
        n_dp = data.shape[axis]
        random.Random(seed).seed(seed)
        if not replace:
            idx = np.arange(n_dp)
            if shuffle:
                np.random.shuffle(idx)
            i_trn = idx[:int(n_dp * r_trn)]
            i_val = idx[int(n_dp * r_trn):int(n_dp * (r_trn + r_val))]
            # i_tst = idx[int(n_dp * (r_trn + r_val)):int(n_dp * (r_trn + r_val + r_tst))]
            i_tst = idx
        else:
            i_trn = np.random.choice(n_dp, size=int(n_dp * r_trn), replace=False)
            i_val = np.random.choice(n_dp, size=int(n_dp * r_val), replace=False)
            i_tst = np.random.choice(n_dp, size=int(n_dp * r_tst), replace=False)
        return i_trn, i_val, i_tst


    def _load_data(self):

        df = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col=0)
        df_stamp = pd.DataFrame(pd.to_datetime(df.index), columns=['TimeStamp'])
        self.scaler = StandardScaler()
        sdf = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)

        df_stamp['month'] = df_stamp.TimeStamp.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.TimeStamp.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.TimeStamp.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.TimeStamp.apply(lambda row: row.hour, 1)
        self.data_stamp = df_stamp.drop(columns=['TimeStamp'])

        temp_idx_trn, temp_idx_val, temp_idx_tst = self._split(sdf, axis=0,
                                                               r_trn=self.r_temp_split[0],
                                                               r_val=self.r_temp_split[1],
                                                               r_tst=self.r_temp_split[-1])

        self.trn = sdf.iloc[temp_idx_trn, :]
        self.val = sdf.iloc[temp_idx_val, :]
        self.tst = sdf.iloc[temp_idx_tst, :]

    def _get_item(self, data, index):
        data = np.float32(data)
        if self.features == 'M':
            s_begin = index
            s_end = s_begin + self.historical_window
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.forecasting_horizon

            seq_x = data[s_begin:s_end]
            seq_y = data[r_begin:r_end]

            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        elif self.features == 'S':
            # # print(data.shape)
            s_begin = index // data.shape[1]
            s_end = s_begin + self.historical_window
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.forecasting_horizon
            # print(index, s_begin, s_end)

            seq_x = data[s_begin:s_end, index % data.shape[1]]
            seq_y = data[r_begin:r_end, index % data.shape[1]]
            # print(seq_x.shape, seq_y.shape)

            seq_x = seq_x[:, np.newaxis]
            seq_y = seq_y[:, np.newaxis]

            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        else:
            raise NotImplementedError('Only support M and S features')

        return np.float32(seq_x), np.float32(seq_y), np.float32(seq_x_mark), np.float32(seq_y_mark)


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def _return_psample(self, data, index):
        # print(self.flag, data.shape, index)
        sample = data.T.iloc[index, :]
        return np.float32(sample)

    def _get_all_samples(self):
        if self.flag == 'train':
            return np.float32(self.trn.T)
        elif self.flag == 'val':
            return np.float32(self.val.T)
        else:
            return np.float32(self.tst.T)

    def __getitem__(self, index):
        if self.flag == 'train':
            return self._get_item(self.trn, index)
        elif self.flag == 'val':
            return self._get_item(self.val, index)
        else:
            return self._get_item(self.tst, index)
    def __len__(self):
        if self.features == 'M':
            if self.flag == 'train':
                return self.trn.shape[0] - self.historical_window - self.forecasting_horizon + 1
            if self.flag == 'val':
                return self.val.shape[0] - self.historical_window - self.forecasting_horizon + 1
            else:
                return self.tst.shape[0] - self.historical_window - self.forecasting_horizon + 1
        if self.features == 'S':
            if self.flag == 'train':
                return (self.trn.shape[0] - self.historical_window - self.forecasting_horizon + 1) * self.trn.shape[1]
            if self.flag == 'val':
                return (self.val.shape[0] - self.historical_window - self.forecasting_horizon + 1) * self.val.shape[1]
            else:
                return (self.tst.shape[0] - self.historical_window - self.forecasting_horizon + 1) * self.tst.shape[1]
