import os
import numpy as np
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features


class Dataset_Meteorology(Dataset):
    def __init__(self, root_path, data_path, size=None, features='MS', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.stations_num = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        # Load primary data
        data = np.load(os.path.join(self.root_path, self.data_path))  # (T, S, 1)
        data = np.squeeze(data)  # (T, S)

        if self.scale:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

        # Use entire data for training
        self.data_x = data
        self.data_y = self.data_x  # Assuming target is the same as input

        # Load additional covariate data
        era5 = np.load(os.path.join(self.root_path, 'global_data.npy'))
        repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data), :, :, :]  # (T, 4, 9, S)
        repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[-1])  # (T, 36, S)
        self.covariate = repeat_era5

        # Process time features
        df_stamp = pd.DataFrame(index=pd.date_range(start='1/1/2000', periods=len(data), freq=self.freq))
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.index.month
            df_stamp['day'] = df_stamp.index.day
            df_stamp['weekday'] = df_stamp.index.weekday
            df_stamp['hour'] = df_stamp.index.hour
            self.data_stamp = df_stamp.values
        elif self.timeenc == 1:
            self.data_stamp = time_features(df_stamp.index, freq=self.freq).transpose(1, 0)

    def __getitem__(self, index):
        station_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, station_id:station_id + 1]
        seq_y = self.data_y[r_begin:r_end, station_id:station_id + 1]

        t1 = self.covariate[s_begin:s_end, :, station_id:station_id + 1].squeeze()
        t2 = self.covariate[r_begin:r_end, :, station_id:station_id + 1].squeeze()

        seq_x = np.concatenate([t1, seq_x], axis=1)
        seq_y = np.concatenate([t2, seq_y], axis=1)

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.stations_num

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
