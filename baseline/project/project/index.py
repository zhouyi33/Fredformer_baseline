# import os
# import numpy as np
# import random
# import torch
# from models import iTransformer
#
#
# def invoke(inputs):
#
#     cwd = os.path.dirname(inputs)
#     save_path = '/home/mw/project'
#
#     fix_seed = 2021
#     random.seed(fix_seed)
#     torch.manual_seed(fix_seed)
#     np.random.seed(fix_seed)
#
#     args = {
#         'model_id': 'v1',
#         'model': 'iTransformer',
#         'data': 'Meteorology',
#         'features': 'M',
#         'checkpoints': './checkpoints/',
#         'seq_len': 168,
#         'label_len': 1,
#         'pred_len': 24,
#         'enc_in': 37,
#         'd_model': 64,
#         'n_heads': 1,
#         'e_layers': 1,
#         'd_ff': 64,
#         'dropout': 0.1,
#         'activation': 'gelu',
#         'output_attention': False
#     }
#
#     class Struct:
#         def __init__(self, **entries):
#             self.__dict__.update(entries)
#     args = Struct(**args)
#
#     test_data_root_path = inputs
#     for i in range(2):
#         if i == 0:
#             data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy")) # (N, L, S, 1)
#         else:
#             data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
#         N, L, S, _ = data.shape # 72, 168, 60
#
#
#         cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy"))
#
#         repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1) # (N, L, 4, 9, S)
#         C1 = repeat_era5.shape[2] * repeat_era5.shape[3]
#         covariate = repeat_era5.reshape(repeat_era5.shape[0], repeat_era5.shape[1], -1, repeat_era5.shape[4]) # (N, L, C1, S)
#         data = data.transpose(0, 1, 3, 2) # (N, L, 1, S)
#         C = C1 + 1
#         data = np.concatenate([covariate, data], axis=2) # (N, L, C, S)
#         data = data.transpose(0, 3, 1, 2) # (N, S, L, C)
#         data = data.reshape(N * S, L, C)
#         data = torch.tensor(data).float().cuda() # (N * S, L, C)
#
#         model = iTransformer.Model(args).cuda()
#
#         if i == 0:
#             model.load_state_dict(torch.load("/home/mw/project/checkpoints/v1_iTransformer_Meteorology_ftMS_sl168_ll1_pl24_dm64_nh1_el1_df64_global_temp/checkpoint_0.pth"))
#         else:
#             model.load_state_dict(torch.load("/home/mw/project/checkpoints/v1_iTransformer_Meteorology_ftMS_sl168_ll1_pl24_dm64_nh1_el1_df64_global_wind/checkpoint_0.pth"))
#         outputs = model(data)
#         outputs = outputs[:, :, -1:].detach().cpu().numpy() # (N * S, P, 1)
#         P = outputs.shape[1]
#         forecast = outputs.reshape(N, S, P, 1) # (N, S, P, 1)
#         forecast = forecast.transpose(0, 2, 1, 3) # (N, P, S, 1)
#         if i == 0:
#             np.save(os.path.join(save_path, "temp_predict.npy"), forecast)
#         else:
#             np.save(os.path.join(save_path, "wind_predict.npy"), forecast)
#
#
import os
import numpy as np
import random
import torch
from models import iTransformer
from config import MockPath
from scipy.interpolate import CubicSpline


def invoke(inputs):
    cwd = os.path.dirname(inputs)
    save_path = '/home/mw/project'

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args_wind = {
        'model_id': 'v1',
        'model': 'iTransformer',
        'data': 'Meteorology',
        'features': 'M',
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 1,
        'pred_len': 24,
        'enc_in': 37,
        'd_model': 256,
        'n_heads': 8,
        'e_layers': 2,
        'd_ff': 1024,
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False
    }

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    args_wind = Struct(**args_wind)

    ## -----
    args_temp = {
        'model_id': 'v1',
        'model': 'iTransformer',
        'data': 'Meteorology',
        'features': 'M',
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 1,
        'pred_len': 24,
        'enc_in': 37,
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_ff': 1024,
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False
    }

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    args_temp = Struct(**args_temp)

    ## ----

    test_data_root_path = inputs
    for i in range(2):
        if i == 0:
            data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy"))  # (N, L, S, 1)
        else:
            data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
        N, L, S, _ = data.shape  # 72, 168, 60

        print('repeat start!', flush=True)

        cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy"))

        repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1)  # (N, L, 4, 9, S)

        # T = cenn_era5_data.shape[1]
        #
        # # 创建一个空数组来存放插值结果
        # new_T = T * 3  # 新的 T 维度为原来的 3 倍
        # interpolated_data = repeat_era5
        # print('interpolate start!', flush=True)
        #
        # # 进行插值
        # for n in range(N):
        #     for t in range(4):
        #         for s in range(S):
        #             for iii in range(9):
        #                 # 创建三次样条插值函数
        #                 cs = CubicSpline(np.arange(T), cenn_era5_data[n, :, t, iii, s])
        #                 # 插值到新的点
        #                 interpolated_data[n, :, t, iii, s] = cs(np.linspace(0, T - 1, new_T))
        #
        # ##########
        # repeat_era5 = interpolated_data
        # print('{}:interpolated!'.format(repeat_era5.shape), flush=True)

        C1 = repeat_era5.shape[2] * repeat_era5.shape[3]
        covariate = repeat_era5.reshape(repeat_era5.shape[0], repeat_era5.shape[1], -1,
                                        repeat_era5.shape[4])  # (N, L, C1, S)
        data = data.transpose(0, 1, 3, 2)  # (N, L, 1, S)
        C = C1 + 1
        data = np.concatenate([covariate, data], axis=2)  # (N, L, C, S)
        data = data.transpose(0, 3, 1, 2)  # (N, S, L, C)
        data = data.reshape(N * S, L, C)
        data = torch.tensor(data).float().cuda()  # (N * S, L, C)

        if i == 0:
            model = iTransformer.Model(args_temp).cuda()
            model.load_state_dict(torch.load("/home/mw/project/checkpoints/checkpoint_temp.pth"))
        else:
            model = iTransformer.Model(args_wind).cuda()
            model.load_state_dict(torch.load("/home/mw/project/checkpoints/checkpoint_wind.pth"))
        outputs = model(data)
        outputs = outputs[:, :, -1:].detach().cpu().numpy()  # (N * S, P, 1)
        P = outputs.shape[1]
        forecast = outputs.reshape(N, S, P, 1)  # (N, S, P, 1)
        forecast = forecast.transpose(0, 2, 1, 3)  # (N, P, S, 1)
        if i == 0:
            np.save(os.path.join(save_path, "temp_predict.npy"), forecast)
        else:
            np.save(os.path.join(save_path, "wind_predict.npy"), forecast)
