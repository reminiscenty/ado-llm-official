import random
import numpy as np
import torch


def create_empty_data_dict(key_list):
    data_dict = {}
    for key in key_list:
        data_dict[key] = []

    return data_dict

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_xy(data_collected, norm_func):
    lb_y = torch.tensor(data_collected["targets"], dtype=torch.double).view(-1, 1)
    lb_x = torch.tensor(data_collected["params_numpy"], dtype=torch.double)
    return norm_func(lb_x), lb_y