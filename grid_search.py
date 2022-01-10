from argparse import PARSER


from typing import *
import numpy as np


class AttributeDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


# PARAMS = {
#     'lr': [2e-5, 5e-6, 1e-6, 5e-7],
#     'hn_hidden_size': [1024, 2048],
#     'detach_ft_in_hn' : [10000, 5000, 7500],
#     'detach_ft_in_tn': [10000, 5000, 7500],
#     'hn_neck_len': [0],
#     'hn_head_len': [1, 2, 3, 4],
#     'taskset_repeats_config': ['10:10-20:5-30:2'],
#     'hn_ln': [True, False],
#     'hn_dropout': [0.12, 0.4, 0.6],
#     'hn_val_epochs': [0, 5, 10, 15],
#     'hn_val_lr': [2e-5, 5e-6, 1e-6, 5e-7],
#     'hn_val_optim': ['adam', 'sgd']
# }


PARAMS = AttributeDict()
PARAMS.lr = [1e-4, 5e-5, 1e-5]
PARAMS.taskset_size = [1, 3, 8, 10]
PARAMS.taskset_print_every = [20]
PARAMS.hn_attention_embedding = [
    True, 
    False
]
PARAMS.hn_hidden_size = [
    256, 
    #512, 
    #768
]
PARAMS.attention_embedding = [
    True, 
    # False
]
PARAMS.detach_ft_in_hn = [
    7000, 
    # 10000,
    #12000
]
PARAMS.detach_ft_in_tn = [
    7000, 
    #10000, 
    #12000
]
PARAMS.hn_neck_len = [0]
PARAMS.hn_head_len = [
    1, 
    2, 
    3, 
    4]
PARAMS.taskset_repeats_config = ['10:10-20:5-30:2']
PARAMS.hn_dropout = [
    0.01, 
    0.4, 
    0.6
]


def get_random_parameters() -> AttributeDict:
    chosen_params = {}
    for key in PARAMS.keys():
        chosen_params[key] = np.random.choice(PARAMS[key], 1)[0]
    return AttributeDict(chosen_params)


if __name__ == '__main__':
    print(get_random_parameters)
