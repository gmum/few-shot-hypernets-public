from copy import deepcopy

import numpy as np
import torch
from torch import nn
from typing import Dict, Optional

from methods.kernels import NNKernel
from methods.meta_template import MetaTemplate


class PermutationHN(MetaTemplate):

    def __init__(self,
                 model_func,
                 n_way: int,
                 n_support: int,
                 target_net_architecture: Optional[nn.Module] = None):
        super().__init__(model_func, n_way, n_support)

        self.conv_out_size = self.feature.final_dim
        self.hn_hidden_size = 256
        self.tn_hidden_size = 128
        self.embedding_size = self.conv_out_size * n_way * n_support

        self.taskset_size = 1
        self.taskset_print_every = 20
        self.taskset_n_permutations = 1
        self.detach_support = False
        self.detach_query = False

        if target_net_architecture is None:
            self.target_net_architecture = nn.Sequential(
                nn.Linear(self.conv_out_size, self.tn_hidden_size),
                nn.ReLU(),
                nn.Linear(self.tn_hidden_size, self.n_way)
            )
        target_net_params = {
            name.replace(".", "-"): par for name, par in self.get_param_dict(target_net_architecture).items()
        }

        # create param dict
        self.target_net_params_shape = {
            name: par.shape for name, par in target_net_params.items()
        }

        self.target_net_param_predictors = nn.ModuleDict()
        for name, param in target_net_params.items():
            self.target_net_param_predictors[name] = nn.Sequential(
                nn.Linear(self.embedding_size, self.hn_hidden_size),
                nn.ReLU(),
                nn.Linear(self.hn_hidden_size, param.numel())
            )
        self.target_network_architecture = target_net_architecture
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def get_param_dict(net: nn.Module) -> Dict[str, nn.Parameter]:
        return {
            n: p
            for (n, p) in net.named_parameters()
        }

    @staticmethod
    def set_from_param_dict(net: nn.Module, param_dict: Dict[str, torch.Tensor]) -> None:
        for (sdk, v) in param_dict.items():
            keys = sdk.split(".")
            param_name = keys[-1]
            m = net
            for k in keys[:-1]:
                try:
                    k = int(k)
                    m = m[k]
                except:
                    m = getattr(m, k)

            param = getattr(m, param_name)
            assert param.shape == v.shape, (sdk, param.shape, v.shape)
            delattr(m, param_name)
            setattr(m, param_name, v)

    def taskset_epochs(self, progress_id: int):
        if progress_id > 30:
            return 1
        if progress_id > 20:
            return 2
        if progress_id > 10:
            return 5
        return 10

    def set_forward_loss(self, x: torch.Tensor):
        pass

    def set_forward(self, x: torch.Tensor, is_feature: bool):
        pass
