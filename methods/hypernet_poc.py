import abc
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from typing import Dict, Optional

from methods.kernels import NNKernel
from methods.meta_template import MetaTemplate


class HyperNetPOC(MetaTemplate):
    def __init__(self, model_func, n_way: int, n_support: int, target_net_architecture: Optional[nn.Module] = None):
        super().__init__(model_func, n_way, n_support)

        conv_out_size = self.feature.final_feat_dim
        hn_hidden_size = 256
        tn_hidden_size = 128
        embedding_size = conv_out_size * self.n_way * self.n_support

        self.taskset_size = 1
        self.taskset_print_every = 20
        self.taskset_n_permutations = 1
        self.conv_out_size = conv_out_size
        self.embedding_size = embedding_size
        self.detach_support = False
        self.detach_query = False

        # TODO #1 - tweak the architecture of the target network
        target_net_architecture = target_net_architecture or nn.Sequential(
            nn.Linear(conv_out_size, tn_hidden_size),
            nn.ReLU(),
            nn.Linear(tn_hidden_size, self.n_way)
        )

        target_net_param_dict = get_param_dict(target_net_architecture)
        target_net_param_dict = {
            name.replace(".", "-"): p
            # replace dots with hyphens bc torch doesn't like dots in modules names
            for name, p in target_net_param_dict.items()
        }
        self.target_net_param_shapes = {
            name: p.shape
            for (name, p)
            in target_net_param_dict.items()
        }

        # TODO 2 - tweak parameter predictors
        self.target_net_param_predictors = nn.ModuleDict()
        for name, param in target_net_param_dict.items():
            self.target_net_param_predictors[name] = nn.Sequential(
                nn.Linear(embedding_size, hn_hidden_size),
                nn.ReLU(),
                nn.Linear(hn_hidden_size, param.numel())
            )
        self.target_network_architecture = target_net_architecture
        self.loss_fn = nn.CrossEntropyLoss()

    def taskset_epochs(self, progress_id: int):
        # TODO - initial bootstrapping - is this essential?
        if progress_id > 30:
            return 1
        if progress_id > 20:
            return 2
        if progress_id > 10:
            return 5
        return 10

    def get_labels(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [n_way, n_shot, hidden_size]
        """
        ys = torch.tensor(list(range(x.shape[0]))).reshape(len(x), 1)
        ys = ys.repeat(1, x.shape[1]).to(x.device)
        return ys.cuda()

    def build_embedding(self, support_feature: torch.Tensor) -> torch.Tensor:
        way, n_support, feat = support_feature.shape
        features = support_feature.reshape(way * n_support, feat)
        features = features.reshape(1, -1)
        return features

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        embedding = self.build_embedding(support_feature)
        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.target_net_param_predictors.items()
        }
        tn = deepcopy(self.target_network_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()

    def set_forward(self, x: torch.Tensor, is_feature: bool = False):
        support_feature, query_feature = self.parse_feature(x, is_feature)

        classifier = self.generate_target_net(support_feature)
        query_feature = query_feature.reshape(
            -1, query_feature.shape[-1]
        )
        y_pred = classifier(query_feature)
        return y_pred

    def query_accuracy(self, x: torch.Tensor):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        correct_this = float(top1_correct)
        count_this = len(y_query)

        return correct_this / count_this

    def get_second_support_query(
            self, support_feature: torch.Tensor, query_feature: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        det_support_feature = support_feature.detach().clone() if self.detach_support else support_feature
        det_query_feature = query_feature.detach().clone() if self.detach_query else query_feature
        return det_support_feature, det_query_feature

    def transform_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_all_labels(self, x: torch.Tensor):
        nw, ne, c, h, w = x.shape
        support_feature, query_feature = self.parse_feature(x, is_feature=False)

        det_support_feature, det_query_feature = self.get_second_support_query(support_feature, query_feature)

        all_feture = torch.cat(
            [
                support_feature.reshape(
                    (self.n_way * self.n_support), det_support_feature.shape[-1]
                ),
                query_feature.reshape(
                    (self.n_way * (ne - self.n_support)), det_query_feature.shape[-1]
                )
            ])
        #print(all_feture.shape)
        all_feature = self.transform_embeddings(
            all_feture
        )
        #print(all_feture.shape)

        y_support = self.get_labels(support_feature)
        y_query = self.get_labels(query_feature)
        all_y = torch.cat([
            y_support.reshape(self.n_way * self.n_support),
            y_query.reshape(self.n_way * (ne - self.n_support))
        ])
        return support_feature, query_feature, all_feature, all_y

    def set_forward_loss(self, x: torch.Tensor) -> torch.Tensor:
        # create all necessary items
        support_feature, query_feature, all_feature, all_y = self.get_all_labels(x)
        # create target net
        classifier = self.generate_target_net(support_feature)
        # predict
        y_pred = classifier(all_feature)
        # return loss
        return self.loss_fn(y_pred, all_y)

    def train_loop(self, epoch, train_loader, optimizer):

        taskset_id = 0
        taskset = []
        n_train = len(train_loader)
        accuracies = []
        for i, (x, _) in enumerate(train_loader):
            taskset.append(x)

            # TODO 3: perhaps the idea of tasksets is redundant and it's better to update weights at every task
            if i % self.taskset_size == (self.taskset_size - 1) or i == (n_train - 1):
                ts_epochs = self.taskset_epochs(epoch)
                loss_sum = torch.tensor(0).cuda()
                for e in range(ts_epochs):
                    loss_sum = torch.tensor(0).cuda()

                    for task_x in taskset:
                        if self.change_way:
                            self.n_way = task_x.size(0)
                        self.n_query = task_x.size(1) - self.n_support

                        # for _ in range(self.taskset_n_permutations):
                        loss = self.set_forward_loss(task_x)
                        loss_sum = loss_sum + loss

                    loss_sum.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                accuracies.extend([
                    self.query_accuracy(task_x) for task_x in taskset
                ])
                acc_mean = np.mean(accuracies) * 100
                acc_std = np.std(accuracies) * 100

                if taskset_id % self.taskset_print_every == 0:
                    print(
                        f"Epoch {epoch} | Taskset {taskset_id} | TS {len(taskset)} | TS epochs {ts_epochs} | Loss {loss_sum.item()} | Train acc {acc_mean:.2f} +- {acc_std:.2f} %")

                taskset_id += 1
                taskset = []


def get_param_dict(net: nn.Module) -> Dict[str, nn.Parameter]:
    return {
        n: p
        for (n, p) in net.named_parameters()
    }


def set_from_param_dict(net: nn.Module, param_dict: Dict[str, torch.Tensor]):
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


class HyperNetConvFromDKT(HyperNetPOC):
    def __init__(self, model_func, n_way: int, n_support: int):
        super().__init__(model_func, n_way, n_support)
        self.feature.trunk.add_module("bn_out", nn.BatchNorm1d(self.feature.final_feat_dim))

        dkt_state_dict = torch.load(
            "save/checkpoints/cross_char/Conv4_DKT_5way_5shot/best_model.tar",
        )

        state = dkt_state_dict["state"]
        state = {
            k: v
            for (k, v)
            in state.items()
            if k.startswith("feature.")
        }
        self.load_state_dict(state, strict=False)


class HyperNetSepJoint(HyperNetPOC):
    def __init__(self, model_func, n_way: int, n_support: int):
        super().__init__(model_func, n_way, n_support)

        self.support_individual_processor = nn.Sequential(
            nn.Linear(self.conv_out_size, self.conv_out_size),
            nn.ReLU(),
            # nn.Linear(self.conv_out_size, self.conv_out_size)
        )

        joint_size = self.conv_out_size * self.n_way
        self.support_joint_processor = nn.Sequential(
            nn.Linear(joint_size, joint_size),
            nn.ReLU(),
            # nn.Linear(joint_size, joint_size)
        )

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """
        way, shot, feat = support_feature.shape
        support_feature_transposed = torch.transpose(support_feature, 0, 1)

        sft_flat = support_feature_transposed.reshape(shot * way, feat)

        sft_separate = self.support_individual_processor(sft_flat)
        sft_joint = sft_separate.reshape(shot, way * feat)
        sft_joint = self.support_joint_processor(sft_joint)

        embedding = sft_joint.reshape(1, way * shot * feat)

        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.target_net_param_predictors.items()
        }
        tn = deepcopy(self.target_network_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()


class HyperNetSupportConv(HyperNetPOC):
    def __init__(self, model_func, n_way: int, n_support: int):
        super().__init__(model_func, n_way, n_support)

        self.feat_size = self.conv_out_size + self.n_way

        self.support_conv_processor = nn.Sequential(
            nn.Conv2d(self.feat_size, self.feat_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.feat_size, self.feat_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.feat_size, self.feat_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.feat_size, self.conv_out_size, kernel_size=3, padding=1),

        )

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """
        way, shot, feat = support_feature.shape
        logits = torch.zeros(way, shot, way).cuda()

        for c in range(way):
            logits[c, torch.arange(shot), c] = 1

        support_feature = torch.cat([logits, support_feature], 2)
        sf = support_feature.permute(2, 0, 1).unsqueeze(0)
        sf_conved = self.support_conv_processor(sf)

        sf_conved = sf_conved.squeeze().permute(1, 2, 0)
        embedding = self.build_embedding(sf_conved)
        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.target_net_param_predictors.items()
        }
        tn = deepcopy(self.target_network_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()


class HyperNetSupportKernel(HyperNetPOC):
    def __init__(self, model_func, n_way: int, n_support: int):
        super().__init__(model_func, n_way, n_support)

        self.kernel = NNKernel(
            input_dim=self.conv_out_size,
            output_dim=self.conv_out_size,
            num_layers=1,
            hidden_dim=self.conv_out_size
        )

        self.kernel_to_embedding = nn.Sequential(
            nn.Linear((n_way * n_support) ** 2, self.embedding_size)
        )

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        way, shot, feat = support_feature.shape
        sf = support_feature.reshape(way * shot, feat)

        sf_k = self.kernel(sf, sf).evaluate()

        sf_k = sf_k.view(1, (way * shot) ** 2)
        embedding = self.kernel_to_embedding(sf_k)

        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.target_net_param_predictors.items()
        }
        tn = deepcopy(self.target_network_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()


class KernelWithSupportClassifier(nn.Module):
    def __init__(self, kernel: NNKernel, support: Optional[torch.Tensor] = None):
        super().__init__()
        self.kernel = kernel
        self.support = support

    def forward(self, query: torch.Tensor):
        way, n_support, feat = self.support.shape
        n_query, feat = query.shape

        similarities = self.kernel.forward(
            self.support.reshape(way * n_support, feat)
            , query)

        res = similarities.reshape(way, n_support, n_query).sum(axis=1).T
        res = res - res.mean(axis=0)

        return res


class HNKernelBetweenSupportAndQuery(HyperNetPOC):
    def __init__(self, model_func, n_way: int, n_support: int):
        target_net_architecture = KernelWithSupportClassifier(NNKernel(64, 16, 1, 128))

        super().__init__(
            model_func, n_way, n_support, target_net_architecture=target_net_architecture)

    def taskset_epochs(self, progress_id: int):
        return 1

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        way, n_support, feat = support_feature.shape

        embedding = self.build_embedding(support_feature)
        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.target_net_param_predictors.items()
        }
        tn: KernelWithSupportClassifier = deepcopy(self.target_network_architecture)
        set_from_param_dict(tn, network_params)
        tn.support = support_feature
        return tn.cuda()


class PointNet(nn.Module):

    def __init__(self, sampling, z_size, channels, batch_norm):
        super().__init__()
        self.sampling = sampling
        self.z_size = z_size
        self.channels = channels
        if batch_norm:
            self._conv1 = nn.Sequential(
                nn.Conv1d(self.channels, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
        else:
            self._conv1 = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.ReLU()
            )

        if batch_norm:
            self._conv2 = nn.Sequential(
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                nn.Conv1d(128, z_size, 1),
                nn.BatchNorm1d(z_size),
                nn.ReLU()
            )
        else:
            self._conv2 = nn.Sequential(
                # nn.Conv1d(64, 128, 1),
                # nn.ReLU(),

                nn.Conv1d(64, z_size, 1),
                nn.ReLU()
            )

        self._pool = nn.MaxPool1d(sampling)
        self._flatten = nn.Flatten(1)

    def forward(self, x):
        x = x.view(-1, 1, self.sampling)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._pool(x)
        return self._flatten(x)


class AbstractPointNetHN(abc.ABC):
    pn = PointNet(
        sampling=64,
        z_size=64,
        channels=1,
        batch_norm=True
    )
    pn = pn.cuda()

    def transform_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.pn(x)


class PointNetHN(AbstractPointNetHN, HNKernelBetweenSupportAndQuery):
    pass


class PointNetHNKernelBetweenSupportAndQuery(AbstractPointNetHN, HNKernelBetweenSupportAndQuery):
    pass


class AbstractPermutationHN(abc.ABC):
    permute_support = True
    permute_query = False

    def permute_set(self, input_set: torch.Tensor, permute_dim: int = 0) -> torch.Tensor:
        """
        Permute set:
        input_set: torch.Tensor
            Tensor of size [x1, ..., xN]
        permute_dim: int
            Axis to permute
        For example for tensor:
        input_set = [[1, 2], [4, 5], [6, 7]]
        permute_dim = 0
        Example transformation result may be:
        [[4, 5], [1, 2], [6, 7]]
        """
        if permute_dim >= len(input_set.shape):
            raise ValueError(f'Input tensor has rank: {len(input_set.shape)}, but get permute_dim: {permute_dim}')
        permutation = torch.randperm(input_set.shape[permute_dim]).to(input_set.device)
        permuted_tensor = input_set.index_select(dim=permute_dim, index=permutation)
        return permuted_tensor

    def get_second_support_query(
            self, support_feature: torch.Tensor, query_feature: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        det_support_feature = support_feature.detach().clone() if self.detach_support else support_feature
        det_query_feature = query_feature.detach().clone() if self.detach_query else query_feature

        det_support_feature = self.permute_set(
            det_support_feature,
            permute_dim=0) if self.permute_support else det_support_feature
        det_query_feature = self.permute_set(
            det_query_feature,
            permute_dim=0) if self.permute_query else det_query_feature

        return det_support_feature, det_query_feature


class PermutationHN(AbstractPermutationHN, HyperNetPOC):
    pass


class PermutationHNKernelBetweenSupportAndQuery(AbstractPermutationHN, HNKernelBetweenSupportAndQuery):
    pass


hn_poc_types = {
    "hn_poc": HyperNetPOC,
    "hn_sep_joint": HyperNetSepJoint,
    "hn_from_dkt": HyperNetConvFromDKT,
    "hn_cnv": HyperNetSupportConv,
    "hn_kernel": HyperNetSupportKernel,
    "hn_sup_kernel": HNKernelBetweenSupportAndQuery,  # the best architecture right now
    "hn_perm": PermutationHN,
    "hn_perm_sup_kernel": PermutationHNKernelBetweenSupportAndQuery,
    "hn_pointnet": PointNetHN
}
