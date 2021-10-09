from copy import deepcopy

import numpy as np
import torch
from torch import nn
from typing import Dict

from methods.kernels import NNKernel
from methods.meta_template import MetaTemplate


class HyperNetPOC(MetaTemplate):
    def __init__(self, model_func, n_way: int, n_support: int):
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

        # TODO #1 - tweak the architecture of the target network
        target_network_architecture = nn.Sequential(
            nn.Linear(conv_out_size, tn_hidden_size),
            nn.ReLU(),
            nn.Linear(tn_hidden_size, self.n_way)
        )

        target_net_param_dict = get_param_dict(target_network_architecture)
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
        self.target_network_architecture = target_network_architecture
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
        return features.reshape(1, -1)

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

    def set_forward_loss(self, x: torch.Tensor):
        nw, ne, c, h, w = x.shape

        support_feature, query_feature = self.parse_feature(x, is_feature=False)

        # support_feature = support_feature.detach().clone()
        # query_feature = query_feature.detach().clone()
        classifier = self.generate_target_net(support_feature)

        all_feature = torch.cat(
            [
                support_feature.reshape(
                    (self.n_way * self.n_support), support_feature.shape[-1]
                ),
                query_feature.reshape(
                    (self.n_way * (ne - self.n_support)), query_feature.shape[-1]
                )
            ])

        y_support = self.get_labels(support_feature)
        y_query = self.get_labels(query_feature)
        all_y = torch.cat([
            y_support.reshape(self.n_way * self.n_support),
            y_query.reshape(self.n_way * (ne - self.n_support))
        ])
        # all_feature = all_feature.detach().clone()
        y_pred = classifier(all_feature)
        return self.loss_fn(y_pred, all_y, )

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

        self.feat_size=  self.conv_out_size + self.n_way

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
            nn.Linear((n_way * n_support)**2, self.embedding_size)
        )

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        way, shot, feat = support_feature.shape
        sf = support_feature.reshape(way*shot, feat)

        sf_k = self.kernel(sf, sf).evaluate()

        sf_k = sf_k.view(1, (way*shot)**2)
        embedding = self.kernel_to_embedding(sf_k)

        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.target_net_param_predictors.items()
        }
        tn = deepcopy(self.target_network_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()

hn_poc_types = {
    "hn_poc": HyperNetPOC,
    "hn_sep_joint": HyperNetSepJoint,
    "hn_from_dkt": HyperNetConvFromDKT,
    "hn_cnv": HyperNetSupportConv,
    "hn_kernel": HyperNetSupportKernel
}
