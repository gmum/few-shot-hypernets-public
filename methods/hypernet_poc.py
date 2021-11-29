from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from typing import Dict, Optional

from torch.utils.data import DataLoader

from methods.kernels import NNKernel
from methods.meta_template import MetaTemplate
from methods.transformer import TransformerEncoder


class HyperNetPOC(MetaTemplate):
    def __init__(
            self, model_func: nn.Module, n_way: int, n_support: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        super().__init__(model_func, n_way, n_support)

        conv_out_size = self.feature.final_feat_dim
        self.taskset_size: int = params.hn_taskset_size
        self.taskset_print_every: int = params.hn_taskset_print_every
        self.hn_hidden_size: int = params.hn_hidden_size
        self.conv_out_size: int = conv_out_size
        self.attention_embedding: bool = params.hn_attention_embedding
        if self.attention_embedding:
            self.embedding_size: int = (conv_out_size + self.n_way) * self.n_way * self.n_support
        else:
            self.embedding_size: int = conv_out_size * self.n_way * self.n_support
        self.detach_ft_in_hn: int = params.hn_detach_ft_in_hn
        self.detach_ft_in_tn: int = params.hn_detach_ft_in_tn
        self.hn_neck_len: int = params.hn_neck_len
        self.hn_head_len: int = params.hn_head_len
        self.taskset_repeats_config: str = params.hn_taskset_repeats

        self.target_net_architecture = target_net_architecture or self.build_target_net_architecture(params)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_hypernet_modules()
        if self.attention_embedding:
            self.init_transformer_architecture(params)

    def build_target_net_architecture(self, params) -> nn.Module:
        tn_hidden_size = params.hn_tn_hidden_size
        layers = []
        for i in range(params.hn_tn_depth):
            is_final = i == (params.hn_tn_depth - 1)
            insize = self.feature.final_feat_dim if i ==0 else tn_hidden_size
            outsize = self.n_way if is_final else tn_hidden_size
            layers.append(nn.Linear(insize, outsize))
            if not is_final:
                layers.append(nn.ReLU())
        res = nn.Sequential(*layers)
        print(res)
        return res

    def init_transformer_architecture(self, params):
        self.transformers_layers_no: int = params.hn_transformer_layers_no
        self.transformer_input_dim: int = self.conv_out_size + self.n_way
        self.transformer_heads: int = params.hn_transformer_heads_no
        self.transformer_dim_feedforward: int = params.hn_transformer_feedforward_dim
        self.transformer_encoder: nn.Module = TransformerEncoder(num_layers=self.transformers_layers_no, input_dim=self.transformer_input_dim, num_heads=self.transformer_heads, dim_feedforward=self.transformer_dim_feedforward)

    def init_hypernet_modules(self):
        target_net_param_dict = get_param_dict(self.target_net_architecture)
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
        neck_modules = []
        if self.hn_neck_len > 0:
            neck_modules = [
                nn.Linear(self.embedding_size, self.hn_hidden_size),
                nn.ReLU()
            ]
            for _ in range(self.hn_neck_len - 1):
                neck_modules.extend(
                    [nn.Linear(self.hn_hidden_size, self.hn_hidden_size), nn.ReLU()]
                )

            neck_modules = neck_modules[:-1]  # remove the last ReLU

        self.hypernet_neck = nn.Sequential(*neck_modules)

        self.hypernet_heads = nn.ModuleDict()
        assert self.hn_head_len >= 1, "Head len must be >= 1!"
        for name, param in target_net_param_dict.items():
            head_in = self.embedding_size if self.hn_neck_len == 0 else self.hn_hidden_size
            head_out = param.numel()
            head_modules = []

            for i in range(self.hn_head_len):
                in_size = head_in if i == 0 else self.hn_hidden_size
                is_final = (i == (self.hn_head_len - 1))
                out_size = head_out if is_final else self.hn_hidden_size
                head_modules.append(nn.Linear(in_size, out_size))
                if not is_final:
                    head_modules.append(nn.ReLU())

            self.hypernet_heads[name] = nn.Sequential(*head_modules)

    def taskset_repeats(self, epoch: int):
        epoch_ceiling_to_n_repeats = {
            int(kv.split(":")[0]): int(kv.split(":")[1])
            for kv in self.taskset_repeats_config.split("-")
        }
        epoch_ceiling_to_n_repeats = {k: v for (k, v) in epoch_ceiling_to_n_repeats.items() if k > epoch}
        if len(epoch_ceiling_to_n_repeats) == 0:
            return 1
        return epoch_ceiling_to_n_repeats[min(epoch_ceiling_to_n_repeats.keys())]

    def get_labels(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [n_way, n_shot, hidden_size]
        """
        ys = torch.tensor(list(range(x.shape[0]))).reshape(len(x), 1)
        ys = ys.repeat(1, x.shape[1]).to(x.device)
        return ys.cuda()

    def build_embedding(self, support_feature: torch.Tensor) -> torch.Tensor:
        way, n_support, feat = support_feature.shape
        if self.attention_embedding:
            features = support_feature.view(1, -1, *(support_feature.size()[2:]))
            attention_features = torch.flatten(self.transformer_encoder.forward(features))
            return attention_features
        features = support_feature.reshape(way * n_support, feat)
        features = features.reshape(1, -1)
        return features

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        embedding = self.build_embedding(support_feature)

        root = self.hypernet_neck(embedding)
        network_params = {
            name.replace("-", "."): param_net(root).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.hypernet_heads.items()
        }
        tn = deepcopy(self.target_net_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()

    def set_forward(self, x: torch.Tensor, is_feature: bool = False):
        #TODO - one-hot!!!
        support_feature, query_feature = self.parse_feature(x, is_feature)

        if self.attention_embedding:
            y_support = self.get_labels(support_feature)
            y_support_one_hot = torch.nn.functional.one_hot(y_support)
            support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
            support_feature = support_feature_with_classes_one_hot

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

    def set_forward_loss(self, x: torch.Tensor, detach_ft_hn: bool = False, detach_ft_tn: bool = False):
        #TODO - one-hot!!!
        nw, ne, c, h, w = x.shape

        support_feature, query_feature = self.parse_feature(x, is_feature=False)

        if self.attention_embedding:
            y_support = self.get_labels(support_feature)
            y_query = self.get_labels(query_feature)
            y_support_one_hot = torch.nn.functional.one_hot(y_support)
            support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
            feature_to_hn = support_feature_with_classes_one_hot.detach() if detach_ft_hn else support_feature_with_classes_one_hot
        else:
            feature_to_hn = support_feature.detach() if detach_ft_hn else support_feature
        

        classifier = self.generate_target_net(feature_to_hn)


        feature_to_classify = torch.cat(
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
        y_to_classify_gt = torch.cat([
            y_support.reshape(self.n_way * self.n_support),
            y_query.reshape(self.n_way * (ne - self.n_support))
        ])

        if detach_ft_tn:
            feature_to_classify = feature_to_classify.detach()

        y_pred = classifier(feature_to_classify)
        return self.loss_fn(y_pred, y_to_classify_gt)

    def train_loop(self, epoch: int, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
        #TODO - transformer - learnable params!!!
        #TODO - add kernel solution!!!
        taskset_id = 0
        taskset = []
        n_train = len(train_loader)
        accuracies = []
        losses = []
        metrics = defaultdict(list)
        ts_repeats = self.taskset_repeats(epoch)

        for i, (x, _) in enumerate(train_loader):
            taskset.append(x)

            # TODO 3: perhaps the idea of tasksets is redundant and it's better to update weights at every task
            if i % self.taskset_size == (self.taskset_size - 1) or i == (n_train - 1):
                loss_sum = torch.tensor(0).cuda()
                for tr in range(ts_repeats):
                    loss_sum = torch.tensor(0).cuda()

                    for task in taskset:
                        if self.change_way:
                            self.n_way = task.size(0)
                        self.n_query = task.size(1) - self.n_support
                        loss = self.set_forward_loss(task)
                        loss_sum = loss_sum + loss

                    optimizer.zero_grad()
                    loss_sum.backward()

                    if tr == 0:
                        for k, p in get_param_dict(self).items():
                            metrics[f"grad_norm/{k}"] = p.grad.abs().mean().item() if p.grad is not None else 0

                    optimizer.step()

                losses.append(loss_sum.item())
                accuracies.extend([
                    self.query_accuracy(task) for task in taskset
                ])
                acc_mean = np.mean(accuracies) * 100
                acc_std = np.std(accuracies) * 100

                if taskset_id % self.taskset_print_every == 0:
                    print(
                        f"Epoch {epoch} | Taskset {taskset_id} | TS {len(taskset)} | TS epochs {ts_repeats} | Loss {loss_sum.item()} | Train acc {acc_mean:.2f} +- {acc_std:.2f} %")

                taskset_id += 1
                taskset = []

        metrics["loss/train"] = np.mean(losses)
        metrics["accuracy/train"] = np.mean(accuracies) * 100
        return metrics


class HNPocWithUniversalFinal(HyperNetPOC):
    def __init__(self, model_func, n_way: int, n_support: int):
        tn = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        super().__init__(model_func, n_way, n_support, target_net_architecture=tn)
        self.final_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_way)
        ).cuda()

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        target_net = super().generate_target_net(support_feature)
        return nn.Sequential(target_net, self.final_layer)


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
    def __init__(
            self, model_func: nn.Module, n_way: int, n_support: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        super().__init__(
            model_func, n_way, n_support, params=params, target_net_architecture=target_net_architecture
        )
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
            for name, param_net in self.hypernet_heads.items()
        }
        tn = deepcopy(self.target_net_architecture)
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
            for name, param_net in self.hypernet_heads.items()
        }
        tn = deepcopy(self.target_net_architecture)
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
            for name, param_net in self.hypernet_heads.items()
        }
        tn = deepcopy(self.target_net_architecture)
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

    def taskset_repeats(self, epoch: int):
        return 1

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        way, n_support, feat = support_feature.shape

        embedding = self.build_embedding(support_feature)
        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.hypernet_heads.items()
        }
        tn: KernelWithSupportClassifier = deepcopy(self.target_net_architecture)
        set_from_param_dict(tn, network_params)
        tn.support = support_feature
        return tn.cuda()


class NoHNKernelBetweenSupportAndQuery(HNKernelBetweenSupportAndQuery):
    """Simply training the "kernel" target net, without using the hypernetwork"""

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        tn = self.target_net_architecture
        tn.support = support_feature
        return tn.cuda()


class ConditionedClassifier(nn.Module):
    def __init__(self, emb_size: int, n_way: int, n_support: int, hidden_size: int = 128):
        super().__init__()
        sup_h_size = 128
        self.net = nn.Sequential(
            nn.Linear((emb_size + sup_h_size), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_way)
        )
        self.support = None
        self.emb_size = emb_size
        self.sup_net = nn.Sequential(
            nn.Linear((n_way * n_support) * emb_size, sup_h_size),
            nn.ReLU(),
            nn.Linear(sup_h_size, sup_h_size)
        )

    def forward(self, query: torch.Tensor):
        b, es = query.shape

        support_emb = self.support.reshape(1, -1).repeat(b, 1)
        support_emb = self.sup_net(support_emb)

        q_cond = torch.cat([
            query, support_emb
        ], dim=1)
        return self.net(q_cond)


class NoHNConditioning(HyperNetPOC):
    def __init__(self, model_func, n_way: int, n_support: int):
        target_net_architecture = ConditionedClassifier(64, n_way, n_support)

        super().__init__(model_func, n_way, n_support, target_net_architecture=target_net_architecture)

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        tn = self.target_net_architecture
        tn.support = support_feature
        return tn.cuda()


hn_poc_types = {
    "hn_poc": HyperNetPOC,
    "hn_sep_joint": HyperNetSepJoint,
    "hn_from_dkt": HyperNetConvFromDKT,
    "hn_cnv": HyperNetSupportConv,
    "hn_kernel": HyperNetSupportKernel,
    "hn_sup_kernel": HNKernelBetweenSupportAndQuery,
    "no_hn_sup_kernel": NoHNKernelBetweenSupportAndQuery,
    "hn_uni_final": HNPocWithUniversalFinal,
    "no_hn_cond": NoHNConditioning
}
