from collections import defaultdict
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import gpytorch
from torch import nn
from torch.utils.data import DataLoader

from methods.kernels import NNKernel, MultiNNKernel, NNKernelNoInner
from methods.hypnettorch_utils import build_hypnettorch
from methods.kernels import NNKernel
from methods.meta_template import MetaTemplate
from methods.transformer import TransformerEncoder
from hypnettorch.mnets import MLP



class HyperNetPOC(MetaTemplate):
    def __init__(
            self, model_func: nn.Module, n_way: int, n_support: int, n_query: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        super().__init__(model_func, n_way, n_support)

        conv_out_size = self.feature.final_feat_dim
        self.n_query = n_query
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
        self.hn_ln: bool = params.hn_ln
        self.hn_dropout: float = params.hn_dropout
        self.hn_val_epochs: int = params.hn_val_epochs
        self.hn_val_lr: float = params.hn_val_lr
        self.hn_val_optim: float = params.hn_val_optim
        self.target_net_architecture = target_net_architecture or self.build_target_net_architecture(params)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_hypernet_modules()
        if self.attention_embedding:
            self.init_transformer_architecture(params)

    def build_target_net_architecture(self, params) -> nn.Module:
        tn_hidden_size = params.hn_tn_hidden_size
        layers = []

        activation_name = params.hn_tn_activation
        activation_fn = (
            nn.Tanh if activation_name == "tanh" else
            nn.ReLU if activation_name == "relu" else
            SinActivation
        )

        for i in range(params.hn_tn_depth):
            is_final = i == (params.hn_tn_depth - 1)
            insize = self.feature.final_feat_dim if i == 0 else tn_hidden_size
            outsize = self.n_way if is_final else tn_hidden_size
            layers.append(nn.Linear(insize, outsize))
            if not is_final:
                layers.append(activation_fn())
        res = nn.Sequential(*layers)
        print(res)
        return res

    def init_transformer_architecture(self, params):
        self.transformer_layers_no: int = params.hn_transformer_layers_no
        self.transformer_input_dim: int = self.conv_out_size + self.n_way
        self.transformer_heads: int = params.hn_transformer_heads_no
        self.transformer_dim_feedforward: int = params.hn_transformer_feedforward_dim
        self.transformer_encoder: nn.Module = TransformerEncoder(num_layers=self.transformer_layers_no, input_dim=self.transformer_input_dim, num_heads=self.transformer_heads, dim_feedforward=self.transformer_dim_feedforward)

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
            if self.hn_ln:
                neck_modules = [nn.LayerNorm(self.embedding_size)] + neck_modules
            for _ in range(self.hn_neck_len - 1):
                if self.hn_ln:
                    neck_modules.append(nn.LayerNorm(self.hn_hidden_size))
                neck_modules.extend(
                    [nn.Dropout(self.hn_dropout), nn.Linear(self.hn_hidden_size, self.hn_hidden_size), nn.ReLU()]
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
                if self.hn_ln:
                    head_modules.append(nn.LayerNorm(in_size))
                head_modules.extend([nn.Dropout(self.hn_dropout), nn.Linear(in_size, out_size)])
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

    def generate_network_params(self, support_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.build_embedding(support_feature)

        root = self.hypernet_neck(embedding)
        network_params = {
            name.replace("-", "."): param_net(root).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.hypernet_heads.items()
        }
        return network_params

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """
        network_params = self.generate_network_params(support_feature)

        tn = deepcopy(self.target_net_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()

    def set_forward(self, x: torch.Tensor, is_feature: bool = False):
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

    def set_forward_with_adaptation(self, x: torch.Tensor):
        self_copy = deepcopy(self)
        metrics = {
            "accuracy/val@-0": self_copy.query_accuracy(x)
        }
        val_opt_type = torch.optim.Adam if self.hn_val_optim == "adam" else torch.optim.SGD
        val_opt = val_opt_type(self_copy.parameters(), lr=self.hn_val_lr)
        for i in range(1, self.hn_val_epochs + 1):
            self_copy.train()
            val_opt.zero_grad()
            loss = self_copy.set_forward_loss(x, train_on_query=False)
            loss.backward()
            val_opt.step()
            self_copy.eval()
            metrics[f"accuracy/val@-{i}"] = self_copy.query_accuracy(x)

        return self_copy.set_forward(x), metrics


    def query_accuracy(self, x: torch.Tensor) -> float:
        scores = self.set_forward(x)
        return accuracy_from_scores(scores, n_way=self.n_way, n_query=self.n_query)

    def set_forward_loss(
            self, x: torch.Tensor, detach_ft_hn: bool = False, detach_ft_tn: bool = False,
            train_on_support: bool = True,
            train_on_query: bool = True
    ):
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

        feature_to_classify = []
        y_to_classify_gt = []
        if train_on_support:
            feature_to_classify.append(
                support_feature.reshape(
                    (self.n_way * self.n_support), support_feature.shape[-1]
                )
            )
            y_support = self.get_labels(support_feature)
            y_to_classify_gt.append(y_support.reshape(self.n_way * self.n_support))
        if train_on_query:
            feature_to_classify.append(
                query_feature.reshape(
                    (self.n_way * (ne - self.n_support)), query_feature.shape[-1]
                )
            )
            y_query = self.get_labels(query_feature)
            y_to_classify_gt.append(y_query.reshape(self.n_way * (ne - self.n_support)))

        feature_to_classify = torch.cat(feature_to_classify)
        y_to_classify_gt = torch.cat(y_to_classify_gt)

        if detach_ft_tn:
            feature_to_classify = feature_to_classify.detach()

        y_pred = classifier(feature_to_classify)
        return self.loss_fn(y_pred, y_to_classify_gt)

    def train_loop(self, epoch: int, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
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


class HyperNetPocWithKernel(HyperNetPOC):
    def __init__(
            self, model_func: nn.Module, n_way: int, n_support: int, n_query: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        super().__init__(
            model_func, n_way, n_support, n_query, params=params, target_net_architecture=target_net_architecture
        )

        # TODO - check!!!
        conv_out_size = self.feature.final_feat_dim
        # Use scalar product instead of a specific kernel
        self.use_scalar_product: bool = params.use_scalar_product

        if not self.use_scalar_product:
            self.kernel_input_dim = conv_out_size + self.n_way if self.attention_embedding else conv_out_size
            self.kernel_output_dim = conv_out_size + self.n_way if self.attention_embedding else conv_out_size
            self.kernel_layers_no = params.hn_kernel_layers_no
            self.kernel_hidden_dim = params.hn_kernel_hidden_dim
            self.kernel_function = NNKernel(self.kernel_input_dim, self.kernel_output_dim,
                                        self.kernel_layers_no, self.kernel_hidden_dim)
        # I will be adding the kernel vector to the stacked images embeddings
        #TODO: add/check changes for attention-like input
        # TODO - check!!!
        self.hn_kernel_invariance: bool = params.hn_kernel_invariance
        self.hn_kernel_invariance_type: str = params.hn_kernel_invariance_type
        self.hn_kernel_convolution_output_dim: int = params.hn_kernel_convolution_output_dim
        self.hn_kernel_invariance_pooling: str = params.hn_kernel_invariance_pooling

        # TODO - check!!!
        # embedding size
        # TODO - add attention based input also
        if self.hn_kernel_invariance:
            if self.hn_kernel_invariance_type == 'attention':
                self.embedding_size: int = conv_out_size * self.n_way * self.n_support + (self.n_way * self.n_support)
            else:
                self.embedding_size: int = conv_out_size * self.n_way * self.n_support + self.hn_kernel_convolution_output_dim
        else:
            self.embedding_size: int = conv_out_size * self.n_way * self.n_support + ((self.n_way * self.n_support) * (self.n_way * self.n_query))

        # invariant operation type
        if self.hn_kernel_invariance:
            if self.hn_kernel_invariance_type == 'attention':
                self.init_kernel_transformer_architecture(params)
            else:
                self.init_kernel_convolution_architecture(params)

        # if self.attention_embedding:
        #     self.embedding_size: int = (conv_out_size + self.n_way) * self.n_way * self.n_support + (self.n_way * self.n_support)
        # else:
        #     self.embedding_size: int = conv_out_size * self.n_way * self.n_support + (self.n_way * self.n_support)

        # self.init_kernel_transformer_architecture(params)
        self.init_hypernet_modules()

    def init_kernel_transformer_architecture(self, params):
        self.kernel_transformer_layers_no: int = params.kernel_transformer_layers_no
        self.kernel_transformer_input_dim: int = self.n_way * self.n_support
        self.kernel_transformer_heads: int = params.kernel_transformer_heads_no
        self.kernel_transformer_dim_feedforward: int = params.kernel_transformer_feedforward_dim
        self.kernel_transformer_encoder: nn.Module = TransformerEncoder(num_layers=self.kernel_transformer_layers_no, input_dim=self.kernel_transformer_input_dim, num_heads=self.kernel_transformer_heads, dim_feedforward=self.kernel_transformer_dim_feedforward)

    def init_kernel_convolution_architecture(self, params):
        # TODO - add convolution-based approach
        self.kernel_1D_convolution: bool = True

    def build_kernel_features_embedding(self, support_feature: torch.Tensor, query_feature: torch.Tensor) -> torch.Tensor:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        supp_way, n_support, supp_feat = support_feature.shape
        query_way, n_query, query_feat = query_feature.shape

        #TODO: add/check changes for attention-like input
        if self.attention_embedding:
            attention_support_features = support_feature.view(1, -1, *(support_feature.size()[2:]))
            support_feature = torch.flatten(self.transformer_encoder.forward(attention_support_features))
            attention_query_features = support_feature.view(1, -1, *(query_feature.size()[2:]))
            query_feature = torch.flatten(self.transformer_encoder.forward(attention_query_features))

        support_features = support_feature.reshape(supp_way * n_support, supp_feat)
        query_features = query_feature.reshape(query_way * n_query, query_feat)

        # TODO - check!!!
        if self.use_scalar_product:
            kernel_values_tensor = torch.matmul(support_features, query_features.T)
        else:
            kernel_values_tensor = self.kernel_function.forward(support_features, query_features)

        # kernel_values_tensor = torch.unsqueeze(kernel_values_tensor.T, 0)
        if self.hn_kernel_invariance:
            if self.hn_kernel_invariance_type == 'attention':
                kernel_values_tensor = torch.unsqueeze(kernel_values_tensor.T, 0)

                if self.hn_kernel_invariance_pooling == 'min':
                    invariant_kernel_values = torch.min(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)[0]
                elif self.hn_kernel_invariance_pooling == 'max':
                    invariant_kernel_values = torch.max(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)[0]
                else:
                    invariant_kernel_values = torch.mean(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)

                return invariant_kernel_values
            else:
                # TODO - add convolutional approach
                kernel_values_tensor = torch.unsqueeze(kernel_values_tensor.T, 0)

                if self.hn_kernel_invariance_pooling == 'min':
                    invariant_kernel_values = torch.min(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)[0]
                elif self.hn_kernel_invariance_pooling == 'max':
                    invariant_kernel_values = torch.max(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)[0]
                else:
                    invariant_kernel_values = torch.mean(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)

                return invariant_kernel_values
        # invariant_kernel_values = torch.mean(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)

        return torch.unsqueeze(torch.flatten(kernel_values_tensor), 0)

    def generate_target_net_with_kernel_features(self, support_feature: torch.Tensor, query_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        embedding = torch.cat((self.build_embedding(support_feature), self.build_kernel_features_embedding(support_feature, query_feature)), 1)

        root = self.hypernet_neck(embedding)
        network_params = {
            name.replace("-", "."): param_net(root).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.hypernet_heads.items()
        }
        tn = deepcopy(self.target_net_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()

    def set_forward(self, x: torch.Tensor, is_feature: bool = False):
        support_feature, query_feature = self.parse_feature(x, is_feature)

        #TODO: add/check changes for attention-like input
        if self.attention_embedding:
            y_support = self.get_labels(support_feature)
            y_query = self.get_labels(query_feature)
            y_support_one_hot = torch.nn.functional.one_hot(y_support)
            support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
            support_feature = support_feature_with_classes_one_hot
            y_query_zeros = torch.zeros((y_query.shape[0], y_query.shape[1], y_support_one_hot.shape[2]))
            query_feature_with_zeros = torch.cat((query_feature, y_query_zeros), 2)
            query_feature = query_feature_with_zeros

        classifier = self.generate_target_net_with_kernel_features(support_feature, query_feature)
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
        nw, ne, c, h, w = x.shape

        support_feature, query_feature = self.parse_feature(x, is_feature=False)

        #TODO: add/check changes for attention-like input
        if self.attention_embedding:
            y_support = self.get_labels(support_feature)
            y_query = self.get_labels(query_feature)
            y_support_one_hot = torch.nn.functional.one_hot(y_support)
            support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
            y_query_zeros = torch.zeros((y_query.shape[0], y_query.shape[1], y_support_one_hot.shape[2]))
            query_feature_with_zeros = torch.cat((query_feature, y_query_zeros), 2)
            feature_to_hn = support_feature_with_classes_one_hot.detach() if detach_ft_hn else support_feature_with_classes_one_hot
            query_feature_to_hn = query_feature_with_zeros
        else:
            feature_to_hn = support_feature.detach() if detach_ft_hn else support_feature
            query_feature_to_hn = query_feature

        classifier = self.generate_target_net_with_kernel_features(feature_to_hn, query_feature_to_hn)

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
                        # self.n_query = task.size(1) - self.n_support
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


class HyperNetPocSupportSupportKernel(HyperNetPOC):
    def __init__(
            self, model_func: nn.Module, n_way: int, n_support: int, n_query: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        super().__init__(
            model_func, n_way, n_support, n_query, params=params, target_net_architecture=target_net_architecture
        )

        # TODO - check!!!
        conv_out_size = self.feature.final_feat_dim
        # Use scalar product instead of a specific kernel
        self.use_scalar_product: bool = params.use_scalar_product
        # Use support embeddings - concatenate them with kernel features
        self.use_support_embeddings: bool = params.use_support_embeddings
        # Remove self relations by matrix K multiplication
        self.no_self_relations: bool = params.no_self_relations

        if not self.use_scalar_product:
            self.kernel_input_dim = conv_out_size + self.n_way if self.attention_embedding else conv_out_size
            self.kernel_output_dim = conv_out_size + self.n_way if self.attention_embedding else conv_out_size
            self.kernel_layers_no = params.hn_kernel_layers_no
            self.kernel_hidden_dim = params.hn_kernel_hidden_dim
            self.kernel_function = NNKernel(self.kernel_input_dim, self.kernel_output_dim,
                                            self.kernel_layers_no, self.kernel_hidden_dim)
        # I will be adding the kernel vector to the stacked images embeddings
        #TODO: add/check changes for attention-like input

        # if self.attention_embedding:
        #     self.embedding_size: int = (conv_out_size + self.n_way) * self.n_way * self.n_support + (self.n_way * self.n_support)
        # else:
        #     self.embedding_size: int = conv_out_size * self.n_way * self.n_support + (self.n_way * self.n_support)


        self.hn_kernel_invariance: bool = params.hn_kernel_invariance
        self.hn_kernel_invariance_type: str = params.hn_kernel_invariance_type
        self.hn_kernel_convolution_output_dim: int = params.hn_kernel_convolution_output_dim
        self.hn_kernel_invariance_pooling: str = params.hn_kernel_invariance_pooling

        # embedding size
        # TODO - add attention based input also
        if self.use_support_embeddings:
            support_embeddings_size = conv_out_size * self.n_way * self.n_support
        else:
            support_embeddings_size = 0

        if self.hn_kernel_invariance:
            if self.hn_kernel_invariance_type == 'attention':
                self.embedding_size: int = support_embeddings_size + (self.n_way * self.n_support)
            else:
                self.embedding_size: int = support_embeddings_size + self.hn_kernel_convolution_output_dim
        else:
            if self.no_self_relations:
                self.embedding_size: int = support_embeddings_size + (((self.n_way * self.n_support) ** 2) - (self.n_way * self.n_support) )
            else:
                self.embedding_size: int = support_embeddings_size + ((self.n_way * self.n_support) ** 2)

        # invariant operation type
        if self.hn_kernel_invariance:
            if self.hn_kernel_invariance_type == 'attention':
                self.init_kernel_transformer_architecture(params)
            else:
                self.init_kernel_convolution_architecture(params)

        self.query_relations_size = self.n_way * self.n_support
        self.target_net_architecture = target_net_architecture or self.build_target_net_architecture(params)
        self.init_hypernet_modules()

    def build_target_net_architecture(self, params) -> nn.Module:
        tn_hidden_size = params.hn_tn_hidden_size
        layers = []
        # TODO - check!!!
        if params.use_support_embeddings:
            common_insize = ((self.n_way * self.n_support) + self.feature.final_feat_dim)
        else:
            common_insize = (self.n_way * self.n_support)

        # common_insize = ((self.n_way * self.n_support) + self.feature.final_feat_dim) if self.use_support_embeddings else (self.n_way * self.n_support)

        for i in range(params.hn_tn_depth):
            is_final = i == (params.hn_tn_depth - 1)
            insize = common_insize if i == 0 else tn_hidden_size
            outsize = self.n_way if is_final else tn_hidden_size
            layers.append(nn.Linear(insize, outsize))
            if not is_final:
                layers.append(nn.ReLU())
        res = nn.Sequential(*layers)
        print(res)
        return res

    def init_kernel_convolution_architecture(self, params):
        # TODO - add convolution-based approach
        self.kernel_1D_convolution: bool = True

    def init_kernel_transformer_architecture(self, params):
        self.kernel_transformer_layers_no: int = params.kernel_transformer_layers_no
        self.kernel_transformer_input_dim: int = self.n_way * self.n_support
        self.kernel_transformer_heads: int = params.kernel_transformer_heads_no
        self.kernel_transformer_dim_feedforward: int = params.kernel_transformer_feedforward_dim
        self.kernel_transformer_encoder: nn.Module = TransformerEncoder(num_layers=self.kernel_transformer_layers_no, input_dim=self.kernel_transformer_input_dim, num_heads=self.kernel_transformer_heads, dim_feedforward=self.kernel_transformer_dim_feedforward)

    def build_relations_features(self, support_feature: torch.Tensor, feature_to_classify: torch.Tensor) -> torch.Tensor:

        supp_way, n_support, supp_feat = support_feature.shape
        n_examples, feat_dim = feature_to_classify.shape
        support_features = support_feature.reshape(supp_way * n_support, supp_feat)

        # TODO - check!!!
        if self.use_scalar_product:
            kernel_values_tensor = torch.matmul(support_features, feature_to_classify.T)
        else:
            kernel_values_tensor = self.kernel_function.forward(support_features, feature_to_classify)

        relations = kernel_values_tensor.reshape(n_examples, supp_way * n_support)

        return relations

    def build_kernel_features_embedding(self, support_feature: torch.Tensor, query_feature: torch.Tensor) -> torch.Tensor:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        supp_way, n_support, supp_feat = support_feature.shape
        # query_way, n_query, query_feat = query_feature.shape

        #TODO: add/check changes for attention-like input


        # if self.attention_embedding:
        #     attention_support_features = support_feature.view(1, -1, *(support_feature.size()[2:]))
        #     support_feature = torch.flatten(self.transformer_encoder.forward(attention_support_features))
        #     attention_query_features = support_feature.view(1, -1, *(query_feature.size()[2:]))
        #     query_feature = torch.flatten(self.transformer_encoder.forward(attention_query_features))

        support_features = support_feature.reshape(supp_way * n_support, supp_feat)
        # query_features = query_feature.reshape(query_way * n_query, query_feat)
        support_features_copy = torch.clone(support_features)

        # TODO - check!!!
        if self.use_scalar_product:
            kernel_values_tensor = torch.matmul(support_features, support_features_copy.T)
        else:
            kernel_values_tensor = self.kernel_function.forward(support_features, support_features_copy)

        # Remove self relations by matrix multiplication
        if self.no_self_relations:
            # non_diagonal_values_matrix = torch.flatten(kernel_values_tensor)[1: ].view(self.n_way * self.n_support - 1, self.n_way * self.n_support + 1)[: ,: -1].reshape(self.n_way * self.n_support, self.n_way * self.n_support - 1)
            # return torch.flatten(non_diagonal_values_matrix)
            zero_diagonal_matrix = torch.ones_like(kernel_values_tensor).cuda() - torch.eye(kernel_values_tensor.shape[0]).cuda()
            # nonzero_indices = zero_diagonal_matrix.nonzero(as_tuple=True)
            #
            kernel_values_tensor = kernel_values_tensor * zero_diagonal_matrix
            return torch.flatten(kernel_values_tensor[kernel_values_tensor != 0.0])
            # kernel_values_tensor = kernel_values_tensor[kernel_values_tensor.nonzero(as_tuple=True)]
            # kernel_values_tensor = kernel_values_tensor[nonzero_indices]

        # TODO - check!!!
        if self.hn_kernel_invariance:
            if self.hn_kernel_invariance_type == 'attention':
                kernel_values_tensor = torch.unsqueeze(kernel_values_tensor.T, 0)

                if self.hn_kernel_invariance_pooling == 'min':
                    invariant_kernel_values = torch.min(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)[0]
                elif self.hn_kernel_invariance_pooling == 'max':
                    invariant_kernel_values = torch.max(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)[0]
                else:
                    invariant_kernel_values = torch.mean(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)

                return invariant_kernel_values
            else:
                # TODO - add convolutional approach
                kernel_values_tensor = torch.unsqueeze(kernel_values_tensor.T, 0)

                if self.hn_kernel_invariance_pooling == 'min':
                    invariant_kernel_values = torch.min(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)[0]
                elif self.hn_kernel_invariance_pooling == 'max':
                    invariant_kernel_values = torch.max(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)[0]
                else:
                    invariant_kernel_values = torch.mean(self.kernel_transformer_encoder.forward(kernel_values_tensor), 1)

                return invariant_kernel_values

        return torch.flatten(kernel_values_tensor)

    def generate_target_net_with_kernel_features(self, support_feature: torch.Tensor, query_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        embedding = self.build_kernel_features_embedding(support_feature, query_feature)

        # TODO - check!!!
        if self.use_support_embeddings:
            embedding = torch.cat((embedding, torch.flatten(support_feature)), 0)

        root = self.hypernet_neck(embedding)
        network_params = {
            name.replace("-", "."): param_net(root).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.hypernet_heads.items()
        }
        tn = deepcopy(self.target_net_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()

    def set_forward(self, x: torch.Tensor, is_feature: bool = False):
        support_feature, query_feature = self.parse_feature(x, is_feature)

        #TODO: add/check changes for attention-like input


        # if self.attention_embedding:
        #     y_support = self.get_labels(support_feature)
        #     y_query = self.get_labels(query_feature)
        #     y_support_one_hot = torch.nn.functional.one_hot(y_support)
        #     support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
        #     support_feature = support_feature_with_classes_one_hot
        #     y_query_zeros = torch.zeros((y_query.shape[0], y_query.shape[1], y_support_one_hot.shape[2]))
        #     query_feature_with_zeros = torch.cat((query_feature, y_query_zeros), 2)
        #     query_feature = query_feature_with_zeros

        classifier = self.generate_target_net_with_kernel_features(support_feature, query_feature)
        query_feature = query_feature.reshape(
            -1, query_feature.shape[-1]
        )

        relational_query_feature = self.build_relations_features(support_feature, query_feature)
        # TODO - check!!!
        if self.use_support_embeddings:
            relational_query_feature = torch.cat((relational_query_feature, query_feature), 1)
        y_pred = classifier(relational_query_feature)
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
        nw, ne, c, h, w = x.shape

        support_feature, query_feature = self.parse_feature(x, is_feature=False)

        #TODO: add/check changes for attention-like input
        if self.attention_embedding:
            y_support = self.get_labels(support_feature)
            y_query = self.get_labels(query_feature)
            y_support_one_hot = torch.nn.functional.one_hot(y_support)
            support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
            y_query_zeros = torch.zeros((y_query.shape[0], y_query.shape[1], y_support_one_hot.shape[2]))
            query_feature_with_zeros = torch.cat((query_feature, y_query_zeros), 2)
            feature_to_hn = support_feature_with_classes_one_hot.detach() if detach_ft_hn else support_feature_with_classes_one_hot
            query_feature_to_hn = query_feature_with_zeros
        else:
            feature_to_hn = support_feature.detach() if detach_ft_hn else support_feature
            query_feature_to_hn = query_feature

        classifier = self.generate_target_net_with_kernel_features(feature_to_hn, query_feature_to_hn)

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

        relational_feature_to_classify = self.build_relations_features(support_feature, feature_to_classify)

        if detach_ft_tn:
            relational_feature_to_classify = relational_feature_to_classify.detach()

        if self.use_support_embeddings:
            relational_feature_to_classify = torch.cat((relational_feature_to_classify, feature_to_classify), 1)

        y_pred = classifier(relational_feature_to_classify)
        return self.loss_fn(y_pred, y_to_classify_gt)

    def train_loop(self, epoch: int, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
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
                        # self.n_query = task.size(1) - self.n_support
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

class HNPocAdaptTN(HyperNetPOC):
    def set_forward_with_adaptation(self, x: torch.Tensor):
        sc = deepcopy(self)
        tn = deepcopy(sc.target_net_architecture)

        support_feature, query_feature = sc.parse_feature(x, is_feature=False)
        tn_params = {
            k: v.clone().detach()
            for (k,v) in sc.generate_network_params(support_feature).items()
        }
        tn.load_state_dict(tn_params)

        y_support = sc.get_labels(support_feature).reshape(sc.n_way, sc.n_support)
        y_support = y_support.reshape(sc.n_way * sc.n_support)

        query_feature = query_feature.reshape(-1, query_feature.shape[-1])
        metrics = {
            "accuracy/val@-0": accuracy_from_scores(tn(query_feature), sc.n_way, sc.n_query)
        }
        val_opt_type = torch.optim.Adam if sc.hn_val_optim == "adam" else torch.optim.SGD

        val_opt = val_opt_type(
            list(tn.parameters()) + list(sc.parameters()),
            lr=sc.hn_val_lr
        )
        for i in range(1, sc.hn_val_epochs + 1):
            sc.train()
            tn.train()
            support_feature, _ = sc.parse_feature(x, is_feature=False)
            support_feature = support_feature.reshape(-1, support_feature.shape[-1])
            val_opt.zero_grad()
            y_pred = tn(support_feature)
            loss_val = sc.loss_fn(y_pred, y_support)
            loss_val.backward()
            val_opt.step()
            sc.eval()
            tn.eval()
            _, query_feature = sc.parse_feature(x, is_feature=False)
            query_feature = query_feature.reshape(-1, query_feature.shape[-1])

            metrics[f"accuracy/val@-{i}"] = accuracy_from_scores(
                tn(query_feature), sc.n_way, sc.n_query
            )

        return tn(query_feature), metrics


class HNPocWithUniversalFinal(HyperNetPOC):
    def __init__(self, model_func, n_way: int, n_support: int, n_query: int):
        tn = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        super().__init__(model_func, n_way, n_support, n_query, target_net_architecture=tn)
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
            self, model_func: nn.Module, n_way: int, n_support: int, n_query: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        super().__init__(
            model_func, n_way, n_support, n_query, params=params, target_net_architecture=target_net_architecture
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
    def __init__(self, model_func, n_way: int, n_support: int, n_query: int):
        super().__init__(model_func, n_way, n_support, n_query)

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
    def __init__(self, model_func, n_way: int, n_support: int, n_query: int):
        super().__init__(model_func, n_way, n_support, n_query)

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
    def __init__(self, model_func, n_way: int, n_support: int, n_query: int):
        super().__init__(model_func, n_way, n_support, n_query)

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
    def __init__(self, model_func, n_way: int, n_support: int, n_query: int):
        target_net_architecture = KernelWithSupportClassifier(NNKernel(64, 16, 1, 128))
        super().__init__(
            model_func, n_way, n_support, n_query, target_net_architecture=target_net_architecture)

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
    def __init__(self, model_func, n_way: int, n_support: int, n_query: int):
        target_net_architecture = ConditionedClassifier(64, n_way, n_support)

        super().__init__(model_func, n_way, n_support, n_query, target_net_architecture=target_net_architecture)

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        tn = self.target_net_architecture
        tn.support = support_feature
        return tn.cuda()



class HNLibClassifier(nn.Module):
    def __init__(self, mlp: MLP):
        super().__init__()
        self.mlp = mlp
        self.weights = None

    def forward(self, x):
        return self.mlp.forward(
            x, weights=self.weights
        )


class HNLib(HyperNetPOC):
    def __init__(self, model_func: nn.Module, n_way: int, n_support: int, params: "ArgparseHNParams"):
        super().__init__(model_func, n_way, n_support, params)

        self.target_net_architecture = HNLibClassifier(
            mlp=MLP(
                n_in=self.feature.final_feat_dim,
                n_out=n_way,
                hidden_layers=[params.hn_tn_hidden_size] * (params.hn_tn_depth - 1)
            )
        )
        self.hypernet_heads = None
        self.hypernet_neck = None

        self.hn = build_hypnettorch(
            target_shapes=self.target_net_architecture.mlp.param_shapes,
            uncond_in_size=self.embedding_size,
            params=params,
        )

        print(self.hn)
        print(self.target_net_architecture)

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        support_feature = support_feature.reshape(1, self.embedding_size)

        weights = self.hn(support_feature)
        tn = deepcopy(self.target_net_architecture)
        tn.weights = weights
        return tn


hn_poc_types = {
    "hn_poc": HyperNetPOC,
    "hn_poc_kernel": HyperNetPocWithKernel,
    "hn_poc_sup_sup_kernel": HyperNetPocSupportSupportKernel,
    "hn_sep_joint": HyperNetSepJoint,
    "hn_from_dkt": HyperNetConvFromDKT,
    "hn_cnv": HyperNetSupportConv,
    "hn_kernel": HyperNetSupportKernel,
    "hn_sup_kernel": HNKernelBetweenSupportAndQuery,
    "no_hn_sup_kernel": NoHNKernelBetweenSupportAndQuery,
    "hn_uni_final": HNPocWithUniversalFinal,
    "no_hn_cond": NoHNConditioning,
    "hn_lib": HNLib,
    "hn_poc_adapt_tn_val": HNPocAdaptTN
}


class SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

def accuracy_from_scores(scores: torch.Tensor, n_way: int, n_query: int) -> float:
    y_query = np.repeat(range(n_way), n_query)
    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    correct_this = float(top1_correct)
    count_this = len(y_query)
    return correct_this / count_this