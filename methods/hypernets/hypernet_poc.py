from collections import defaultdict
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from methods.hypernets.utils import get_param_dict, set_from_param_dict, SinActivation, accuracy_from_scores
from methods.meta_template import MetaTemplate
from methods.transformer import TransformerEncoder

ALLOWED_AGGREGATIONS = ["concat", "mean", "max_pooling", "min_pooling"]


class HyperNetPOC(MetaTemplate):
    def __init__(
            self, model_func: nn.Module, n_way: int, n_support: int, n_query: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        super().__init__(model_func, n_way, n_support)

        self.feat_dim = self.feature.final_feat_dim = 64 if params.dataset == "cross_char" else 1600

        self.n_query = n_query
        self.taskset_size: int = params.hn_taskset_size
        self.taskset_print_every: int = params.hn_taskset_print_every
        self.hn_hidden_size: int = params.hn_hidden_size
        self.attention_embedding: bool = params.hn_attention_embedding
        self.sup_aggregation: str = params.hn_sup_aggregation
        self.detach_ft_in_hn: int = params.hn_detach_ft_in_hn
        self.detach_ft_in_tn: int = params.hn_detach_ft_in_tn
        self.hn_neck_len: int = params.hn_neck_len
        self.hn_head_len: int = params.hn_head_len
        self.taskset_repeats_config: str = params.hn_taskset_repeats
        self.hn_dropout: float = params.hn_dropout
        self.hn_val_epochs: int = params.hn_val_epochs
        self.hn_val_lr: float = params.hn_val_lr
        self.hn_val_optim: float = params.hn_val_optim
        self.hn_S_test: int = params.hn_S_test

        self.hn_kld_const_scaler = 10**(params.hn_kld_const_scaler)
        self.hn_kld_dynamic_scale = 10**(params.hn_kld_start_val)
        self.hn_kld_stop_val = 10**(params.hn_kld_stop_val)
        self.hn_step = None
        self.hn_use_kld_from = params.hn_use_kld_from
        self.hn_use_kld_scheduler = params.hn_use_kld_scheduler

        # self.epoch_state_dict = {}

        self.dataset_size = 0
        self.embedding_size = self.init_embedding_size(params)
        self.target_net_architecture = target_net_architecture or self.build_target_net_architecture(params)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_hypernet_modules()
        if self.attention_embedding:
            self.init_transformer_architecture(params)

        print(self.target_net_architecture)

    def init_embedding_size(self, params) -> int:
        if self.attention_embedding:
            return (self.feat_dim + self.n_way) * self.n_way * self.n_support
        else:
            assert self.sup_aggregation in ALLOWED_AGGREGATIONS
            if self.sup_aggregation == "concat":
                return self.feat_dim * self.n_way * self.n_support
            elif self.sup_aggregation in ["mean", "max_pooling", "min_pooling"]:
                return self.feat_dim * self.n_way

    def build_target_net_architecture(self, params) -> nn.Module:
        tn_hidden_size = params.hn_tn_hidden_size
        layers = []


        for i in range(params.hn_tn_depth):
            is_final = i == (params.hn_tn_depth - 1)
            insize = self.feat_dim if i == 0 else tn_hidden_size
            outsize = self.n_way if is_final else tn_hidden_size
            layers.append(nn.Linear(insize, outsize))
            if not is_final:
                layers.append(nn.ReLU())
        res = nn.Sequential(*layers)
        print(res)
        return res

    def init_transformer_architecture(self, params):
        transformer_input_dim: int = self.feat_dim + self.n_way
        self.transformer_encoder: nn.Module = TransformerEncoder(
            num_layers=params.hn_transformer_layers_no, input_dim=transformer_input_dim,
            num_heads=params.hn_transformer_heads_no, dim_feedforward=params.hn_transformer_feedforward_dim)

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

        self.init_hypernet_neck()

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
                head_modules.extend([nn.Dropout(self.hn_dropout), nn.Linear(in_size, out_size)])
                if not is_final:
                    head_modules.append(nn.ReLU())

            self.hypernet_heads[name] = nn.Sequential(*head_modules)

    def init_hypernet_neck(self):
        neck_modules = []
        if self.hn_neck_len > 0:

            neck_modules = [
                nn.Linear(self.embedding_size, self.hn_hidden_size),
                nn.ReLU()
            ]
            for _ in range(self.hn_neck_len - 1):
                neck_modules.extend(
                    [nn.Dropout(self.hn_dropout), nn.Linear(self.hn_hidden_size, self.hn_hidden_size), nn.ReLU()]
                )

            neck_modules = neck_modules[:-1]  # remove the last ReLU

        self.hypernet_neck = nn.Sequential(*neck_modules)

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

    def maybe_aggregate_support_feature(self, support_feature: torch.Tensor) -> torch.Tensor:
        way, n_support, feat = support_feature.shape
        if self.sup_aggregation == "concat":
            features = support_feature.reshape(way * n_support, feat)
        elif self.sup_aggregation == "sum":
            features = support_feature.sum(dim=1)
            way, feat = features.shape
            assert (way, feat) == (self.n_way, self.feat_dim)
        elif self.sup_aggregation == "mean":
            features = support_feature.mean(dim=1)
            way, feat = features.shape
            assert (way, feat) == (self.n_way, self.feat_dim)
        else:
            raise TypeError(self.sup_aggregation)

        return features

    def build_embedding(self, support_feature: torch.Tensor) -> torch.Tensor:
        way, n_support, feat = support_feature.shape
        if self.attention_embedding:
            features = support_feature.view(1, -1, *(support_feature.size()[2:]))
            attention_features = torch.flatten(self.transformer_encoder.forward(features))
            return attention_features

        features = self.maybe_aggregate_support_feature(support_feature)
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

    def set_forward(self, x: torch.Tensor, is_feature: bool = False, permutation_sanity_check: bool = False):
        support_feature, query_feature = self.parse_feature(x, is_feature)

        if self.attention_embedding:
            y_support = self.get_labels(support_feature)
            y_support_one_hot = torch.nn.functional.one_hot(y_support)
            support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
            support_feature = support_feature_with_classes_one_hot

        classifier = self.generate_target_net(support_feature)

        # get parameters of classifier
        bayesian_params_dict = self.upload_mu_and_sigma_histogram(classifier, test=True)

        final_y_pred = []

        for sample in range(self.hn_S_test):
            query_feature = query_feature.reshape(
                -1, query_feature.shape[-1]
            )
            y_pred = classifier(query_feature)
            final_y_pred.append(y_pred)

            if permutation_sanity_check:
                ### random permutation test
                perm = torch.randperm(len(query_feature))
                rev_perm = torch.argsort(perm)
                query_perm = query_feature[perm]
                assert torch.equal(query_perm[rev_perm], query_feature)
                y_pred_perm = classifier(query_perm)
                assert torch.equal(y_pred_perm[rev_perm], y_pred)

        return torch.stack(final_y_pred).mean(dim=0), bayesian_params_dict

    def set_forward_with_adaptation(self, x: torch.Tensor):
        self_copy = deepcopy(self)
        metrics = {
            "accuracy/val@-0": self_copy.query_accuracy(x)
        }
        val_opt_type = torch.optim.Adam if self.hn_val_optim == "adam" else torch.optim.SGD
        val_opt = val_opt_type(self_copy.parameters(), lr=self.hn_val_lr)
        if self.hn_val_epochs > 0:
            for i in range(1, self.hn_val_epochs + 1):
                self_copy.train()
                val_opt.zero_grad()
                loss = self_copy.set_forward_loss(x, train_on_query=False)
                loss.backward()
                val_opt.step()
                self_copy.eval()
                metrics[f"accuracy/val@-{i}"] = self_copy.query_accuracy(x)
        y_pred, bayesian_params_dict = self_copy.set_forward(x, permutation_sanity_check=True)
        return y_pred, bayesian_params_dict, metrics

    def query_accuracy(self, x: torch.Tensor) -> float:
        scores, _ = self.set_forward(x)
        return accuracy_from_scores(scores, n_way=self.n_way, n_query=self.n_query)

    def set_forward_loss(
            self, x: torch.Tensor, detach_ft_hn: bool = False, detach_ft_tn: bool = False,
            train_on_support: bool = True,
            train_on_query: bool = True
    ):
        n_way, n_examples, c, h, w = x.shape

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
                    (self.n_way * (n_examples - self.n_support)), query_feature.shape[-1]
                )
            )
            y_query = self.get_labels(query_feature)
            y_to_classify_gt.append(y_query.reshape(self.n_way * (n_examples - self.n_support)))

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
        losses = [] # kld_scaled + crossentropy
        kld_losses = [] # kld
        crossentropy_losses = [] # crossentropy
        metrics = defaultdict(list)
        ts_repeats = self.taskset_repeats(epoch)
        self.dataset_size = len(train_loader.dataset)


        self._scale_step()
        reduction = self.hn_kld_dynamic_scale

        if not self.hn_use_kld_scheduler:
            reduction = 1

        for i, (x, _) in enumerate(train_loader):
            taskset.append(x)

            # TODO 3: perhaps the idea of tasksets is redundant and it's better to update weights at every task
            if i % self.taskset_size == (self.taskset_size - 1) or i == (n_train - 1):
                crossentropy_loss_sum = torch.tensor(0.0).cuda()
                kld_loss_sum = torch.tensor(0.0).cuda()
                for tr in range(ts_repeats):
                    crossentropy_loss_sum = torch.tensor(0.0).cuda()
                    kld_loss_sum = torch.tensor(0.0).cuda()

                    for task in taskset:
                        if self.change_way:
                            self.n_way = task.size(0)
                        self.n_query = task.size(1) - self.n_support
                        crossentropy_loss, kld_loss, hist_data = self.set_forward_loss(task, epoch=epoch)
                        crossentropy_loss_sum += crossentropy_loss
                        kld_loss_sum += kld_loss

                    if epoch >= self.hn_use_kld_from:
                        loss_sum = crossentropy_loss_sum + kld_loss_sum * reduction * self.hn_kld_const_scaler
                    else:
                        loss_sum = crossentropy_loss_sum
                    
                    optimizer.zero_grad()
                    loss_sum.backward()

                    if tr == 0:
                        for k, p in get_param_dict(self).items():
                            if(k.split('.')[0] != "target_net_architecture"):
                                metrics[f"grad_norm/{k}"] = p.grad.abs().mean().item() if p.grad is not None else 0

                    optimizer.step()

                losses.append(loss_sum.item())
                kld_losses.append(kld_loss_sum.item())
                crossentropy_losses.append(crossentropy_loss_sum.item())
                accuracies.extend([
                    self.query_accuracy(task) for task in taskset
                ])

                acc_mean = np.mean(accuracies) * 100
                acc_std = np.std(accuracies) * 100

                if taskset_id % self.taskset_print_every == 0:
                    print(
                        f"Epoch {epoch} | Taskset {taskset_id} | TS {len(taskset)} | TS epochs {ts_repeats} | Loss {loss_sum.item()} | KLD_Loss {kld_loss_sum.item()} | Train acc {acc_mean:.2f} +- {acc_std:.2f} %")

                taskset_id += 1
                taskset = []

        metrics["loss/train"] = np.mean(losses)
        metrics["kld_loss/train"] = np.mean(kld_losses)
        metrics["kld_loss_scaled/train"] = np.mean(kld_losses) * reduction * self.hn_kld_const_scaler
        metrics["crossentropy_loss/train"] = np.mean(crossentropy_losses)
        metrics["accuracy/train"] = np.mean(accuracies) * 100
        metrics["kld_scale"] = reduction * self.hn_kld_const_scaler

        return metrics, hist_data

    def _scale_step(self):
        if self.hn_step is None:
            self.hn_step = np.power(1 / self.hn_kld_dynamic_scale * self.hn_kld_stop_val, 1 / self.stop_epoch)
            
        self.hn_kld_dynamic_scale = self.hn_kld_dynamic_scale * self.hn_step


class PPAMixin(HyperNetPOC):
    def build_target_net_architecture(self, params) -> nn.Module:
        assert params.hn_tn_depth == 1, "In PPA the target network must be a single linear layer, please use `--hn_tn_depth=1`"
        return super().build_target_net_architecture(params)

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
        self.init_hypernet_neck()

        self.hypernet_heads = nn.ModuleDict()
        assert self.hn_head_len >= 1, "Head len must be >= 1!"

        # assert False, self.target_net_param_shapes
        for name, param in target_net_param_dict.items():
            head_in = self.embedding_size if self.hn_neck_len == 0 else self.hn_hidden_size
            head_modules = []
            assert param.numel() % self.n_way == 0, f"Each param in PPA should be divisible by {self.n_way=}, but {name} is of {param.shape=} -> {param.numel()=}"
            head_out = param.numel() // self.n_way
            for i in range(self.hn_head_len):
                in_size = head_in if i == 0 else self.hn_hidden_size
                is_final = (i == (self.hn_head_len - 1))
                out_size = head_out if is_final else self.hn_hidden_size
                head_modules.extend([nn.Dropout(self.hn_dropout), nn.Linear(in_size, out_size)])
                if not is_final:
                    head_modules.append(nn.ReLU())

            self.hypernet_heads[name] = nn.Sequential(*head_modules)

class HypernetPPA(PPAMixin, HyperNetPOC):
    """Based loosely on https://arxiv.org/abs/1706.03466"""

    def taskset_repeats(self, epoch: int):
        return 1

    def init_embedding_size(self, params) -> int:
        if self.attention_embedding:
            raise NotImplementedError()
        else:
            assert self.sup_aggregation in ALLOWED_AGGREGATIONS
            if self.sup_aggregation == "concat":
                return self.feat_dim * self.n_support
            elif self.sup_aggregation in ["mean", "max_pooling", "min_pooling"]:
                return self.feat_dim



    def build_embedding(self, support_feature: torch.Tensor) -> torch.Tensor:
        way, n_support, feat = support_feature.shape
        if self.attention_embedding:
            features = support_feature.view(1, -1, *(support_feature.size()[2:]))
            attention_features = torch.flatten(self.transformer_encoder.forward(features))
            return attention_features

        features = self.maybe_aggregate_support_feature(support_feature)
        return features

    def generate_network_params(self, support_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.build_embedding(support_feature)
        assert embedding.shape[0] == self.n_way

        root = self.hypernet_neck(embedding)
        network_params = {
            name.replace("-", "."): param_net(root).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.hypernet_heads.items()
        }

        return network_params
