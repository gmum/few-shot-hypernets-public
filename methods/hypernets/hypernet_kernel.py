from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from methods.hypernets import HyperNetPOC
from methods.hypernets.utils import get_param_dict, set_from_param_dict, accuracy_from_scores
from methods.kernels import NNKernel
from methods.transformer import TransformerEncoder


class HyperNetPocSupportSupportKernel(HyperNetPOC):
    def __init__(
            self, model_func: nn.Module, n_way: int, n_support: int, n_query: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        super().__init__(
            model_func, n_way, n_support, n_query, params=params, target_net_architecture=target_net_architecture
        )

        # TODO - check!!!
        conv_out_size = 1600
        # conv_out_size = self.feature.final_feat_dim
        # Use scalar product instead of a specific kernel
        self.use_scalar_product: bool = params.use_scalar_product
        # Use cosine distance instead of a specific kernel
        self.use_cosine_distance: bool = params.use_cosine_distance
        # Use support embeddings - concatenate them with kernel features
        self.use_support_embeddings: bool = params.use_support_embeddings
        # Remove self relations by matrix K multiplication
        self.no_self_relations: bool = params.no_self_relations

        self.n_support_size_context = 1 if self.sup_aggregation in ["mean", "min_pooling", "max_pooling"] else self.n_support

        if (not self.use_scalar_product) and (not self.use_cosine_distance):
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
            support_embeddings_size = conv_out_size * self.n_way * self.n_support_size_context
        else:
            support_embeddings_size = 0

        if self.hn_kernel_invariance:
            if self.hn_kernel_invariance_type == 'attention':
                self.embedding_size: int = support_embeddings_size + (self.n_way * self.n_support_size_context)
            else:
                self.embedding_size: int = support_embeddings_size + self.hn_kernel_convolution_output_dim
        else:
            if self.no_self_relations:
                self.embedding_size: int = support_embeddings_size + (((self.n_way * self.n_support_size_context) ** 2) - (self.n_way * self.n_support_size_context) )
            else:
                self.embedding_size: int = support_embeddings_size + ((self.n_way * self.n_support_size_context) ** 2)

        # invariant operation type
        if self.hn_kernel_invariance:
            if self.hn_kernel_invariance_type == 'attention':
                self.init_kernel_transformer_architecture(params)
            else:
                self.init_kernel_convolution_architecture(params)

        self.query_relations_size = self.n_way * self.n_support_size_context
        self.target_net_architecture = target_net_architecture or self.build_target_net_architecture(params)
        self.init_hypernet_modules()

    def pw_cosine_distance(self, input_a, input_b):
        normalized_input_a = torch.nn.functional.normalize(input_a)
        normalized_input_b = torch.nn.functional.normalize(input_b)
        res = torch.mm(normalized_input_a, normalized_input_b.T)
        res *= -1 # 1-res without copy
        res += 1
        return res

    def build_target_net_architecture(self, params) -> nn.Module:
        tn_hidden_size = params.hn_tn_hidden_size
        layers = []
        if params.use_support_embeddings:
            common_insize = ((self.n_way * self.n_support_size_context) + self.feature.final_feat_dim)
        else:
            common_insize = (self.n_way * self.n_support_size_context)

        for i in range(params.hn_tn_depth):
            is_final = i == (params.hn_tn_depth - 1)
            insize = common_insize if i == 0 else tn_hidden_size
            outsize = self.n_way if is_final else tn_hidden_size
            layers.append(nn.Linear(insize, outsize))
            if not is_final:
                layers.append(nn.ReLU())

        res =  nn.Sequential(*layers)
        print(res)
        return res


    def process_few_shots(self, support_feature: torch.Tensor) -> torch.Tensor:
        """
        Process embeddings for few shot learning
        """
        if self.n_support > 1:
            if self.sup_aggregation == 'mean':
                return torch.mean(support_feature, axis=1).reshape(self.n_way, 1, -1)
            elif self.sup_aggregation == 'max_pooling':
                pooled, _ = torch.max(support_feature, axis=1)
                pooled = pooled.reshape(self.n_way, 1, -1)
                return pooled
            elif self.sup_aggregation == 'min_pooling':
                pooled, _ = torch.min(support_feature, axis=1)
                pooled = pooled.reshape(self.n_way, 1, -1)
                return pooled

        return support_feature


    def parse_feature(self, x, is_feature) -> Tuple[torch.Tensor, torch.Tensor]:
        support_feature, query_feature = super().parse_feature(x, is_feature)
        support_feature = self.process_few_shots(support_feature)
        return support_feature, query_feature

    def init_kernel_convolution_architecture(self, params):
        # TODO - add convolution-based approach
        self.kernel_1D_convolution: bool = True

    def init_kernel_transformer_architecture(self, params):
        self.kernel_transformer_layers_no: int = params.kernel_transformer_layers_no
        self.kernel_transformer_input_dim: int = self.n_way * self.n_support_size_context
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
        elif self.use_cosine_distance:
            kernel_values_tensor = self.pw_cosine_distance(support_features, feature_to_classify)
        else:
            kernel_values_tensor = self.kernel_function.forward(support_features, feature_to_classify)

        relations = kernel_values_tensor.T

        return relations

    def build_kernel_features_embedding(self, support_feature: torch.Tensor) -> torch.Tensor:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        supp_way, n_support, supp_feat = support_feature.shape
        support_features = support_feature.reshape(supp_way * n_support, supp_feat)
        support_features_copy = torch.clone(support_features)

        # TODO - check!!!
        if self.use_scalar_product:
            kernel_values_tensor = torch.matmul(support_features, support_features_copy.T)
        elif self.use_cosine_distance:
            kernel_values_tensor = self.pw_cosine_distance(support_features, support_features_copy)
        else:
            kernel_values_tensor = self.kernel_function.forward(support_features, support_features_copy)

        # Remove self relations by matrix multiplication
        if self.no_self_relations:
            zero_diagonal_matrix = torch.ones_like(kernel_values_tensor).cuda() - torch.eye(kernel_values_tensor.shape[0]).cuda()
            kernel_values_tensor = kernel_values_tensor * zero_diagonal_matrix
            return torch.flatten(kernel_values_tensor[kernel_values_tensor != 0.0])


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

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        embedding = self.build_kernel_features_embedding(support_feature)
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
        tn.support_feature = support_feature
        return tn.cuda()

    def set_forward(self, x: torch.Tensor, is_feature: bool = False, return_perm: bool = False):
        support_feature, query_feature = self.parse_feature(x, is_feature)

        classifier = self.generate_target_net(support_feature)
        query_feature = query_feature.reshape(
            -1, query_feature.shape[-1]
        )

        relational_query_feature = self.build_relations_features(support_feature, query_feature)
        # TODO - check!!!
        if self.use_support_embeddings:
            relational_query_feature = torch.cat((relational_query_feature, query_feature), 1)
        y_pred = classifier(relational_query_feature)

        ### random permutation test
        perm = torch.randperm(len(query_feature))
        rev_perm = torch.argsort(perm)
        qp = query_feature[perm]
        y_pred_perm = classifier(self.build_relations_features(support_feature, qp))
        assert torch.equal(y_pred_perm[rev_perm], y_pred)
        # print("perm test ok")
        ###
        if return_perm:
            return y_pred, (perm, rev_perm, y_pred_perm)

        return y_pred


    def set_forward_loss(
            self, x: torch.Tensor, detach_ft_hn: bool = False, detach_ft_tn: bool = False,
            train_on_support: bool = True,
            train_on_query: bool = True
    ):
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

        classifier = self.generate_target_net(feature_to_hn)

        feature_to_classify = []
        y_to_classify_gt = []
        if train_on_support:
            feature_to_classify.append(
                support_feature.reshape(
                    (self.n_way * self.n_support_size_context), support_feature.shape[-1]
                )
            )
            y_support = self.get_labels(support_feature)
            y_to_classify_gt.append(y_support.reshape(self.n_way * self.n_support_size_context))
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
                self.embedding_size: int = conv_out_size * self.n_way * self.n_support_size_context + (self.n_way * self.n_support_size_context)
            else:
                self.embedding_size: int = conv_out_size * self.n_way * self.n_support_size_context + self.hn_kernel_convolution_output_dim
        else:
            self.embedding_size: int = conv_out_size * self.n_way * self.n_support_size_context + ((self.n_way * self.n_support_size_context) * (self.n_way * self.n_query))

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
        self.kernel_transformer_input_dim: int = self.n_way * self.n_support_size_context
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

    def generate_target_net(self, support_feature: torch.Tensor, query_feature: torch.Tensor) -> nn.Module:
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

        classifier = self.generate_target_net(support_feature, query_feature)
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

        classifier = self.generate_target_net(feature_to_hn, query_feature_to_hn)

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


class ClassifierSupportQueryKernel(nn.Module):
    """A classifier (to be generated by hypernet) which calculates a kernel between support and query
    and treats that as logits."""
    def __init__(self, kernel: NNKernel, support: Optional[torch.Tensor] = None):
        super().__init__()
        self.kernel = kernel
        self.support = support

    def forward(self, query: torch.Tensor):
        way, n_support, feat = self.support.shape
        n_query, feat = query.shape

        similarities = self.kernel.forward(
            self.support.reshape(way * n_support, feat),
            query
        )

        res = similarities.reshape(way, n_support, n_query).sum(axis=1).T
        res = res - res.mean(axis=0)

        return res


class HNKernelBetweenSupportAndQuery(HyperNetPOC):
    """A hypernet which generates a `ClassifierSupportQueryKernel`"""
    def __init__(self, model_func, n_way: int, n_support: int, n_query: int):
        target_net_architecture = ClassifierSupportQueryKernel(NNKernel(64, 16, 1, 128))
        super().__init__(
            model_func, n_way, n_support, n_query, target_net_architecture=target_net_architecture)

    def taskset_repeats(self, epoch: int):
        return 1

    def generate_target_net(self, support_feature: torch.Tensor) -> ClassifierSupportQueryKernel:
        way, n_support, feat = support_feature.shape

        embedding = self.build_embedding(support_feature)
        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.hypernet_heads.items()
        }
        tn: ClassifierSupportQueryKernel = deepcopy(self.target_net_architecture)
        set_from_param_dict(tn, network_params)
        tn.support = support_feature
        return tn.cuda()


class NoHNKernelBetweenSupportAndQuery(HNKernelBetweenSupportAndQuery):
    """A non-hypernet which trains and returns a single instance of ClassifierSupportQueryKernel"""

    def generate_target_net(self, support_feature: torch.Tensor) -> ClassifierSupportQueryKernel:
        tn = self.target_net_architecture
        tn.support = support_feature
        return tn.cuda()