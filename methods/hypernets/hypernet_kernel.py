from copy import deepcopy
import math
from re import S
from typing import Optional, Tuple

import torch
from torch import nn

import numpy as np

from matplotlib import pyplot as plt

from backbone import BayesLinear, BayesLinear2
from utils import kl_diag_gauss_with_standard_gauss

from methods.hypernets import HyperNetPOC
from methods.hypernets.utils import set_from_param_dict, accuracy_from_scores
from methods.kernel_convolutions import KernelConv
from methods.kernels import init_kernel_function
from methods.transformer import TransformerEncoder


class HyperShot(HyperNetPOC):
    def __init__(
            self, model_func: nn.Module, n_way: int, n_support: int, n_query: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        super().__init__(
            model_func, n_way, n_support, n_query, params=params, target_net_architecture=target_net_architecture
        )

        self.loss_kld = kl_diag_gauss_with_standard_gauss
        self.S: int = params.hn_S # sampling
        self.bayesian_model = params.hn_bayesian_model
        self.use_kld = params.hn_use_kld

        # TODO - check!!!

        # Use support embeddings - concatenate them with kernel features
        self.hn_use_support_embeddings: bool = params.hn_use_support_embeddings
        # Remove self relations by matrix K multiplication
        self.hn_no_self_relations: bool = params.hn_no_self_relations

        self.kernel_function = init_kernel_function(
            kernel_input_dim=self.feat_dim + self.n_way if self.attention_embedding else self.feat_dim,
            params=params
        )

        # embedding size
        # TODO - add attention based input also
        self.embedding_size = self.init_embedding_size(params)

        # I will be adding the kernel vector to the stacked images embeddings
        # TODO: add/check changes for attention-like input
        self.hn_kernel_invariance: bool = params.hn_kernel_invariance
        if self.hn_kernel_invariance:
            self.hn_kernel_invariance_type: str = params.hn_kernel_invariance_type
            self.hn_kernel_invariance_pooling: str = params.hn_kernel_invariance_pooling

            if self.hn_kernel_invariance_type == 'attention':
                self.init_kernel_transformer_architecture(params)
            else:
                self.init_kernel_convolution_architecture(params)

        self.query_relations_size = self.n_way * self.n_support_size_context
        self.target_net_architecture = target_net_architecture or self.build_target_net_architecture(params)
        self.init_hypernet_modules()

    def init_embedding_size(self, params) -> int:
        if params.hn_use_support_embeddings:
            support_embeddings_size = self.feat_dim * self.n_way * self.n_support_size_context
        else:
            support_embeddings_size = 0

        if params.hn_kernel_invariance:
            if params.hn_kernel_invariance_type == 'attention':
                return support_embeddings_size + (self.n_way * self.n_support_size_context)
            else:
                return support_embeddings_size + params.hn_kernel_convolution_output_dim

        else:
            if params.hn_no_self_relations:
                return support_embeddings_size + (
                        ((self.n_way * self.n_support_size_context) ** 2) - (
                        self.n_way * self.n_support_size_context))
            else:
                return support_embeddings_size + ((self.n_way * self.n_support_size_context) ** 2)

    @property
    def n_support_size_context(self) -> int:
        return 1 if self.sup_aggregation in ["mean", "min_pooling", "max_pooling"] else self.n_support

    def build_target_net_architecture(self, params) -> nn.Module:
        tn_hidden_size = params.hn_tn_hidden_size
        layers = []
        if params.hn_use_support_embeddings:
            common_insize = ((self.n_way * self.n_support_size_context) + self.feat_dim)
        else:
            common_insize = (self.n_way * self.n_support_size_context)

        for i in range(params.hn_tn_depth):
            is_final = i == (params.hn_tn_depth - 1)
            insize = common_insize if i == 0 else tn_hidden_size
            outsize = self.n_way if is_final else tn_hidden_size
            if params.hn_blayer == 1:
                layers.append(BayesLinear2(insize, outsize, bias=True, bayesian=params.hn_bayesian_model, start_reparam = params.hn_start_reparam, steps_reparam = params.hn_steps_reparam))
            else:
                layers.append(BayesLinear(insize, outsize))

            if not is_final:
                layers.append(nn.ReLU())

        res = nn.Sequential(*layers)
        print(res)
        return res

    def maybe_aggregate_support_feature(self, support_feature: torch.Tensor) -> torch.Tensor:
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
        support_feature = self.maybe_aggregate_support_feature(support_feature)
        return support_feature, query_feature

    def init_kernel_convolution_architecture(self, params):
        # TODO - add convolution-based approach
        self.kernel_2D_convolution: bool = True
        self.kernel_conv: nn.Module = KernelConv(self.n_support, params.hn_kernel_convolution_output_dim)

    def init_kernel_transformer_architecture(self, params):
        kernel_transformer_input_dim: int = self.n_way * self.n_support_size_context
        self.kernel_transformer_encoder: nn.Module = TransformerEncoder(
            num_layers=params.kernel_transformer_layers_no,
            input_dim=kernel_transformer_input_dim,
            num_heads=params.kernel_transformer_heads_no,
            dim_feedforward=params.kernel_transformer_feedforward_dim
        )

    def build_relations_features(self, support_feature: torch.Tensor,
                                 feature_to_classify: torch.Tensor) -> torch.Tensor:

        supp_way, n_support, supp_feat = support_feature.shape
        n_examples, feat_dim = feature_to_classify.shape
        support_features = support_feature.reshape(supp_way * n_support, supp_feat)

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

        kernel_values_tensor = self.kernel_function.forward(support_features, support_features_copy)

        # Remove self relations by matrix multiplication
        if self.hn_no_self_relations:
            zero_diagonal_matrix = torch.ones_like(kernel_values_tensor).cuda() - torch.eye(
                kernel_values_tensor.shape[0]).cuda()
            kernel_values_tensor = kernel_values_tensor * zero_diagonal_matrix
            return torch.flatten(kernel_values_tensor[kernel_values_tensor != 0.0])

        if self.hn_kernel_invariance:
            # TODO - check!!!
            if self.hn_kernel_invariance_type == 'attention':
                kernel_values_tensor = torch.unsqueeze(kernel_values_tensor.T, 0)
                encoded = self.kernel_transformer_encoder.forward(kernel_values_tensor)

                if self.hn_kernel_invariance_pooling == 'min':
                    invariant_kernel_values, _ = torch.min(encoded, 1)
                elif self.hn_kernel_invariance_pooling == 'max':
                    invariant_kernel_values, _ = torch.max(encoded, 1)
                else:
                    invariant_kernel_values = torch.mean(encoded, 1)

                return invariant_kernel_values
            else:
                # TODO - add convolutional approach
                kernel_values_tensor = torch.unsqueeze(torch.unsqueeze(kernel_values_tensor.T, 0), 0)
                invariant_kernel_values = torch.flatten(self.kernel_conv.forward(kernel_values_tensor))

                return invariant_kernel_values

        return kernel_values_tensor

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        embedding = self.build_kernel_features_embedding(support_feature)
        embedding = embedding.reshape(1, self.embedding_size)
        # TODO - check!!!
        if self.hn_use_support_embeddings:
            embedding = torch.cat((embedding, torch.flatten(support_feature)), 0)

        root = self.hypernet_neck(embedding)
        network_params = {
            name.replace("-", "."): param_net(root).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.hypernet_heads.items()
        }

        tn = deepcopy(self.target_net_architecture)
        set_from_param_dict(tn, network_params)
        tn.support_feature = support_feature
        return tn.cuda(), network_params

    def set_forward(self, x: torch.Tensor, is_feature: bool = False, permutation_sanity_check: bool = False):
        support_feature, query_feature = self.parse_feature(x, is_feature)

        classifier, _ = self.generate_target_net(support_feature)
        query_feature = query_feature.reshape(
            -1, query_feature.shape[-1]
        )

        relational_query_feature = self.build_relations_features(support_feature, query_feature)
        # TODO - check!!!
        if self.hn_use_support_embeddings:
            relational_query_feature = torch.cat((relational_query_feature, query_feature), 1)
        y_pred = classifier(relational_query_feature)

        if permutation_sanity_check:
            ### random permutation test
            perm = torch.randperm(len(query_feature))
            rev_perm = torch.argsort(perm)
            query_perm = query_feature[perm]
            relation_perm = self.build_relations_features(support_feature, query_perm)
            assert torch.equal(relation_perm[rev_perm], relational_query_feature)
            y_pred_perm = classifier(relation_perm)
            assert torch.equal(y_pred_perm[rev_perm], y_pred)

        return y_pred

    def set_forward_with_adaptation(self, x: torch.Tensor):
        y_pred, metrics = super().set_forward_with_adaptation(x)
        support_feature, query_feature = self.parse_feature(x, is_feature=False)
        query_feature = query_feature.reshape(
            -1, query_feature.shape[-1]
        )
        relational_query_feature = self.build_relations_features(support_feature, query_feature)
        metrics["accuracy/val_relational"] = accuracy_from_scores(relational_query_feature, self.n_way, self.n_query)
        return y_pred, metrics

    def set_forward_loss(
            self, x: torch.Tensor, detach_ft_hn: bool = False, detach_ft_tn: bool = False,
            train_on_support: bool = True,
            train_on_query: bool = True,
            epoch: int = -1,
    ):
        nw, ne, c, h, w = x.shape

        support_feature, query_feature = self.parse_feature(x, is_feature=False)

        # TODO: add/check changes for attention-like input
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

        classifier, hn_out = self.generate_target_net(feature_to_hn)
        self.last_classifier = classifier

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

        if self.hn_use_support_embeddings:
            relational_feature_to_classify = torch.cat((relational_feature_to_classify, feature_to_classify), 1)

        total_crossentropy_loss = 0
        total_kld_loss = 0

        total_sigma = 0

        for _ in range(self.S):
            y_pred = classifier(relational_feature_to_classify)

            crossentropy_loss = 0
            kld_loss = 0      
            for m in classifier.modules() :
                if isinstance(m, (BayesLinear)):
                    w_mean, w_logvar = torch.tensor_split(m.weight, 2, dim=0)
                    b_mean, b_logvar = torch.tensor_split(m.bias, 2, dim=0)
                    if self.use_kld:
                        kld_loss += self.loss_kld(w_mean, w_logvar) + self.loss_kld(b_mean, b_logvar)
                elif isinstance(m, (BayesLinear2)):
                    if self.use_kld:
                        kld_loss += self.loss_kld(m.weight_mu, m.weight_log_var) + self.loss_kld(m.bias_mu, m.bias_log_var)
                        total_sigma += m.bias_log_var+m.weight_log_var
            crossentropy_loss += self.loss_fn(y_pred, y_to_classify_gt)

            # deprecated scaling (moved to hypernet_poc.py)
            #kld_loss *= self.kld_const/self.D #self.dataset_size
            #loss *= 1/(self.n_query+self.n_support)

            total_crossentropy_loss += crossentropy_loss
            total_kld_loss += kld_loss

        # divide by number of sampled predictions
        total_crossentropy_loss /= S
        total_kld_loss /= S

        print(f'Epoch {epoch}: {hn_out}')

        if epoch <= 2:
            return total_sigma/2, total_sigma/2, self.upload_mu_and_sigma_histogram(classifier, epoch)

        if self.use_kld:
            return total_crossentropy_loss, total_kld_loss, self.upload_mu_and_sigma_histogram(classifier, epoch)
        else:
            return total_crossentropy_loss, 0, self.upload_mu_and_sigma_histogram(classifier, epoch)

    def upload_mu_and_sigma_histogram(self, classifier : nn.Module, epoch : int):

        mu_weight = []
        mu_bias = []

        sigma_weight = []
        sigma_bias = []

        for module in classifier.modules():
            if isinstance(module, (BayesLinear2)):
                mu_weight.append(module.weight_mu.clone().data.cpu().numpy().flatten())
                mu_bias.append(module.bias_mu.clone().data.cpu().numpy().flatten())
                sigma_weight.append(torch.exp(0.5 * module.weight_log_var).clone().data.cpu().numpy().flatten())
                sigma_bias.append(torch.exp(0.5 * module.bias_log_var).clone().data.cpu().numpy().flatten())


        mu_weight = np.concatenate(mu_weight)
        mu_bias = np.concatenate(mu_bias)
        sigma_weight = np.concatenate(sigma_weight)
        sigma_bias = np.concatenate(sigma_bias)

        return {
            "mu_weight": mu_weight,
            "mu_bias": mu_bias,
            "sigma_weight": sigma_weight,
            "sigma_bias": sigma_bias
        }

    def get_mu_and_sigma(self):

        param_dict = {}

        i = 0

        for module in self.last_classifier.modules():
            if isinstance(module, (BayesLinear2)):

                weight_mu = module.weight_mu.clone().data.cpu().numpy().flatten()
                bias_mu = module.bias_mu.clone().data.cpu().numpy().flatten()
                weight_sigma = torch.exp(0.5 * module.weight_log_var).clone().data.cpu().numpy().flatten()
                bias_sigma = torch.exp(0.5 * module.bias_log_var).clone().data.cpu().numpy().flatten()

                param_dict[f"Layer {i+1} / weight_mu"] = weight_mu
                param_dict[f"Layer {i+1} / bias_mu"] = bias_mu
                param_dict[f"Layer {i+1} / weight_sigma"] = weight_sigma
                param_dict[f"Layer {i+1} / bias_sigma"] = bias_sigma
                i = i + 1
        
        return param_dict
