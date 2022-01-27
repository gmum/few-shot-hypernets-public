"""
Architectures we tried out but didn't pursue out for one reason or another
"""
from copy import deepcopy
from typing import Optional

import torch
from torch import nn

from methods.hypernets import HyperNetPOC
from methods.hypernets.utils import set_from_param_dict, accuracy_from_scores
from methods.kernels import NNKernel


class HNPocAdaptTN(HyperNetPOC):
    """
    A variant where in adaptation we tune the weights of the generated target network instead of the entire hypernet.
    """
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
    """
    A variant where the final layer of the classifier is universal
    (so the target network must output something which fits into this final layer and is classifiable,
    instead of a classifier)
    """
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


class HyperNetConvFromDKT(HyperNetPOC):
    def __init__(
            self, model_func: nn.Module, n_way: int, n_support: int, n_query: int,
            params: "ArgparseHNParams", target_net_architecture: Optional[nn.Module] = None
    ):
        """Backbone initialized from pre-trained DKT"""
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
    """
    Additional processsing of support examples after backbone.
    First, they are fed separately through the individual processor and then their representations are joined
    and fed through the joint processor.
    """
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
        """The support array ([n_support x n_way x feature_shape]) is jointly processed by a convolution"""
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
    """
    Target network (which classifies the embeddings of query)
    is generated from a kernel between support and support. This obv. did not work.
    """
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


class ConditionedClassifier(nn.Module):
    """A classifier which classifies the query embeddings concatenated embeddings of all supports."""
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
    """A non-hypernet, which trains one instance of ConditionedClassifier"""
    def __init__(self, model_func, n_way: int, n_support: int, n_query: int):
        target_net_architecture = ConditionedClassifier(64, n_way, n_support)

        super().__init__(model_func, n_way, n_support, n_query, target_net_architecture=target_net_architecture)

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        tn = self.target_net_architecture
        tn.support = support_feature
        return tn.cuda()