from copy import deepcopy

import numpy as np
import torch
from torch import nn
from typing import Dict
from torch.autograd import Variable

from methods.meta_template import MetaTemplate
from methods.transformer import TransformerEncoder
from backbone import ConvNetS

class HyperNetPOC(MetaTemplate):
    def __init__(self, model_func, n_way: int, n_support: int):
        super().__init__(model_func, n_way, n_support)
        print(self.feature.final_feat_dim)

        conv_out_size = 64 # final conv size
        #hidden_size = 512
        hidden_size = 512
        #param_head_size = conv_out_size * self.n_way * self.n_support
        param_head_size = (conv_out_size + self.n_way) * self.n_way * self.n_support
        #param_head_size = conv_out_size + self.n_way

        self.taskset_size = 1
        self.taskset_print_every = 20

        self.transformers_layers_no = 1
        self.transformer_input_dim = conv_out_size + self.n_way
        self.transformer_heads = 1
        self.transformer_dim_feedforward = 512
        self.transformer_encoder = TransformerEncoder(num_layers=self.transformers_layers_no, input_dim=self.transformer_input_dim, num_heads=self.transformer_heads, dim_feedforward=self.transformer_dim_feedforward)


        # TODO #1 - tweak the architecture of the target network
        target_network_architecture = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_way)
        )
        #target_network_architecture = nn.Sequential(
        #    ConvNetS(4),
        #    nn.ReLU(),
        #    nn.Linear(conv_out_size, self.n_way)
        #)

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
                nn.Linear(param_head_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, param.numel())
            )
        #for name, param in target_net_param_dict.items():
        #    self.target_net_param_predictors[name] = nn.Sequential(
        #        nn.Linear(param_head_size, hidden_size),
        #        nn.ReLU(),
        #        nn.Linear(hidden_size, hidden_size),
        #        nn.ReLU(),
        #        nn.Linear(hidden_size, hidden_size),
        #        nn.ReLU(),
        #        nn.Linear(hidden_size, hidden_size),
        #        nn.ReLU(),
        #        nn.Linear(hidden_size, hidden_size),
        #        nn.ReLU(),
        #        nn.Linear(hidden_size, param.numel())
        #    )
        self.target_network_architecture = target_network_architecture
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.feature_extractor = self.feature

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
        x: [n_way, (n_support+n_query), c, h, w]
        """
        ys = torch.tensor(list(range(x.shape[0]))).reshape(len(x), 1)
        ys = ys.repeat(1, x.shape[1]).to(x.device)
        return ys.cuda()

    def generate_target_net(self, support_feature: torch.Tensor) -> nn.Module:
        """
        x_support: [n_way, n_support, hidden_size]
        """

        support_feature = support_feature.view(1, -1, *(support_feature.size()[2:]))
        attention_features = self.transformer_encoder.forward(support_feature)
        #embedding = torch.mean(attention_features, (0, 1))
        embedding = torch.flatten(attention_features)
        #nw, ns, feat = support_feature.shape
        #features = support_feature.reshape(nw * ns, feat)

        #embedding = features.reshape(1, -1)

        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.target_net_param_shapes[name])
            for name, param_net in self.target_net_param_predictors.items()
        }
        tn = deepcopy(self.target_network_architecture)
        set_from_param_dict(tn, network_params)
        return tn.cuda()

    def set_forward(self, x: torch.Tensor, is_feature: bool=False):
        support_feature, query_feature = self.parse_feature(x, is_feature)
        #print('support_feature')
        #print(support_feature.shape)
        #print('query_feature')
        #print(query_feature.shape)
        
        y_support = self.get_labels(support_feature)
        y_support_one_hot = torch.nn.functional.one_hot(y_support)
        support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
        classifier = self.generate_target_net(support_feature_with_classes_one_hot)
        #classifier = self.generate_target_net(support_feature)
        

        query_feature = query_feature.reshape(
            -1, query_feature.shape[-1]
        )
        

        y_pred = classifier(query_feature)
        #x_clone = x.clone()
        #x_clone = Variable(x_clone.cuda())
        #print('x_clone')
        #print(x_clone.shape)
        #print(x_clone.contiguous().view( self.n_way * (self.n_support + self.n_query), *x_clone.size()[2:]).shape)
        #print(x_clone.contiguous().view( self.n_way * (self.n_support + self.n_query), *x_clone.size()[2:])[self.n_way*self.n_support:].shape)
        #y_pred = classifier(x_clone.contiguous().view( self.n_way * (self.n_support + self.n_query), *x_clone.size()[2:])[self.n_way*self.n_support:])
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
        #print('support_feature')
        #print(support_feature.shape)
        #print('query_feature')
        #print(query_feature.shape)

        y_support = self.get_labels(support_feature)
        y_query = self.get_labels(query_feature)
        y_support_one_hot = torch.nn.functional.one_hot(y_support)
        support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
        classifier = self.generate_target_net(support_feature_with_classes_one_hot)
        #classifier = self.generate_target_net(support_feature)

        all_feature = torch.cat(
            [
                support_feature.reshape(
                    (self.n_way * self.n_support), support_feature.shape[-1]
                ),
                query_feature.reshape(
                    (self.n_way * (ne - self.n_support)), query_feature.shape[-1]
                )
            ])

        #y_support = self.get_labels(support_feature)
        #y_query = self.get_labels(query_feature)
        #print('y_support')
        #print(y_support.shape)
        #print(y_support)
        #y_support_one_hot = torch.nn.functional.one_hot(y_support)
        #print('y_support_one_hot')
        #print(y_support_one_hot.shape)
        #print(y_support_one_hot)
        #y_support_unsqueeze = torch.unsqueeze(y_support, 2)
        #print('y_support_unsqueeze')
        #print(y_support_unsqueeze.shape)
        #print(y_support_unsqueeze)
        #support_feature_with_classes = torch.cat((support_feature, y_support_unsqueeze), 2)
        #support_feature_with_classes_one_hot = torch.cat((support_feature, y_support_one_hot), 2)
        #print('support_feature_with_classes')
        #print(support_feature_with_classes.shape)
        #print('support_feature_with_classes_one_hot')
        #print(support_feature_with_classes_one_hot.shape)

        #print('y_query')
        #print(y_query)
        #print(y_query.shape)
        all_y = torch.cat([
            y_support.reshape(self.n_way * self.n_support),
            y_query.reshape(self.n_way * (ne - self.n_support))
        ])
        #print('all_feature')
        #print(all_feature.shape)
        

        y_pred = classifier(all_feature)
        
        #x_clone = x.clone()
        #x_clone = Variable(x_clone.cuda())
        #y_pred = classifier(x_clone.contiguous().view( self.n_way * (self.n_support + self.n_query), *x_clone.size()[2:]))
        

        #print('y_all')
        #print(all_y.shape)
        #print(all_y)
        #print('y_pred')
        #print(y_pred.shape)
        #print(y_pred)
        return self.loss_fn(y_pred, all_y, )

    def train_loop(self, epoch, train_loader, optimizer ):
        optimizer = torch.optim.Adam([{'params': self.target_net_param_predictors.parameters(), 'lr': 1e-4},
                                      {'params': self.transformer_encoder.parameters(), 'lr': 1e-4},
                                      {'params': self.feature_extractor.parameters(), 'lr': 1e-4}])

        taskset_id = 0
        taskset = []
        n_train = len(train_loader)
        accuracies = []
        for i, (x,_) in enumerate(train_loader):
            taskset.append(x)

            # TODO 3: perhaps the idea of tasksets is redundant and it's better to update weights at every task
            if i % self.taskset_size == (self.taskset_size-1) or i == (n_train-1):
                ts_epochs = self.taskset_epochs(epoch)
                loss_sum = torch.tensor(0).cuda()
                for e in range(ts_epochs):
                    loss_sum = torch.tensor(0).cuda()

                    for task_x in taskset:
                        self.n_query = task_x.size(1) - self.n_support
                        if self.change_way:
                            self.n_way = task_x.size(0)
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
