from copy import deepcopy

import torch
from torch import nn
from typing import Dict

from methods.meta_template import MetaTemplate


class HyperNetPOC(MetaTemplate):
    def __init__(self, model_func, n_way: int, n_support: int):
        super().__init__(model_func, n_way, n_support)
        print(self.feature.final_feat_dim)
        hidden_size = 64 # final conv size

        target_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_way)
        )

        param_head_size = hidden_size * self.n_way * self.n_support

        param_dict = get_param_dict(target_network)

        param_dict = {
            name.replace(".", "-"): p
            for name, p in param_dict.items()
        }

        self.param_shapes = {
            name: p.shape
            for (name, p)
            in param_dict.items()
        }

        self.param_nets = nn.ModuleDict()

        for name, param in param_dict.items():
            self.param_nets[name] = nn.Sequential(
                nn.Linear(param_head_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, param.numel())
            )
        self.target_network = target_network
        self.loss_fn = nn.CrossEntropyLoss()

        self.taskset_size = 32

    def taskset_epochs(self, progress_id: int):
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

        nw, ns, feat = support_feature.shape
        features = support_feature.reshape(nw * ns, feat)

        embedding = features.reshape(1, -1)

        network_params = {
            name.replace("-", "."): param_net(embedding).reshape(self.param_shapes[name])
            for name, param_net in self.param_nets.items()
        }
        tn = deepcopy(self.target_network)
        set_from_param_dict(tn, network_params)
        return tn.cuda()

    def set_forward(self, x: torch.Tensor, is_feature: bool=False):
        import matplotlib.pyplot as plt


        # print(x.shape)
        # fig, ax = plt.subplots(5, 5, figsize=(15,15))
        # for i in range(5):
        #     for j in range(5):
        #         ax[i, j].imshow(x[i,j,0])
        #         ax[i,j].axis("off")
        # plt.show()
        support_feature, query_feature = self.parse_feature(x, is_feature)

        classifier = self.generate_target_net(support_feature)
        query_feature = query_feature.reshape(
            -1, query_feature.shape[-1]
        )
        y_pred = classifier(query_feature)
        return y_pred

    def set_forward_loss(self, x: torch.Tensor):
        # y = self.get_labels(x)
        nw, ne, c, h, w = x.shape
        # y = y.reshape(nw * ne)

        support_feature, query_feature = self.parse_feature(x, is_feature=False)


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
        y_pred = classifier(all_feature)
        return self.loss_fn(y_pred, all_y, )

    def train_loop(self, epoch, train_loader, optimizer ):

        taskset_id = 0
        taskset = []

        n_train = len(train_loader)
        for i, (x,_) in enumerate(train_loader):
            taskset.append(x)

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

                print(f"Epoch {epoch} | Taskset {taskset_id} | TS {len(taskset)} | TS epochs {ts_epochs} | Loss {loss_sum.item()}")
                taskset_id += 1
                taskset = []







            # if i % print_freq==0:
            #     #print(optimizer.state_dict()['param_groups'][0]['lr'])
            #     print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))


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
