from collections import defaultdict
from copy import deepcopy
from time import time

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import backbone
from methods.hypernets.utils import get_param_dict, accuracy_from_scores, kl_diag_gauss_with_standard_gauss, \
    reparameterize
from methods.maml import MAML


# hypernetwork for target network params
class BHyperNet(nn.Module):
    def __init__(self, hn_hidden_size, n_way, embedding_size, feat_dim, out_neurons, params):
        super(BHyperNet, self).__init__()

        self.hn_head_len = params.hn_head_len

        head = [nn.Linear(embedding_size, hn_hidden_size), nn.ReLU()]

        if self.hn_head_len > 2:
            for i in range(self.hn_head_len - 2):
                head.append(nn.Linear(hn_hidden_size, hn_hidden_size))
                head.append(nn.ReLU())

        self.head = nn.Sequential(*head)

        # tails to equate weights with distributions
        tail_mean = [nn.Linear(hn_hidden_size, out_neurons)]
        tail_logvar = [nn.Linear(hn_hidden_size, out_neurons)]

        self.tail_mean = nn.Sequential(*tail_mean)
        self.tail_logvar = nn.Sequential(*tail_logvar)

    def forward(self, x):
        out = self.head(x)
        out_mean = self.tail_mean(out)
        out_logvar = self.tail_logvar(out)
        return out_mean, out_logvar


class BayesHMAML(MAML):
    def __init__(self, model_func, n_way, n_support, n_query, params=None, approx=False):
        super(BayesHMAML, self).__init__(model_func, n_way, n_support, n_query, params=params)
        # loss function components
        self.loss_fn = nn.CrossEntropyLoss()  # crossentropy
        self.loss_kld = kl_diag_gauss_with_standard_gauss  # Kullback–Leibler divergence
        self.kl_w = params.hm_kl_w
        self.kl_scale = params.kl_scale
        self.kl_step = None  # increase step for share of kld in loss
        self.kl_stop_val = params.kl_stop_val

        # num of weight set draws for softvoting
        self.weight_set_num_train = params.hm_weight_set_num_train  # train phase
        self.weight_set_num_test = params.hm_weight_set_num_test if params.hm_weight_set_num_test != 0 else None  # test phase

        # target network dims
        self.hn_tn_hidden_size = params.hn_tn_hidden_size
        self.hn_tn_depth = params.hn_tn_depth
        self._init_classifier()

        self.enhance_embeddings = params.hm_enhance_embeddings

        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

        self.hn_sup_aggregation = params.hn_sup_aggregation
        self.hn_hidden_size = params.hn_hidden_size
        self.hm_lambda = params.hm_lambda
        self.hm_save_delta_params = params.hm_save_delta_params
        self.hm_use_class_batch_input = params.hm_use_class_batch_input
        self.hn_adaptation_strategy = params.hn_adaptation_strategy
        self.hm_support_set_loss = params.hm_support_set_loss
        self.hm_maml_warmup = params.hm_maml_warmup
        self.hm_maml_warmup_epochs = params.hm_maml_warmup_epochs
        self.hm_maml_warmup_switch_epochs = params.hm_maml_warmup_switch_epochs
        self.hm_maml_update_feature_net = params.hm_maml_update_feature_net
        self.hm_update_operator = params.hm_update_operator
        self.hm_load_feature_net = params.hm_load_feature_net
        self.hm_feature_net_path = params.hm_feature_net_path
        self.hm_detach_feature_net = params.hm_detach_feature_net
        self.hm_detach_before_hyper_net = params.hm_detach_before_hyper_net
        self.hm_set_forward_with_adaptation = params.hm_set_forward_with_adaptation
        self.hn_val_lr = params.hn_val_lr
        self.hn_val_epochs = params.hn_val_epochs
        self.hn_val_optim = params.hn_val_optim

        self.alpha = 0
        self.hn_alpha_step = params.hn_alpha_step

        if self.hn_adaptation_strategy == 'increasing_alpha' and self.hn_alpha_step < 0:
            raise ValueError('hn_alpha_step is not positive!')

        self.single_test = False
        self.epoch = -1
        self.start_epoch = -1
        self.stop_epoch = -1

        self.calculate_embedding_size()

        self._init_hypernet_modules(params)
        self._init_feature_net()

    def _init_feature_net(self):
        if self.hm_load_feature_net:
            print(f'loading feature net model from location: {self.hm_feature_net_path}')
            model_dict = torch.load(self.hm_feature_net_path)
            self.feature.load_state_dict(model_dict['state'])

    def _init_classifier(self):
        assert self.hn_tn_hidden_size % self.n_way == 0, f"hn_tn_hidden_size {self.hn_tn_hidden_size} should be the multiple of n_way {self.n_way}"
        layers = []

        for i in range(self.hn_tn_depth):
            in_dim = self.feat_dim if i == 0 else self.hn_tn_hidden_size
            out_dim = self.n_way if i == (self.hn_tn_depth - 1) else self.hn_tn_hidden_size

            linear = backbone.Linear_fw(in_dim, out_dim)
            linear.bias.data.fill_(0)

            layers.append(linear)

        self.classifier = nn.Sequential(*layers)

    def _init_hypernet_modules(self, params):
        target_net_param_dict = get_param_dict(self.classifier)

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

        self.hypernet_heads = nn.ModuleDict()

        for name, param in target_net_param_dict.items():
            if self.hm_use_class_batch_input and name[-4:] == 'bias':
                # notice head_out val when using this strategy
                continue

            bias_size = param.shape[0] // self.n_way

            head_in = self.embedding_size
            head_out = (param.numel() // self.n_way) + bias_size if self.hm_use_class_batch_input else param.numel()
            # make hypernetwork for target network param
            self.hypernet_heads[name] = BHyperNet(self.hn_hidden_size, self.n_way, head_in, self.feat_dim, head_out,
                                                  params)

    def calculate_embedding_size(self):

        n_classes_in_embedding = 1 if self.hm_use_class_batch_input else self.n_way
        n_support_per_class = 1 if self.hn_sup_aggregation == 'mean' else self.n_support
        single_support_embedding_len = self.feat_dim + self.n_way + 1 if self.enhance_embeddings else self.feat_dim
        self.embedding_size = n_classes_in_embedding * n_support_per_class * single_support_embedding_len

    def apply_embeddings_strategy(self, embeddings):
        if self.hn_sup_aggregation == 'mean':
            new_embeddings = torch.zeros(self.n_way, *embeddings.shape[1:])

            for i in range(self.n_way):
                lower = i * self.n_support
                upper = (i + 1) * self.n_support
                new_embeddings[i] = embeddings[lower:upper, :].mean(dim=0)

            return new_embeddings.cuda()

        return embeddings

    def get_support_data_labels(self):
        return torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()  # labels for support data

    def get_hn_delta_params(self, support_embeddings):
        if self.hm_detach_before_hyper_net:
            support_embeddings = support_embeddings.detach()

        if self.hm_use_class_batch_input:
            delta_params_list = []

            for name, param_net in self.hypernet_heads.items():

                support_embeddings_resh = support_embeddings.reshape(
                    self.n_way, -1
                )

                delta_params_mean, params_logvar = param_net(support_embeddings_resh)
                bias_neurons_num = self.target_net_param_shapes[name][0] // self.n_way

                if self.hn_adaptation_strategy == 'increasing_alpha' and self.alpha < 1:
                    delta_params_mean = delta_params_mean * self.alpha
                    params_logvar = params_logvar * self.alpha

                weights_delta_mean = delta_params_mean[:, :-bias_neurons_num].contiguous().view(
                    *self.target_net_param_shapes[name])
                bias_delta_mean = delta_params_mean[:, -bias_neurons_num:].flatten()

                weights_logvar = params_logvar[:, :-bias_neurons_num].contiguous().view(
                    *self.target_net_param_shapes[name])
                bias_logvar = params_logvar[:, -bias_neurons_num:].flatten()

                delta_params_list.append([weights_delta_mean, weights_logvar])
                delta_params_list.append([bias_delta_mean, bias_logvar])
            return delta_params_list
        else:
            delta_params_list = []

            for name, param_net in self.hypernet_heads.items():

                flattened_embeddings = support_embeddings.flatten()

                delta_mean, logvar = param_net(flattened_embeddings)

                if name in self.target_net_param_shapes.keys():
                    delta_mean = delta_mean.reshape(self.target_net_param_shapes[name])
                    logvar = logvar.reshape(self.target_net_param_shapes[name])

                if self.hn_adaptation_strategy == 'increasing_alpha' and self.alpha < 1:
                    delta_mean = self.alpha * delta_mean
                    logvar = self.alpha * logvar

                delta_params_list.append([delta_mean, logvar])
            return delta_params_list

    def _update_weight(self, weight, update_value):
        if self.hm_update_operator == 'minus':
            if weight.fast is None:
                weight.fast = weight - update_value
            else:
                weight.fast = weight.fast - update_value
        elif self.hm_update_operator == 'plus':
            if weight.fast is None:
                weight.fast = weight + update_value
            else:
                weight.fast = weight.fast + update_value
        elif self.hm_update_operator == 'multiply':
            if weight.fast is None:
                weight.fast = weight * update_value
            else:
                weight.fast = weight.fast * update_value

    def _update_weight(self, weight, update_mean, logvar, train_stage=False):

        if update_mean is None and logvar is None:
            return
        if weight.mu is None:
            weight.mu = weight - update_mean
        else:
            weight.mu = weight.mu - update_mean

        if logvar is None:  # used in maml warmup
            weight.fast = []
            weight.fast.append(weight.mu)
        else:
            weight.logvar = logvar

            weight.fast = []
            if train_stage:
                for _ in range(self.weight_set_num_train):  # sample fast parameters for training
                    weight.fast.append(reparameterize(weight.mu, weight.logvar))
            else:
                if self.weight_set_num_test is not None:
                    for _ in range(self.weight_set_num_test):  # sample fast parameters for testing
                        weight.fast.append(reparameterize(weight.mu, weight.logvar))
                else:
                    weight.fast.append(weight.mu)  # return expected value

    def _scale_step(self):
        if self.kl_step is None:
            # scale step is calculated so that share of kld in loss increases kl_scale -> kl_stop_val
            self.kl_step = np.power(1 / self.kl_scale * self.kl_stop_val, 1 / self.stop_epoch)

        self.kl_scale = self.kl_scale * self.kl_step

    def _get_p_value(self):
        if self.epoch < self.hm_maml_warmup_epochs:
            return 1.0
        # warmup coef p decreases 1 -> 0
        elif self.hm_maml_warmup_epochs <= self.epoch < self.hm_maml_warmup_epochs + self.hm_maml_warmup_switch_epochs:
            return (self.hm_maml_warmup_switch_epochs + self.hm_maml_warmup_epochs - self.epoch) / (
                    self.hm_maml_warmup_switch_epochs + 1)
        return 0.0

    def _update_network_weights(self, delta_params_list, support_embeddings, support_data_labels, train_stage=False):
        if self.hm_maml_warmup and not self.single_test:
            p = self._get_p_value()

            if p > 0.0:
                fast_parameters = []

                if self.hm_maml_update_feature_net:
                    fet_fast_parameters = list(self.feature.parameters())
                    for weight in self.feature.parameters():
                        weight.fast = None
                    self.feature.zero_grad()
                    fast_parameters = fast_parameters + fet_fast_parameters

                clf_fast_parameters = list(self.classifier.parameters())
                for weight in self.classifier.parameters():
                    weight.fast = None
                    weight.mu = None
                    # weight.logvar = None
                self.classifier.zero_grad()
                fast_parameters = fast_parameters + clf_fast_parameters

                for task_step in range(self.task_update_num):
                    scores = self.classifier(support_embeddings)

                    set_loss = self.loss_fn(scores, support_data_labels)
                    reduction = self.kl_scale
                    for weight in self.classifier.parameters():
                        if weight.logvar is not None:
                            if weight.mu is not None:
                                set_loss = set_loss + self.kl_w * reduction * self.loss_kld(weight.mu, weight.logvar)
                            else:
                                set_loss = set_loss + self.kl_w * reduction * self.loss_kld(weight, weight.logvar)

                    grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True,
                                               allow_unused=True)  # build full graph support gradient of gradient

                    if self.approx:
                        grad = [g.detach() for g in
                                grad]  # do not calculate gradient of gradient if using first order approximation

                    if self.hm_maml_update_feature_net:
                        # update weights of feature network
                        for k, weight in enumerate(self.feature.parameters()):
                            update_value = self.train_lr * p * grad[k]
                            self._update_weight(weight, update_value)

                    classifier_offset = len(fet_fast_parameters) if self.hm_maml_update_feature_net else 0

                    if p == 1:
                        # update weights of classifier network by adding gradient
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = (self.train_lr * grad[classifier_offset + k])
                            update_mean, logvar = delta_params_list[k]
                            self._update_weight(weight, update_value, logvar, train_stage)

                    elif 0.0 < p < 1.0:
                        # update weights of classifier network by adding gradient and output of hypernetwork
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = self.train_lr * p * grad[classifier_offset + k]
                            update_mean, logvar = delta_params_list[k]
                            update_mean = (1 - p) * update_mean + update_value
                            self._update_weight(weight, update_mean, logvar, train_stage)
            else:
                for k, weight in enumerate(self.classifier.parameters()):
                    update_mean, logvar = delta_params_list[k]
                    self._update_weight(weight, update_mean, logvar, train_stage)
        else:
            for k, weight in enumerate(self.classifier.parameters()):
                update_mean, logvar = delta_params_list[k]
                self._update_weight(weight, update_mean, logvar, train_stage)

    def _get_list_of_delta_params(self, maml_warmup_used, support_embeddings, support_data_labels):
        # if not maml_warmup_used:

        if self.enhance_embeddings:
            with torch.no_grad():
                logits = self.classifier.forward(support_embeddings).detach()
                logits = F.softmax(logits, dim=1)

            labels = support_data_labels.view(support_embeddings.shape[0], -1)
            support_embeddings = torch.cat((support_embeddings, logits, labels), dim=1)

        for weight in self.parameters():
            weight.fast = None
        for weight in self.classifier.parameters():
            weight.mu = None
            # weight.logvar = None
        self.zero_grad()

        support_embeddings = self.apply_embeddings_strategy(support_embeddings)

        delta_params = self.get_hn_delta_params(support_embeddings)

        if self.hm_save_delta_params and len(self.delta_list) == 0:
            self.delta_list = [{'delta_params': delta_params}]

        return delta_params

    # else:
    #    return [torch.zeros(*i).cuda() for (_, i) in self.target_net_param_shapes.items()]

    def forward(self, x):
        out = self.feature.forward(x)

        if self.hm_detach_feature_net:
            out = out.detach()

        scores = self.classifier.forward(out)
        return scores

    # for neptune log
    # def _mu_sigma(self, calc_sigma):
    #     if calc_sigma:
    #         sigma = {}
    #         mu = {}
    #         for name, weight in self.classifier.named_parameters():
    #             m, logvar = weight.mu, weight.logvar
    #             logvar = torch.cat([t.view(-1) for t in logvar])
    #             sigma[name] = torch.exp(0.5 * logvar).clone().data.cpu().numpy()
    #             m = torch.cat([t.view(-1) for t in m])
    #             mu[name] = m.clone().data.cpu().numpy()
    #     else:
    #         mu = None
    #         sigma = None
    #
    #     return sigma, mu

    def set_forward(self, x, is_feature=False, train_stage=False):
        assert is_feature == False, 'MAML do not support fixed feature'

        x = x.cuda()
        x_var = Variable(x)
        support_data = x_var[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support,
                                                                            *x.size()[2:])  # support data
        query_data = x_var[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query,
                                                                          *x.size()[2:])  # query data
        support_data_labels = self.get_support_data_labels()

        support_embeddings = self.feature(support_data)

        if self.hm_detach_feature_net:
            support_embeddings = support_embeddings.detach()

        maml_warmup_used = (
                (not self.single_test) and self.hm_maml_warmup and (self.epoch < self.hm_maml_warmup_epochs))

        delta_params_list = self._get_list_of_delta_params(maml_warmup_used, support_embeddings, support_data_labels)

        self._update_network_weights(delta_params_list, support_embeddings, support_data_labels, train_stage)

        if self.hm_set_forward_with_adaptation and not train_stage:
            scores = self.forward(support_data)
            return scores, None
        else:
            if self.hm_support_set_loss and train_stage and not maml_warmup_used:
                query_data = torch.cat((support_data, query_data))

            scores = self.forward(query_data)

            # sum of delta params for regularization
            if self.hm_lambda != 0:
                total_delta_sum = sum([delta_params.pow(2.0).sum() for delta_params in delta_params_list])

                return scores, total_delta_sum
            else:
                return scores, None

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    # def set_forward_loss(self, x, calc_sigma):
    def set_forward_loss(self, x):
        scores, total_delta_sum = self.set_forward(x, is_feature=False, train_stage=True)

        # calc_sigma = calc_sigma and (self.epoch == self.stop_epoch - 1 or self.epoch % 100 == 0)
        # sigma, mu = self._mu_sigma(calc_sigma)

        query_data_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()
        if self.hm_support_set_loss:
            support_data_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
            query_data_labels = torch.cat((support_data_labels, query_data_labels))

        reduction = self.kl_scale

        loss_ce = self.loss_fn(scores, query_data_labels)

        loss_kld = torch.zeros_like(loss_ce)
        loss_kld_no_scale = torch.zeros_like(loss_ce)

        for name, weight in self.classifier.named_parameters():
            if weight.mu is not None and weight.logvar is not None:
                val = self.loss_kld(weight.mu, weight.logvar)
                loss_kld_no_scale = loss_kld_no_scale + val
                loss_kld = loss_kld + self.kl_w * reduction * val

        loss = loss_ce + loss_kld

        if self.hm_lambda != 0:
            loss = loss + self.hm_lambda * total_delta_sum

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = query_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(query_data_labels)) * 100

        return loss, loss_ce, loss_kld, loss_kld_no_scale, task_accuracy  # , sigma, mu

    def set_forward_loss_with_adaptation(self, x):
        scores, _ = self.set_forward(x, is_feature=False, train_stage=False)
        support_data_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()

        reduction = self.kl_scale

        loss_ce = self.loss_fn(scores, support_data_labels)

        loss_kld = torch.zeros_like(loss_ce)

        for name, weight in self.classifier.named_parameters():
            if weight.mu is not None and weight.logvar is not None:
                loss_kld = loss_kld + self.kl_w * reduction * self.loss_kld(weight.mu, weight.logvar)

        loss = loss_ce + loss_kld

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = support_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(support_data_labels)) * 100

        return loss, task_accuracy

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        loss_ce_all = []
        loss_kld_all = []
        loss_kld_no_scale_all = []
        acc_all = []
        optimizer.zero_grad()

        self.delta_list = []

        # train
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"

            # calc_sigma = i + 1 == len(train_loader)
            # loss, loss_ce, loss_kld, loss_kld_no_scale, task_accuracy, sigma, mu = self.set_forward_loss(x, calc_sigma)
            # loss, loss_ce, loss_kld, loss_kld_no_scale, task_accuracy = self.set_forward_loss(x, calc_sigma)
            loss, loss_ce, loss_kld, loss_kld_no_scale, task_accuracy = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()  # .data[0]
            loss_all.append(loss)
            loss_ce_all.append(loss_ce.item())
            loss_kld_all.append(loss_kld.item())
            loss_kld_no_scale_all.append(loss_kld_no_scale.item())
            acc_all.append(task_accuracy)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []

            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(self.epoch, self.stop_epoch, i,
                                                                             len(train_loader),
                                                                             avg_loss / float(i + 1)))

        self._scale_step()

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)

        metrics = {"accuracy/train": acc_mean}

        loss_ce_all = np.asarray(loss_ce_all)
        loss_ce_mean = np.mean(loss_ce_all)

        metrics["loss_ce"] = loss_ce_mean

        loss_kld_all = np.asarray(loss_kld_all)
        loss_kld_mean = np.mean(loss_kld_all)

        metrics["loss_kld"] = loss_kld_mean

        loss_kld_no_scale_all = np.asarray(loss_kld_no_scale_all)
        loss_kld_no_scale_mean = np.mean(loss_kld_no_scale_all)

        metrics["loss_kld_no_scale"] = loss_kld_no_scale_mean

        if self.hn_adaptation_strategy == 'increasing_alpha':
            metrics['alpha'] = self.alpha

        if self.hm_save_delta_params and len(self.delta_list) > 0:
            delta_params = {"epoch": self.epoch, "delta_list": self.delta_list}
            metrics['delta_params'] = delta_params

        if self.alpha < 1:
            self.alpha += self.hn_alpha_step

        return metrics  # , sigma, mu

    def test_loop(self, test_loader, return_std=False, return_time: bool = False):  # overwrite parrent function

        acc_all = []
        self.delta_list = []
        acc_at = defaultdict(list)

        iter_num = len(test_loader)

        eval_time = 0

        if self.hm_set_forward_with_adaptation:
            for i, (x, _) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(0), "MAML do not support way change"
                s = time()
                acc_task, acc_at_metrics = self.set_forward_with_adaptation(x)
                t = time()
                for (k, v) in acc_at_metrics.items():
                    acc_at[k].append(v)
                acc_all.append(acc_task)
                eval_time += (t - s)

        else:
            for i, (x, _) in enumerate(test_loader):
                # print(x.shape)
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(0), f"MAML do not support way change, {self.n_way=}, {x.size(0)=}"
                s = time()
                correct_this, count_this = self.correct(x)
                t = time()
                acc_all.append(correct_this / count_this * 100)
                eval_time += (t - s)

        metrics = {
            k: np.mean(v) if len(v) > 0 else 0
            for (k, v) in acc_at.items()
        }

        num_tasks = len(acc_all)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print("Num tasks", num_tasks)

        ret = [acc_mean]
        if return_std:
            ret.append(acc_std)
        if return_time:
            ret.append(eval_time)
        ret.append(metrics)

        return ret

    def set_forward_with_adaptation(self, x: torch.Tensor):
        self_copy = deepcopy(self)

        # deepcopy does not copy "fast" parameters so it should be done manually
        for param1, param2 in zip(self.feature.parameters(), self_copy.feature.parameters()):
            if hasattr(param1, 'fast'):
                if param1.fast is not None:
                    param2.fast = param1.fast.clone()
                else:
                    param2.fast = None

        for param1, param2 in zip(self.classifier.parameters(), self_copy.classifier.parameters()):
            if hasattr(param1, 'fast'):
                if param1.fast is not None:
                    param2.fast = list(param1.fast)
                else:
                    param2.fast = None
            if hasattr(param1, 'mu'):
                if param1.mu is not None:
                    param2.mu = param1.mu.clone()
                else:
                    param2.mu = None
            if hasattr(param1, 'logvar'):
                if param1.logvar is not None:
                    param2.logvar = param1.logvar.clone()
                else:
                    param2.logvar = None

        metrics = {
            "accuracy/val@-0": self_copy.query_accuracy(x)
        }

        val_opt_type = torch.optim.Adam if self.hn_val_optim == "adam" else torch.optim.SGD
        val_opt = val_opt_type(self_copy.parameters(), lr=self.hn_val_lr)

        if self.hn_val_epochs > 0:
            for i in range(1, self.hn_val_epochs + 1):
                self_copy.train()
                val_opt.zero_grad()
                loss, val_support_acc = self_copy.set_forward_loss_with_adaptation(x)
                loss.backward()
                val_opt.step()
                self_copy.eval()
                metrics[f"accuracy/val_support_acc@-{i}"] = val_support_acc
                metrics[f"accuracy/val_loss@-{i}"] = loss.item()
                metrics[f"accuracy/val@-{i}"] = self_copy.query_accuracy(x)

        # free CUDA memory by deleting "fast" parameters
        for param in self_copy.parameters():
            param.fast = None
            param.mu = None
            param.logvar = None

        return metrics[f"accuracy/val@-{self.hn_val_epochs}"], metrics

    def query_accuracy(self, x: torch.Tensor) -> float:
        scores, _ = self.set_forward(x, train_stage=True)
        return 100 * accuracy_from_scores(scores, n_way=self.n_way, n_query=self.n_query)

    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        logits, _ = self.set_forward(x)
        return logits

    def correct(self, x):
        scores, _ = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)
