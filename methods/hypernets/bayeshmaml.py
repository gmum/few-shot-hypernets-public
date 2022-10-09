from copy import deepcopy
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import backbone
from methods.hypernets.utils import get_param_dict, kl_diag_gauss_with_standard_gauss, \
    reparameterize
from methods.hypernets.hypermaml import HyperMAML


class BHyperNet(nn.Module):
    """bayesian hypernetwork for target network params"""

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


class BayesHMAML(HyperMAML):
    def __init__(self, model_func, n_way, n_support, n_query, params=None, approx=False):
        super(BayesHMAML, self).__init__(model_func, n_way, n_support, n_query, approx=approx, params=params)
        # loss function component
        self.loss_kld = kl_diag_gauss_with_standard_gauss  # Kullback–Leibler divergence
        self.kl_scale = params.kl_scale
        self.kl_step = None  # increase step for share of kld in loss
        self.kl_stop_val = params.kl_stop_val

        # num of weight set draws for softvoting
        self.weight_set_num_train = params.hm_weight_set_num_train  # train phase
        self.weight_set_num_test = params.hm_weight_set_num_test if params.hm_weight_set_num_test != 0 else None  # test phase


    def _init_classifier(self):
        assert self.hn_tn_hidden_size % self.n_way == 0, f"hn_tn_hidden_size {self.hn_tn_hidden_size} should be the multiple of n_way {self.n_way}"
        layers = []

        for i in range(self.hn_tn_depth):
            in_dim = self.feat_dim if i == 0 else self.hn_tn_hidden_size
            out_dim = self.n_way if i == (self.hn_tn_depth - 1) else self.hn_tn_hidden_size

            linear = backbone.BLinear_fw(in_dim, out_dim)
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

    def _update_weight(self, weight, update_mean, logvar, train_stage=False):
        """ get distribution associated with weight. Sample weights for target network. """
        if update_mean is None and logvar is None:
            return
        # if weight.mu is None:
        if not hasattr(weight, 'mu') or weight.mu is None:
            weight.mu = None
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
        """calculate regularization step for kld"""
        if self.kl_step is None:
            # scale step is calculated so that share of kld in loss increases kl_scale -> kl_stop_val
            self.kl_step = np.power(1 / self.kl_scale * self.kl_stop_val, 1 / self.stop_epoch)

        self.kl_scale = self.kl_scale * self.kl_step

    def _get_p_value(self):
        if self.epoch < self.hm_maml_warmup_epochs:
            return 1.0

        elif self.hm_maml_warmup_epochs <= self.epoch < self.hm_maml_warmup_epochs + self.hm_maml_warmup_switch_epochs:
            return (self.hm_maml_warmup_switch_epochs + self.hm_maml_warmup_epochs - self.epoch) / (
                    self.hm_maml_warmup_switch_epochs + 1)
        return 0.0


    def _update_network_weights(self, delta_params_list, support_embeddings, support_data_labels, train_stage=False):
        if self.hm_maml_warmup and not self.single_test:
            p = self._get_p_value()
            # warmup coef p decreases 1 -> 0
            if p > 0.0:
                fast_parameters = []

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
                                # set_loss = set_loss + self.kl_w * reduction * self.loss_kld(weight.mu, weight.logvar)
                                set_loss = set_loss + reduction * self.loss_kld(weight.mu, weight.logvar)
                            else:
                                # set_loss = set_loss + self.kl_w * reduction * self.loss_kld(weight, weight.logvar)
                                set_loss = set_loss + reduction * self.loss_kld(weight, weight.logvar)

                    grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True,
                                               allow_unused=True)  # build full graph support gradient of gradient

                    if self.approx:
                        grad = [g.detach() for g in
                                grad]  # do not calculate gradient of gradient if using first order approximation

                    if p == 1:
                        # update weights of classifier network by adding gradient
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = (self.train_lr * grad[k])
                            update_mean, logvar = delta_params_list[k]
                            self._update_weight(weight, update_value, logvar, train_stage)

                    elif 0.0 < p < 1.0:
                        # update weights of classifier network by adding gradient and output of hypernetwork
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = self.train_lr * p * grad[k]
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

    def set_forward_loss(self, x):
        """Adapt and forward using x. Return scores and total losses"""
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

        for name, weight in self.classifier.named_parameters():
            if weight.mu is not None and weight.logvar is not None:
                val = self.loss_kld(weight.mu, weight.logvar)
                # loss_kld = loss_kld + self.kl_w * reduction * val
                loss_kld = loss_kld + reduction * val

        loss = loss_ce + loss_kld

        if self.hm_lambda != 0:
            loss = loss + self.hm_lambda * total_delta_sum

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = query_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(query_data_labels)) * 100

        return loss, loss_ce, loss_kld, task_accuracy


    def set_forward_loss_with_adaptation(self, x):
        """returns loss and accuracy from adapted model (copy)"""
        scores, _ = self.set_forward(x, is_feature=False, train_stage=False)  # scores from adapted copy
        support_data_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()

        reduction = self.kl_scale

        loss_ce = self.loss_fn(scores, support_data_labels)

        loss_kld = torch.zeros_like(loss_ce)

        for name, weight in self.classifier.named_parameters():
            if weight.mu is not None and weight.logvar is not None:
                # loss_kld = loss_kld + self.kl_w * reduction * self.loss_kld(weight.mu, weight.logvar)
                loss_kld = loss_kld + reduction * self.loss_kld(weight.mu, weight.logvar)

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
        # loss_kld_no_scale_all = []
        acc_all = []
        optimizer.zero_grad()

        self.delta_list = []

        # train
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"

            loss, loss_ce, loss_kld, task_accuracy = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()  # .data[0]
            loss_all.append(loss)
            loss_ce_all.append(loss_ce.item())
            loss_kld_all.append(loss_kld.item())
            # loss_kld_no_scale_all.append(loss_kld_no_scale.item())
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

        if self.hn_adaptation_strategy == 'increasing_alpha':
            metrics['alpha'] = self.alpha

        if self.hm_save_delta_params and len(self.delta_list) > 0:
            delta_params = {"epoch": self.epoch, "delta_list": self.delta_list}
            metrics['delta_params'] = delta_params

        if self.alpha < 1:
            self.alpha += self.hn_alpha_step

        return metrics

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
