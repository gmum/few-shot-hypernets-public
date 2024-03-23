from collections import defaultdict
from typing import Tuple

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test

    def upload_mu_and_sigma_histogram(self, classifier : nn.Module, test = False):

        mu_weight = []
        mu_bias = []

        sigma_weight = []
        sigma_bias = []

        for module in classifier.modules():
            if isinstance(module, (BayesLinear)):
                mu_weight.append(module.weight_mu.clone().data.cpu().numpy().flatten())
                mu_bias.append(module.bias_mu.clone().data.cpu().numpy().flatten())
                sigma_weight.append(torch.exp(0.5 * (module.weight_log_var-4)).clone().data.cpu().numpy().flatten())
                sigma_bias.append(torch.exp(0.5 * (module.bias_log_var-4)).clone().data.cpu().numpy().flatten())


        mu_weight = np.concatenate(mu_weight)
        mu_bias = np.concatenate(mu_bias)
        sigma_weight = np.concatenate(sigma_weight)
        sigma_bias = np.concatenate(sigma_bias)

        if not test:
            return {
                "mu_weight": mu_weight,
                "mu_bias": mu_bias,
                "sigma_weight": sigma_weight,
                "sigma_bias": sigma_bias
            }
        else:
            return {
                "mu_weight_test": mu_weight,
                "mu_bias_test": mu_bias,
                "sigma_weight_test": sigma_weight,
                "sigma_bias_test": sigma_bias
            }

    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature) -> Tuple[torch.Tensor, torch.Tensor]:
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):       
        scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer ):            
        print_freq = 10

        avg_loss=0
        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))

    def test_loop(self, test_loader, record = None, return_std: bool = False, epoch: int = -1):
        correct =0
        count = 0
        acc_all = []
        acc_at = defaultdict(list)

        bnn_params_dict =  {
                "mu_weight_test": [],
                "mu_bias_test": [],
                "sigma_weight_test": [],
                "sigma_bias_test": []
        }
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            y_query = np.repeat(range( self.n_way ), self.n_query )

            try:
                scores, bayesian_params_dict, acc_at_metrics = self.set_forward_with_adaptation(x)

                # append from current eval
                bnn_params_dict["mu_weight_test"].append(bayesian_params_dict["mu_weight_test"])
                bnn_params_dict["mu_bias_test"].append(bayesian_params_dict["mu_bias_test"])
                bnn_params_dict["sigma_weight_test"].append(bayesian_params_dict["sigma_weight_test"])
                bnn_params_dict["sigma_bias_test"].append(bayesian_params_dict["sigma_bias_test"])

                for (k,v) in acc_at_metrics.items():
                    acc_at[k].append(v)
            except Exception as e:
                scores, bayesian_params_dict = self.set_forward(x)
                # append from current eval
                bnn_params_dict["mu_weight_test"].append(bayesian_params_dict["mu_weight_test"])
                bnn_params_dict["mu_bias_test"].append(bayesian_params_dict["mu_bias_test"])
                bnn_params_dict["sigma_weight_test"].append(bayesian_params_dict["sigma_weight_test"])
                bnn_params_dict["sigma_bias_test"].append(bayesian_params_dict["sigma_bias_test"])

            scores = scores.reshape((self.n_way * self.n_query, self.n_way))

            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query)
            correct_this = float(top1_correct)
            count_this = len(y_query)
            acc_all.append(correct_this/ count_this*100)

        metrics = {
            k: np.mean(v) if len(v) > 0 else 0
            for (k,v) in acc_at.items()
        }

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print(metrics)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))


        # convert list of numpy arrays to numpy arrays

        bnn_params_dict =  {
                f"mu_weight_test_mean@{epoch}": np.concatenate(bnn_params_dict["mu_weight_test"]).mean(axis=0),
                f"mu_bias_test_mean@{epoch}": np.concatenate(bnn_params_dict["mu_bias_test"]).mean(axis=0),
                f"sigma_weight_test_mean@{epoch}": np.concatenate(bnn_params_dict["sigma_weight_test"]).mean(axis=0),
                f"sigma_bias_test_mean@{epoch}": np.concatenate(bnn_params_dict["sigma_bias_test"]).mean(axis=0),
                f"mu_weight_test_std@{epoch}": np.concatenate(bnn_params_dict["mu_weight_test"]).std(axis=0),
                f"mu_bias_test_std@{epoch}": np.concatenate(bnn_params_dict["mu_bias_test"]).std(axis=0),
                f"sigma_weight_test_std@{epoch}": np.concatenate(bnn_params_dict["sigma_weight_test"]).std(axis=0),
                f"sigma_bias_test_std@{epoch}": np.concatenate(bnn_params_dict["sigma_bias_test"]).std(axis=0)
        }

        if return_std:
            return acc_mean, acc_std, metrics, bnn_params_dict
        else:
            return acc_mean, metrics, bnn_params_dict

    def set_forward_adaptation(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
