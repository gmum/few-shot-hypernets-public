# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import torch
import backbone
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from backbone import Conv4

from copy import deepcopy
from typing import Dict, Optional
from collections import defaultdict
from torch.autograd import Variable
from methods.meta_template import MetaTemplate
from methods.hypernets.utils import get_param_dict, set_from_param_dict, SinActivation, accuracy_from_scores
from tqdm import tqdm
from time import time

class MAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, n_query, params=None, approx = False):
        super(MAML, self).__init__(model_func, n_way, n_support, change_way = False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.maml_adapt_classifier = params.maml_adapt_classifier

        self.n_task     = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx #first order approx.        

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def set_forward(self,x, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature' 
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data
        
        if self.maml_adapt_classifier:
            fast_parameters = list(self.classifier.parameters())
            for weight in self.classifier.parameters():
                weight.fast = None
        else:
            fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
            for weight in self.parameters():
                weight.fast = None

        self.zero_grad()

        for task_step in (list(range(self.task_update_num))):
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn( scores, y_a_i) 
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
            if self.approx:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            parameters = self.classifier.parameters() if self.maml_adapt_classifier else self.parameters()
            for k, weight in enumerate(parameters):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature = False)
        query_data_labels = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).cuda()
        loss = self.loss_fn(scores, query_data_labels)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = query_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(query_data_labels)) * 100

        return loss, task_accuracy

    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        acc_all = []
        optimizer.zero_grad()

        #train
        for i, (x,_) in enumerate(train_loader):        
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"

            loss, task_accuracy = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()#.data[0]
            loss_all.append(loss)
            acc_all.append(task_accuracy)

            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
        
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        
        metrics = {"accuracy/train": acc_mean}
        
        return metrics 

    def test_loop(self, test_loader, return_std = False, return_time: bool = False): #overwrite parrent function
        correct = 0
        count = 0
        acc_all = []
        eval_time = 0
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            s = time()
            correct_this, count_this = self.correct(x)
            t = time()
            eval_time += (t -s)
            acc_all.append(correct_this/ count_this *100 )

        num_tasks = len(acc_all)
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        print("Num tasks", num_tasks)

        ret = [acc_mean]
        if return_std:
            ret.append(acc_std)
        if return_time:
            ret.append(eval_time)
        ret.append({})

        return ret


    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        logits = self.set_forward(x)
        return logits


class HyperNet(nn.Module):
    def __init__(self, hn_hidden_size, n_way, embedding_size, feat_dim, out_neurons, params):
        super(HyperNet, self).__init__()

        self.hn_head_len = params.hn_head_len
        self.hn_activation = params.hn_activation
        
        head = [nn.Linear(embedding_size, hn_hidden_size)]
        if params.hn_relu:
            head.append(nn.ReLU())
    
        if self.hn_head_len > 2:
            for i in range(self.hn_head_len - 2):
                head.append(nn.Linear(hn_hidden_size, hn_hidden_size))
                if params.hn_relu:
                    head.append(nn.ReLU())

        self.head = nn.Sequential(*head)

        tail = [nn.Linear(hn_hidden_size, out_neurons)]
        if self.hn_activation == 'sigmoid':
            tail.append(nn.Sigmoid())

        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        out = self.head(x)
        out = self.tail(out)
        return out

class HyperMAML(MAML):
    def __init__(self, model_func, n_way, n_support, n_query, params=None, approx = False):
        super(HyperMAML, self).__init__(model_func, n_way, n_support, n_query, params=params)
        self.loss_fn = nn.CrossEntropyLoss()

        self.hn_tn_hidden_size = params.hn_tn_hidden_size
        self.hn_tn_depth = params.hn_tn_depth
        self._init_classifier()

        self.enhance_embeddings = params.hn_enhance_embeddings
        
        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx # first order approx.
        
        self.hn_embeddings_strategy = params.hn_embeddings_strategy
        self.hn_hidden_size = params.hn_hidden_size
        self.hn_lambda = params.hn_lambda
        self.hn_save_delta_params = params.hn_save_delta_params
        self.hn_use_class_batch_input = params.hn_use_class_batch_input
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

        # print(self)

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
            if self.hn_use_class_batch_input and name[-4:] == 'bias':
                continue
            
            bias_size = param.shape[0] // self.n_way

            head_in = self.embedding_size
            head_out = (param.numel() // self.n_way) + bias_size if self.hn_use_class_batch_input else param.numel()
            head_modules = []
        
            self.hypernet_heads[name] = HyperNet(self.hn_hidden_size, self.n_way, head_in, self.feat_dim, head_out, params)

    def calculate_embedding_size(self):

        n_classes_in_embedding = 1 if self.hn_use_class_batch_input else self.n_way
        n_support_per_class = 1 if self.hn_embeddings_strategy == 'class_mean' else self.n_support
        single_support_embedding_len = self.feat_dim + self.n_way + 1 if self.enhance_embeddings else self.feat_dim
        self.embedding_size = n_classes_in_embedding * n_support_per_class * single_support_embedding_len


    def apply_embeddings_strategy(self, embeddings):
        if self.hn_embeddings_strategy == 'class_mean':
            new_embeddings = torch.zeros(self.n_way, *embeddings.shape[1:])

            for i in range(self.n_way):
                lower = i * self.n_support
                upper = (i + 1) * self.n_support
                new_embeddings[i] = embeddings[lower:upper, :].mean(dim=0)

            return new_embeddings.cuda()

        return embeddings

    def get_support_data_labels(self):
        return torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda() # labels for support data

    def get_hn_delta_params(self, support_embeddings):
        if self.hm_detach_before_hyper_net:
            support_embeddings = support_embeddings.detach()

        if self.hn_use_class_batch_input:
            delta_params_list = []

            for name, param_net in self.hypernet_heads.items():

                support_embeddings_resh = support_embeddings.reshape(
                    self.n_way, -1
                )

                delta_params = param_net(support_embeddings_resh)
                bias_neurons_num = self.target_net_param_shapes[name][0] // self.n_way

                if self.hn_adaptation_strategy == 'increasing_alpha' and self.alpha < 1:
                    delta_params = delta_params * self.alpha

                weights_delta = delta_params[:, :-bias_neurons_num]
                bias_delta = delta_params[: ,-bias_neurons_num:].flatten()
                delta_params_list.extend([weights_delta, bias_delta])


            return delta_params_list
        else:
            delta_params_list = []

            for name, param_net in self.hypernet_heads.items():
                flattened_embeddings = support_embeddings.flatten()

                delta = param_net(flattened_embeddings)

                if name in self.target_net_param_shapes.keys():
                    delta = delta.reshape(self.target_net_param_shapes[name])

                if self.hn_adaptation_strategy == 'increasing_alpha' and self.alpha < 1:
                    delta = self.alpha * delta

                delta_params_list.append(delta)

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

    def _get_p_value(self):
        if self.epoch < self.hm_maml_warmup_epochs:
            return 1.0
        elif self.hm_maml_warmup_epochs <= self.epoch < self.hm_maml_warmup_epochs + self.hm_maml_warmup_switch_epochs:
            return (self.hm_maml_warmup_switch_epochs + self.hm_maml_warmup_epochs - self.epoch) / (self.hm_maml_warmup_switch_epochs + 1)
        return 0.0

    def _update_network_weights(self, delta_params_list, support_embeddings, support_data_labels):
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
                self.classifier.zero_grad()
                fast_parameters = fast_parameters + clf_fast_parameters
                
                for task_step in range(self.task_update_num):
                    scores = self.classifier(support_embeddings)

                    set_loss = self.loss_fn(scores, support_data_labels) 

                    grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True, allow_unused=True) #build full graph support gradient of gradient

                    if self.approx:
                        grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation

                    if self.hm_maml_update_feature_net:
                        # update weights of feature networ
                        for k, weight in enumerate(self.feature.parameters()):
                            update_value = self.train_lr * p * grad[k]
                            self._update_weight(weight, update_value)

                    classifier_offset = len(fet_fast_parameters) if self.hm_maml_update_feature_net else 0

                    if p == 1:
                        # update weights of classifier network by adding gradient
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = (self.train_lr * grad[classifier_offset + k])
                            self._update_weight(weight, update_value)

                    elif 0.0 < p < 1.0:
                        # update weights of classifier network by adding gradient and output of hypernetwork
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = ((self.train_lr * p * grad[classifier_offset + k]) + ((1 - p) * delta_params_list[k]))
                            self._update_weight(weight, update_value)
            else:
                for k, weight in enumerate(self.classifier.parameters()):
                    update_value = delta_params_list[k]
                    self._update_weight(weight, update_value)
        else:
            for k, weight in enumerate(self.classifier.parameters()):
                update_value = delta_params_list[k]
                self._update_weight(weight, update_value)


    def _get_list_of_delta_params(self, maml_warmup_used, support_embeddings, support_data_labels):
        if not maml_warmup_used:

            if self.enhance_embeddings:
                with torch.no_grad():
                    logits = self.classifier.forward(support_embeddings).detach()
                    logits = F.softmax(logits, dim=1)
            
                labels = support_data_labels.view(support_embeddings.shape[0], -1)
                support_embeddings = torch.cat((support_embeddings, logits, labels), dim=1)

            for weight in self.parameters():
                weight.fast = None
            self.zero_grad()

            support_embeddings = self.apply_embeddings_strategy(support_embeddings)

            delta_params = self.get_hn_delta_params(support_embeddings)

            if self.hn_save_delta_params and len(self.delta_list) == 0: 
                self.delta_list = [{'delta_params': delta_params}]

            return delta_params
        else:
            return [torch.zeros(*i).cuda() for (_, i) in self.target_net_param_shapes.items()]

    def forward(self, x):
        out  = self.feature.forward(x)
        
        if self.hm_detach_feature_net:
            out = out.detach()

        scores  = self.classifier.forward(out)
        return scores

    def set_forward(self, x, is_feature = False, train_stage = False):
        assert is_feature == False, 'MAML do not support fixed feature'

        x = x.cuda()
        x_var = Variable(x)
        support_data = x_var[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support, *x.size()[2:]) # support data
        query_data = x_var[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query,  *x.size()[2:]) # query data
        support_data_labels = self.get_support_data_labels()

        support_embeddings = self.feature(support_data)   

        if self.hm_detach_feature_net:
            support_embeddings = support_embeddings.detach()

        maml_warmup_used = ((not self.single_test) and self.hm_maml_warmup and (self.epoch < self.hm_maml_warmup_epochs))    

        delta_params_list = self._get_list_of_delta_params(maml_warmup_used, support_embeddings, support_data_labels)
        
        self._update_network_weights(delta_params_list, support_embeddings, support_data_labels)

        if self.hm_set_forward_with_adaptation and not train_stage:
            scores = self.forward(support_data)
            return scores, None
        else:
            if self.hm_support_set_loss and train_stage and not maml_warmup_used:
                query_data = torch.cat((support_data, query_data))

            scores = self.forward(query_data)
                
            # sum of delta params for regularization
            if self.hn_lambda != 0:
                total_delta_sum = sum([delta_params.pow(2.0).sum() for delta_params in delta_params_list])

                return scores, total_delta_sum
            else:
                return scores, None

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x):
        scores, total_delta_sum = self.set_forward(x, is_feature = False, train_stage = True)
        query_data_labels = Variable( torch.from_numpy( np.repeat(range(self.n_way), self.n_query))).cuda()

        if self.hm_support_set_loss:
            support_data_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
            query_data_labels = torch.cat((support_data_labels, query_data_labels))

        loss = self.loss_fn(scores, query_data_labels)

        if self.hn_lambda != 0:
            loss = loss + self.hn_lambda * total_delta_sum

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = query_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(query_data_labels)) * 100

        return loss, task_accuracy

    def set_forward_loss_with_adaptation(self, x):
        scores, _ = self.set_forward(x, is_feature = False, train_stage = False)
        support_data_labels = Variable( torch.from_numpy( np.repeat(range(self.n_way), self.n_support))).cuda()

        loss = self.loss_fn(scores, support_data_labels)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = support_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(support_data_labels)) * 100

        return loss, task_accuracy

    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        acc_all = []
        optimizer.zero_grad()

        self.delta_list = []
        
        #train
        for i, (x,_) in enumerate(train_loader):        
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"

            loss, task_accuracy = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()#.data[0]
            loss_all.append(loss)
            acc_all.append(task_accuracy)

            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(self.epoch, self.stop_epoch, i, len(train_loader), avg_loss/float(i+1)))

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        
        metrics = {"accuracy/train": acc_mean}

        if self.hn_adaptation_strategy == 'increasing_alpha':
            metrics['alpha'] =  self.alpha

        if self.hn_save_delta_params and len(self.delta_list) > 0:
            delta_params = {"epoch": self.epoch, "delta_list": self.delta_list}
            metrics['delta_params'] = delta_params

        if self.alpha < 1:
            self.alpha += self.hn_alpha_step

        return metrics                      

    def test_loop(self, test_loader, return_std = False, return_time: bool = False): #overwrite parrent function

        acc_all = []
        self.delta_list = []
        acc_at = defaultdict(list)

        iter_num = len(test_loader) 
        
        eval_time = 0
        
        if self.hm_set_forward_with_adaptation:
            for i, (x,_) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                assert self.n_way  ==  x.size(0), "MAML do not support way change"
                s = time()
                acc_task, acc_at_metrics = self.set_forward_with_adaptation(x)
                t = time()
                for (k, v) in acc_at_metrics.items():
                    acc_at[k].append(v)
                acc_all.append(acc_task)
                eval_time += (t -s)

        else:
            for i, (x,_) in enumerate(test_loader):
                print(x.shape)
                self.n_query = x.size(1) - self.n_support
                assert self.n_way  ==  x.size(0), f"MAML do not support way change, {self.n_way=}, {x.size(0)=}"
                s = time()
                correct_this, count_this = self.correct(x)
                t = time()
                acc_all.append(correct_this/ count_this *100 )
                eval_time += (t -s)

        
        metrics = {
            k: np.mean(v) if len(v) > 0 else 0
            for (k,v) in acc_at.items()
        }

        num_tasks = len(acc_all)
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
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
        for param1, param2 in zip(self.parameters(), self_copy.parameters()):
            if hasattr(param1, 'fast'):
                if param1.fast is not None:
                    param2.fast = param1.fast.clone()
                else:
                    param2.fast = None

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
        y_query = np.repeat(range( self.n_way ), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)