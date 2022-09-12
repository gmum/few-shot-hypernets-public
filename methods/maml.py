# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import torch
from tqdm import tqdm

import backbone
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from methods.meta_template import MetaTemplate
from time import time
from itertools import permutations

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
        self.params = params

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def set_forward(self,x, is_feature = False, return_upd_res: bool = False):
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

        original_parameters = {
            k: p.detach().cpu().numpy()
            for k, p in self.named_parameters()
        }
        original_activations = self.feature.forward(x_a_i)
        original_scores = self.classifier.forward(original_activations)

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

        if return_upd_res:
            updated_parameters = {
                k: p.fast.detach().cpu().numpy()
                for k, p in self.named_parameters()
                if p.fast is not None
            }
            updated_activations = self.feature.forward(x_a_i)
            updated_scores = self.classifier.forward(updated_activations)

            results = {
                "original": {
                    "parameters": original_parameters,
                    "activations": original_activations.detach().cpu().numpy(),
                    "scores": original_scores.detach().cpu().numpy()
                },
                "updated": {
                    "parameters": updated_parameters,
                    "activations": updated_activations.detach().cpu().numpy(),
                    "scores": updated_scores.detach().cpu().numpy()
                }
            }

            return scores, results

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

    def correct(self, x, return_upd_res: bool):
        if return_upd_res:
            scores, update_results = self.set_forward(x, return_upd_res=True)
        else:
            scores = self.set_forward(x, return_upd_res=False)
            update_results = dict(original=None)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query), update_results

    def test_loop(self, test_loader, return_std = False, return_time: bool = False, save_upd_results: bool = False): #overwrite parrent function
        correct = 0
        count = 0
        acc_all = []
        eval_time = 0
        iter_num = len(test_loader)
        tq_loader = tqdm(enumerate(test_loader), total=iter_num)

        records = []

        for i, (x,y) in tq_loader:
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            labels = y.T[0].detach().cpu().numpy()

            perm_records = []
            perms_to_check = [[0,1,2,3,4]]
            if save_upd_results:
                perms_to_check = permutations([0,1,2,3,4])

            for p, perm in enumerate(perms_to_check):
                perm_tensor = torch.Tensor(perm).long()
                x_ = x[perm_tensor]
                y_ = y[perm_tensor]
                original_labels = y_.T[0]

                s = time()

                correct_this, count_this, update_results = self.correct(x_, return_upd_res=save_upd_results)
                t = time()
                eval_time += (t -s)
                acc = correct_this/ count_this *100
                acc_all.append( acc)
                tq_loader.set_description(f"{i=} {p=} {acc=:.2f}")
                if perm == (0,1,2,3,4):
                    original_record = update_results["original"]

                del update_results["original"]

                perm_record = {
                    "original_labels": original_labels.detach().cpu().numpy(),
                    "update_results": update_results,
                    "accuracy": acc,
                    "perm": perm,
                }
                perm_records.append(perm_record)





            if save_upd_results:
                record = {
                    "original": original_record,
                    "perm_records": perm_records,
                    "labels": labels
                }
                from pathlib import Path
                records_path = Path(self.params.checkpoint_dir) / "updates_records"
                # records_path = Path("maml_minim_1_shot_perm")
                # records_path = Path("maml_update_only_cls_minim_1_shot_perm")
                records_path.mkdir(exist_ok=True, parents=True)
                r_path = records_path / f"{i}.pkl"
                with r_path.open("wb") as f:
                    import pickle
                    pickle.dump(record, f)


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


