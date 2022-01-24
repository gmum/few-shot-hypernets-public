from pathlib import Path

import torch
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import time
from typing import Type

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.hypernet_poc import HyperNetPOC, hn_poc_types
from methods.protonet import ProtoNet
from methods.DKT import DKT
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, get_resume_file, parse_args, get_best_file , get_assigned_file

def _set_seed(seed, verbose=True):
    if(seed!=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        if(verbose): print("[INFO] Setting SEED: " + str(seed))   
    else:
        if(verbose): print("[INFO] Setting SEED: None")

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100 
    return acc


def single_test(params):
    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        # params.model = 'Conv4S'

    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    elif params.method == 'DKT':
        model           = DKT(model_dict[params.model], **few_shot_params)
    elif params.method == 'matchingnet':
        model           = MatchingNet( model_dict[params.model], **few_shot_params )
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6': 
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S': 
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
        if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
            model.n_task     = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    elif params.method in list(hn_poc_types.keys()):
        few_shot_params['n_query'] = 15
        hn_type: Type[HyperNetPOC] = hn_poc_types[params.method]
        model = hn_type(model_dict[params.model], params=params, **few_shot_params)
        # model = HyperNetPOC(model_dict[params.model], **few_shot_params)

    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    if params.checkpoint_suffix != "":
        checkpoint_dir = checkpoint_dir + "_" + params.checkpoint_suffix

    #modelfile   = get_resume_file(checkpoint_dir)

    if not params.method in ['baseline', 'baseline++'] : 
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        else:
            modelfile   = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])
        else:
            print("[WARNING] Cannot find 'best_file.tar' in: " + str(checkpoint_dir))

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    if params.method in ['maml', 'maml_approx', 'DKT'] + list(hn_poc_types.keys()): #maml do not support testing with feature
        if 'Conv' in params.model:
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84 
        else:
            image_size = 224

        datamgr         = SetDataManager(image_size, n_eposide = iter_num, **few_shot_params)
        
        if params.dataset == 'cross':
            train_file = configs.data_dir['miniImagenet'] + 'all.json'
            if split == 'base':
                loadfile = configs.data_dir['miniImagenet'] + 'all.json'
            else:
                loadfile   = configs.data_dir['CUB'] + split +'.json'
        elif params.dataset == 'cross_char':
            train_file = configs.data_dir['omniglot'] + 'noLatin.json'
            if split == 'base':
                loadfile = configs.data_dir['omniglot'] + 'noLatin.json' 
            else:
                loadfile  = configs.data_dir['emnist'] + split +'.json' 
        else:
            train_file = configs.data_dir[params.dataset] + 'base.json'
            loadfile    = configs.data_dir[params.dataset] + split + '.json'

        train_loader = datamgr.get_data_loader(train_file, aug=False)
        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        # print(model.test_loop( novel_loader, return_std = True))

        train_save_path = test_save_path = None
        if isinstance(model, (HyperNetPOC, MAML)):
            base_save_path = Path(checkpoint_dir) / "test_save_weights"

            train_save_path = base_save_path / "train"
            test_save_path = base_save_path / "test"

            train_save_path.mkdir(exist_ok=True, parents=True)
            test_save_path.mkdir(exist_ok=True, parents=True)

        print("#### TRAIN ####", train_file)
        model.test_loop(train_loader, return_std=True, save_path=train_save_path)

        print("#### TEST ####", loadfile)
        acc_mean, acc_std, _ = model.test_loop( novel_loader, return_std = True, save_path=test_save_path)

    else:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    with open('./record/results.txt' , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-adapted' if params.adaptation else ''
        if params.method in ['baseline', 'baseline++'] :
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
        else:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )
    return acc_mean

def perform_test(params):
    seed = params.seed
    repeat = params.repeat
    # repeat the test N times changing the seed in range [seed, seed+repeat]
    accuracy_list = list()
    for i in range(seed, seed + repeat):
        if (seed != 0):
            _set_seed(i)
        else:
            _set_seed(0)
        accuracy_list.append(single_test(params))

    mean_acc = np.mean(accuracy_list)
    std_acc = np.std(accuracy_list)
    print("-----------------------------")
    print(
        'Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' % (repeat, mean_acc, std_acc ))
    print("-----------------------------")
    return {
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "n_seeds": repeat
    }

def main():        
    params = parse_args('test')
    perform_test(params)


if __name__ == '__main__':
    main()
