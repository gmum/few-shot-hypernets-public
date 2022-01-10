from pathlib import Path
from typing import Optional

import neptune.new as neptune
import numpy as np
import os
import glob
import argparse
import time

from neptune.new import Run

import backbone
import configs
import hn_args
from methods.hypernet_poc import hn_poc_types

model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101)

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script), formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     )
    parser.add_argument('--seed' , default=0, type=int,  help='Seed for Numpy and pyTorch. Default: 0 (None)')
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='baseline',  choices=['DKT', 'protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx', "protonet"] + list(hn_poc_types.keys()),
                        help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}/hn_poc') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument("--checkpoint_suffix", type=str,default="", help="Suffix for custom experiment differentiation" )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optim", type=str, choices=["adam", "sgd"], help="Optimizer", default="adam")
    parser.add_argument('--hn_dropout', type=float, default=0.01)
    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--repeat', default=5, type=int, help ='Repeat the test N times with different seeds and take the mean. The seeds range is [seed, seed+repeat]')
    else:
       raise ValueError('Unknown script')

    parser = hn_args.add_hn_args_to_parser(parser)
    return parser.parse_args()

def parse_args_regression(script):
        parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
        parser.add_argument('--seed' , default=0, type=int,  help='Seed for Numpy and pyTorch. Default: 0 (None)')
        parser.add_argument('--model'       , default='Conv3',   help='model: Conv{3} / MLP{2}')
        parser.add_argument('--method'      , default='DKT',   help='DKT / transfer')
        parser.add_argument('--dataset'     , default='QMUL',    help='QMUL / sines')
        parser.add_argument('--spectral', action='store_true', help='Use a spectral covariance kernel function')

        if script == 'train_regression':
            parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
            parser.add_argument('--stop_epoch'  , default=100, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
            parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        elif script == 'test_regression':
            parser.add_argument('--n_support', default=5, type=int, help='Number of points on trajectory to be given as support points')
            parser.add_argument('--n_test_epochs', default=10, type=int, help='How many test people?')
        return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

def setup_neptune(params) -> Run:
    try:
        run_name = Path(params.checkpoint_dir).relative_to(Path(configs.save_dir) / "checkpoints").name
        run_file = Path(params.checkpoint_dir) / "NEPTUNE_RUN.txt"

        run_id = None
        if params.resume and run_file.exists():
            with run_file.open("r") as f:
                run_id = f.read()
                print("Resuming neptune run", run_id)

        run = neptune.init(
            "konrad-karanowski/GMUM-FSHN", 
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOWI3NGE4Ni0wN2JjLTRhZDUtODgzYy1iYzdhOGIyNDQwZGEifQ==",
            name=run_name,
            source_files="**/*.py",
            tags=[params.checkpoint_suffix] if params.checkpoint_suffix != "" else [],
            run=run_id
        )
        with run_file.open("w") as f:
            f.write(run._short_id)
            print("Starting neptune run", run._short_id)
        run["params"] = vars(params)
        return run

    except Exception as e:
        print("Cannot initialize neptune because of", e)
        pass