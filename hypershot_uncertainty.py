import sys
import os
import torch
from io_utils import setup_neptune, model_dict
from methods.hypernets.hypernet_kernel import HyperShot
import json

def read_args(args_path):
    with open(args_path) as json_args:
        args_dict = json.load(json_args)
        return args_dict
 
def create_model_instance(args_path):
    args_dict = read_args(args_path)
    n_query = max(1, int(16 * args_dict['test_n_way'] / args_dict['train_n_way']))
    train_few_shot_params = dict(n_way=args_dict['train_n_way'], n_support=args_dict['n_shot'], n_query=n_query)
    return HyperShot(model_dict[args_dict['model']], **train_few_shot_params).cuda()


def experiment(model_path):

    best_model_path = os.path.join(model_path, "best_model.tar")
    args_path = os.path.join(model_path, "args.json")

    model = create_model_instance(args_path)
    tmp = torch.load(best_model_path)

    model.load_state_dict(tmp['state'])
    print(model.S)

    #Parameters: N, M, model, dataset
    #1. Load Model
    #2. Load dataset [(S, Q)]
    #3. For N (S, Q) pairs
    #   - Select ith (S, Q) pair
    #   - Select another one with disjoint support (S', Q')
    #   - Eval model M times on (S, Q') and generate histogram tagged as ith

if __name__ == '__main__':

    model_path = sys.argv[1]
    experiment(model_path)