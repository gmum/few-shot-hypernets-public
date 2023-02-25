import sys
import os
import torch
from io_utils import setup_neptune, model_dict
from methods.hypernets.hypernet_kernel import HyperShot
import json

def read_args(args_path):
    with open(args_path) as args:
        data = json.load(args)
        return data
 
def create_model_instance(args_path):
    args = read_args(args_path)
    return HyperShot(model_dict[args.model], **args).cuda()


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