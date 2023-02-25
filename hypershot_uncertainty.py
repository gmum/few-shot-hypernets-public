import os
import torch
from io_utils import setup_neptune, model_dict, parse_args
from methods.hypernets.hypernet_kernel import HyperShot
import json
 
def create_model_instance(params):
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot, n_query=n_query)
    return HyperShot(model_dict[params.model], params=params, **train_few_shot_params).cuda()

def experiment():
    params = parse_args('train') # We need to parse the same parameters as during training
    model_path = os.environ.get('MODELPATH')


    # Load model
    model = create_model_instance(params)
    tmp = torch.load(model_path)
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
    experiment()