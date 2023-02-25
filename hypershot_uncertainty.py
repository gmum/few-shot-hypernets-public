import os
import torch
from io_utils import setup_neptune, model_dict, parse_args
from methods.hypernets.hypernet_kernel import HyperShot
import json


# NOTE: This uncertainty experiment was created on the master branch.
# But still we have to use it on other branches with different implementations of model architectures (and different set of parameters).
# If it is necessary to use this on other branches but differences in code does not allow to merge master you can do the following:
# Checkout those files from master:
# * `hypershot_uncertainty.py`
# * `hypershot_uncertainty.sh`
# * `parse_args.py`
# Then in io_utils.py create a function `create_parser` that simply creates parser and returns it (see how it works on master branch).
# Parsers of different tested architectures may differ so results of mergin io_utils.py might be dangerous and time-consuming.
# To run an experiment just copy a file `experiment_template.sh` and customize it.
 
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