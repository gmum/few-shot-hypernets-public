import json
import os
from functools import reduce
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from neptune.new.types import File

import configs
from data.datamgr import SetDataManager
from io_utils import model_dict, parse_args, setup_neptune
from methods.hypernets.hypernet_kernel import HyperShot

# NOTE: This uncertainty experiment was created on the master branch.
# But still we have to use it on other branches with different implementations of model architectures (and different set of parameters).
# If it is necessary to use this on other branches but differences in code does not allow to merge master you can do the following:
# Checkout those files from master:
# * `hypershot_uncertainty.py`
# * `hypershot_uncertainty.sh`
# * `parse_args.py`
# Then in io_utils.py create a function `create_parser` that simply creates parser and returns it (see how it works on master branch).
# Parsers of different tested architectures may differ so results of mergin io_utils.py might be dangerous and time-consuming.

# NOTE: To run an experiment just copy a file `experiment_template.sh` and customize it.

def getCheckpointDir(params, configs):
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        configs.save_dir,
        params.dataset,
        params.model,
        params.method
    )

    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    if params.checkpoint_suffix != "":
        checkpoint_dir = checkpoint_dir + "_" + params.checkpoint_suffix

    if params.dataset == "cross":
        if not Path(checkpoint_dir).exists():
            checkpoint_dir = checkpoint_dir.replace("cross", "miniImagenet")

    assert Path(checkpoint_dir).exists(), checkpoint_dir
    return checkpoint_dir

def train_fs_params(params):
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
    return dict(n_way=params.train_n_way, n_support=params.n_shot, n_query=n_query)

def create_model_instance(params):
    return HyperShot(model_dict[params.model], params=params, **train_fs_params(params)).cuda()

def load_dataset(params):
    file = configs.data_dir['omniglot'] + 'noLatin.json'
    if params.dataset == 'cross':
        file = configs.data_dir['miniImagenet'] + 'all.json'
    elif params.dataset == 'cross_char':
        file = configs.data_dir['omniglot'] + 'noLatin.json'
    else:
        file = configs.data_dir[params.dataset] + 'base.json'

    image_size = 224
    if params.dataset in ['omniglot', 'cross_char']:
        image_size = 28
    else:
        image_size = 84

    data_mgr = SetDataManager(image_size, **train_fs_params(params))
    return iter(data_mgr.get_data_loader(file, aug=False))

def upload_hist(neptune_run, arr, i):
    fig = plt.figure()
    plt.hist(arr, edgecolor="black", range=[0, 1], bins=25)
    mu = np.mean(arr)
    std = np.std(arr)
    plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
    neptune_run[f"Histogram{i}"].upload(File.as_image(fig))
    plt.close(fig)

def experiment(N):
    params = parse_args('train') # We need to parse the same parameters as during training
    params.checkpoint_dir = getCheckpointDir(params, configs)
    neptune_run = setup_neptune(params)

    model_path = os.environ.get('MODELPATH')

    # Load model
    model = create_model_instance(params)
    tmp = torch.load(model_path)
    model.load_state_dict(tmp['state'])
    model.eval()

    dataset = load_dataset(params)

    def take_next():
        return next(dataset, (None, None))

    def cond(x, y):
        return (x is not None) and (y is not None)

    X = []
    Y = []
    x, y = take_next()
    while cond(x, y):
        Y.append(x.cpu())
        X.append(y.cpu())
        x, y = take_next()
        while cond(x, y) and (reduce(np.intersect1d, (*Y, y)).size > 0): 
            x, y = take_next()
        
    S, Q = model.parse_feature(torch.as_tensor(np.array(X)), is_feature=False)
    i = 0
    for s, q in zip(enumerate(S), enumerate(Q)):
        classifier, _ = model.generate_target_net(s)
        r = []
        for _ in range(N):
            r.append(torch.nn.functional.softmax(classifier(q), dim=1)[0].clone().data.cpu().numpy())
        upload_hist(neptune_run, r, i)
        i += 1

if __name__ == '__main__':
    experiment(30)