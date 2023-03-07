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

def train_fs_params(params):
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
    return dict(n_way=params.train_n_way, n_support=params.n_shot, n_query=n_query)

def create_model_instance(params):
    return HyperShot(model_dict[params.model], params=params, **train_fs_params(params)).cuda()

def get_image_size(params):
    image_size = 224
    if params.dataset in ['omniglot', 'cross_char']:
        image_size = 28
    else:
        image_size = 84
    return image_size

def load_dataset(params):
    file = configs.data_dir['omniglot'] + 'noLatin.json'
    if params.dataset == 'cross':
        file = configs.data_dir['miniImagenet'] + 'all.json'
    elif params.dataset == 'cross_char':
        file = configs.data_dir['omniglot'] + 'noLatin.json'
    else:
        file = configs.data_dir[params.dataset] + 'base.json'

    image_size = get_image_size(params)

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
    params.checkpoint_dir = os.environ.get('BASEPATH')
    neptune_run = setup_neptune(params)

    model_path = os.environ.get('MODELPATH')

    # Load model
    model = create_model_instance(params)
    tmp = torch.load(model_path)
    model.load_state_dict(tmp['state'])

    dataset = load_dataset(params)

    def take_next():
        return next(dataset, (None, None))

    def cond(x, y):
        return (x is not None) and (y is not None)

    X = torch.Tensor()
    Y = torch.Tensor()
    x, y = take_next()
    while cond(x, y):
        Y = torch.cat((Y, y), 0)
        X = torch.cat((X, x), 0)
        x, y = take_next()
        while cond(x, y) and (len(reduce(np.intersect1d, (*Y, y))) > 0): 
            x, y = take_next()

    #sorry for ugly calculations, just making it work in a hurry
    ims = get_image_size(params) 
    bb = model.n_way*(model.n_support + model.n_query)
    bs = bb*ims*ims
    bn = int(torch.numel(X)/(bs*(X.size()[2])))
    B = torch.reshape(X, (bn, model.n_way, model.n_support + model.n_query, *X.size()[2:]))

    S = torch.Tensor().cuda()
    Q = torch.Tensor().cuda()
    for b in B:
        s, q = model.parse_feature(b, is_feature=False)
        s = torch.reshape(s, (1, *s.size()))
        q = torch.reshape(q, (1, *q.size()))
        S = torch.cat((S, s), 0)
        Q = torch.cat((Q, q), 0)

    print(S.shape)
    print(Q.shape)

    model.n_query = X[0].size(1) - model.n_support #found that n_query gets changed
    model.eval()

    i = 0
    for s in S:
        q = Q[i]
        q = q.reshape(-1, q.shape[-1])
        print(s.shape)
        print(q.shape)
        classifier, _ = model.generate_target_net(s)
        rel = model.build_relations_features(support_feature=s, feature_to_classify=q)
        r = [] 
        for _ in range(N):
            r.append(torch.stack(classifier(rel).clone().data.cpu().numpy()).mean(dim=0))
        upload_hist(neptune_run, r, i)
        i += 1

if __name__ == '__main__':
    experiment(30)