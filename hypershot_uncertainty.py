import json
import os
from functools import reduce
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

import configs
from data.datamgr import SetDataManager
from io_utils import model_dict, parse_args
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

# def upload_hist(neptune_run, n, arr, i):
#     for j in range(n):
#         fig = plt.figure()
#         plt.hist(arr[j], edgecolor="black", range=[0, 1], bins=25)
#         mu = np.mean(arr[j])
#         std = np.std(arr[j])
#         plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
#         neptune_run[f"Histogram C: {j}, I: {i}"].upload(File.as_image(fig))
#         plt.close(fig)


def find_targets_with_non_empty_difference(QY1, QY2):
    QY1 = set(QY1.flatten().tolist())
    QY2 = set(QY2.flatten().tolist())

    diff = QY2.difference(QY1)

    if len(diff) == 0:
        return None

    return next(iter(diff))

def experiment(N):
    params = parse_args('train') # We need to parse the same parameters as during training
    print(f"Setting checkpoint_dir to {os.environ.get('BASEPATH')}")
    params.checkpoint_dir = os.environ.get('BASEPATH')

    print(f"Loading model from {os.environ.get('MODELPATH')}")
    model_path = os.environ.get('MODELPATH')

    # Load model
    model = create_model_instance(params)
    tmp = torch.load(model_path)
    model.load_state_dict(tmp['state'])

    dataset = load_dataset(params)

    def take_next():
        return next(dataset, (None, None))

    def isAnyNone(x, y):
        return (x is None) or (y is None)

    X = torch.Tensor()
    Y = torch.Tensor()
    x, y = take_next()
    while not isAnyNone(x, y):
        Y = torch.cat((Y, y), 0)
        X = torch.cat((X, x), 0)
        x, y = take_next()

    ims = get_image_size(params) 
    bb = model.n_way*(model.n_support + model.n_query)
    bs = bb*ims*ims
    bn = int(torch.numel(X)/(bs*(X.size()[2])))
    B = torch.reshape(X, (bn, model.n_way, model.n_support + model.n_query, *X.size()[2:]))
    Y = torch.reshape(Y, (bn, model.n_way, model.n_support + model.n_query))

    # Here is our main support, query pair with targets (classifier will be generated from S1)
    S1 = torch.Tensor().cuda()
    Q1 = torch.Tensor().cuda()
    SY1 = torch.Tensor().cuda()
    QY1 = torch.Tensor().cuda()

    # Here will be support, query pair such that set difference of QY2 \ QY1 is non empty 
    # (it means there is(in QY2) a class such that it is out of distribution)
    S2 = torch.Tensor().cuda()
    Q2 = torch.Tensor().cuda()
    SY2 = torch.Tensor().cuda()
    QY2 = torch.Tensor().cuda()

    zippedDataset = [(b,y) for b,y in zip(B,Y)]

    b, y = zippedDataset[0]
    s, q = model.parse_feature(b, is_feature=False)
    sy = y[:, :model.n_support].cuda()
    qy = y[:, model.n_support:].cuda()
    s = torch.reshape(s, (1, *s.size()))
    q = torch.reshape(q, (1, *q.size()))
    S1 = torch.cat((S1, s), 0)
    Q1 = torch.cat((Q1, q), 0)
    sy = torch.reshape(sy, (1, *sy.size()))
    qy = torch.reshape(qy, (1, *qy.size()))
    SY1 = torch.cat((SY1, sy), 0)
    QY1 = torch.cat((QY1, qy), 0)

    # Now we need to find the other pair that meets earlier mentioned condition
    # We will simply check how certain class that meets this requirement behaves when we pass it through the classifier
    # We expect the classifier to be uncertain about proper target

    desired_class = None

    for b, y in zippedDataset:
        S2 = torch.Tensor().cuda()
        Q2 = torch.Tensor().cuda()
        SY2 = torch.Tensor().cuda()
        QY2 = torch.Tensor().cuda()
        s, q = model.parse_feature(b, is_feature=False)
        sy = y[:, :model.n_support].cuda()
        qy = y[:, model.n_support:].cuda()
        s = torch.reshape(s, (1, *s.size()))
        q = torch.reshape(q, (1, *q.size()))
        S2 = torch.cat((S2, s), 0)
        Q2 = torch.cat((Q2, q), 0)
        sy = torch.reshape(sy, (1, *sy.size()))
        qy = torch.reshape(qy, (1, *qy.size()))
        SY2 = torch.cat((SY2, sy), 0)
        QY2 = torch.cat((QY2, qy), 0)

        desired_class = find_targets_with_non_empty_difference(QY1, QY2)

        if desired_class:
            break
        else:
            continue

    print(f"desired_class {desired_class}")
    print(QY1.shape)
    print(QY2.shape)
    print("======")
    print(QY1)
    print(QY2)
    print("======")

    # Now we need to get the exact index of this class
    QY2_index = (QY2 == desired_class).nonzero(as_tuple=False)[0] # of course there might be more than one element of this class
    print("QY2 index: {QY2_index}")

    model.n_query = X[0].size(1) - model.n_support #found that n_query gets changed
    model.eval()

    s1 = next(iter(S1))
    q1 = next(iter(q1))

    s1y = next(iter(SY1))
    q1y = next(iter(QY1))

    s2 = next(iter(S2))
    q2 = next(iter(S1))

    s2y = next(iter(SY2))
    q2y = next(iter(QY2))


    # Here we prepare q1 and classifier generated with s1

    q1 = q1.reshape(-1, q.shape[-1])
    classifier, _ = model.generate_target_net(s1)
    rel = model.build_relations_features(support_feature=s1, feature_to_classify=q1)
    r = [[] for _ in range(model.n_way)]
    for _ in range(N):
            print('---')
            o = classifier(rel)
            print(o.shape)
            sample = torch.nn.functional.softmax(classifier(rel), dim=1)[0].clone().data.cpu().numpy()
            # for j in range(model.n_way):
            #     r[j].append(sample[j])

    # Next step is to do the same but instead of q1 use s1

    # Final step is to test how it works on samples that are out of distribution
    # We simply use q2 instead of q1 and check how probabilities changes for element with QY2_index value (probably one redundant dimension, because it is a whole batch)

if __name__ == '__main__':
    experiment(1)