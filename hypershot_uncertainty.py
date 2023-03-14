import json
import os
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    neptune_run = setup_neptune(params)

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

    zippedDataset = [(b,y) for b,y in zip(B,Y)]

    # We get first task (s1,q1) we need only one! and then we will focus on probabilities for only one image!
    b, y = zippedDataset[0]
    s1, q1 = model.parse_feature(b, is_feature=False)
    sy1 = y[:, :model.n_support].cuda()
    qy1 = y[:, model.n_support:].cuda()
    # s1 = torch.reshape(s1, (1, *s1.size()))
    # q1 = torch.reshape(q1, (1, *q1.size()))
    # sy1 = torch.reshape(sy1, (1, *sy1.size()))
    # qy1 = torch.reshape(qy1, (1, *qy1.size()))

    # Now we need to find the other pair that has class such that this class cannot be found in s1

    desired_class = None

    sy2 = torch.Tensor()
    qy2 = torch.Tensor()
    for b, y in zippedDataset:
        s2, q2 = model.parse_feature(b, is_feature=False)
        sy2 = y[:, :model.n_support].cuda()
        qy2 = y[:, model.n_support:].cuda()
        # s2 = torch.reshape(s2, (1, *s2.size()))
        # q2 = torch.reshape(q2, (1, *q2.size()))
        # sy2 = torch.reshape(sy2, (1, *sy2.size()))
        # qy2 = torch.reshape(qy2, (1, *qy2.size()))

        desired_class = find_targets_with_non_empty_difference(sy1, sy2)

        if desired_class:
            break
        else:
            continue

    print(f"desired_class {desired_class}")
    print(sy1.shape)
    print(sy2.shape)
    print("======")
    print(qy1.shape)
    print(qy2.shape)
    print("======")

    #NOTE!! WE NEED TO RESHAPE qy{1,2} to [80] sy{1,2} to [5] and since this will be the output of the classifier for each class
    # and we need to track index of desired_element in classifier output

    sy1 = sy1.flatten()
    sy2 = sy2.flatten()
    qy1 = qy1.flatten()
    qy2 = qy2.flatten()

    # THEN:
    # we need to get the exact index of this class (after reshape!)
    qy2_index = (qy2 == desired_class).nonzero(as_tuple=False)[0] # of course there might be more than one element of this class
    print(f"QY2 index: {qy2_index}")

    # for those images from distribution we just pick first element
    qy1_index = torch.tensor([0], device='cuda:0')
    sy1_index = torch.tensor([0], device='cuda:0')

    model.n_query = X[0].size(1) - model.n_support #found that n_query gets changed
    model.eval()

    # Here we prepare q1 and classifier generated with s1

    q1p = torch.clone(q1)

    # S1 Q1
    R1 = [ [] for _ in range(model.n_way) ]
    q1 = q1.reshape(-1, q1.shape[-1])
    classifier, _ = model.generate_target_net(s1)
    rel = model.build_relations_features(support_feature=s1, feature_to_classify=q1)
    for _ in range(N):
        o = classifier(rel)[0].flatten()
        sample = torch.nn.functional.softmax(o).clone().data.cpu().numpy()
        for i in range(model.n_way):
            R1[i].append(sample[i])


    # in this loop we do a forward pass (above)
    # we get tensor [80,5] 80 is number of images, and 5 is number of classes
    # in other words each element has 5 class probabilities
    # now we focus on measuring probabilities of element with qy1_index (since forward pass was for q1) Probably something like: sample[0,:]
    # gather data from N sampling stages and find its expected value and standard deviation

    # THEN:

    # S1, S1

    R2 = [ [] for _ in range(model.n_way) ]
    q1p[0] = s1[0]
    q1p = q1p.reshape(-1, q1p.shape[-1])
    classifier, _ = model.generate_target_net(s1)
    rel = model.build_relations_features(support_feature=s1, feature_to_classify=q1p)
    for _ in range(N):
        o = classifier(rel)[0].flatten()
        sample = torch.nn.functional.softmax(o).clone().data.cpu().numpy()
        for i in range(model.n_way):
            R2[i].append(sample[i])


    # do a forward pass for s1 tensor (buld_relation_features for support_feature=s1, feature_to_classify=s1)
    # if it will result in wrong dimension there is a workaround
    # in tensor q1 we can swap first image with first image from s1 (it will be again sample[0, :] to get probability for each class) (PROBABLY THE BEST SOLUTION SO PLZ GO FOR IT)
    # (of course most of the images in tensor still will be from this query set but we just need to focus on probabilities of this one image as previously for q1)

    # S1, Q2

    R3 = [ [] for _ in range(model.n_way) ]
    q2 = q2.reshape(-1, q2.shape[-1])
    classifier, _ = model.generate_target_net(s1)
    rel = model.build_relations_features(support_feature=s1, feature_to_classify=q2)
    for _ in range(N):
        o = classifier(rel)[qy2_index].flatten()
        print(o.shape)
        sample = torch.nn.functional.softmax(o).clone().data.cpu().numpy()
        print(sample.shape)
        for i in range(model.n_way):
            R3[i].append(sample[i])

    # finally we want to pass q2 to build_relational_features as feature_to_classify=q2
    # and focus on probabilities for qy2_index (those are probabilities of a class that does not exist in support set s1) HERE IS A CHANGE sample[qy2_index, :]

    df = pd.DataFrame(columns=['Class', 'Type', 'value'])
    for i in range(model.n_way):
        df1 = pd.DataFrame(R1[i], columns=['value'])
        df1['Class'] = i
        df1['Type'] = "Element from query set"

        df2 = pd.DataFrame(R2[i], columns=['value'])
        df2['Class'] = i
        df2['Type'] = "Element from support set"

        df3 = pd.DataFrame(R3[i], columns=['value'])
        df3['Class'] = i
        df3['Type'] = "Element ou of distribution"
        df = df.append(pd.concat([df1, df2, df3]))

    df.head()
    fig = plt.figure()
    sns.boxplot(data=mdf, x='Class', y='value', hue='Type')
    neptune_run[f"Plot"].upload(File.as_image(fig))
    plt.close(fig)

if __name__ == '__main__':
    experiment(50)