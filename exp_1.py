import pickle
import shutil
from pathlib import Path
from functools import reduce
import torch
import torch.optim
import torch.utils.data.sampler
from torch.nn import functional as F
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from methods.hypernets.hypernet_kernel import HyperShot
from neptune.new.types import File
import os
from os import path
import configs
from data.datamgr import SetDataManager

from methods.hypernets.hypermaml import HyperMAML
from io_utils import model_dict, parse_args, get_best_file, setup_neptune
from methods.hypernets.utils import reparameterize

save_numeric_data = True
def plot_mu_sigma(neptune_run, model, i, save_numeric_data=save_numeric_data):
    # get flattened mu and sigma
    param_dict = model.get_mu_and_sigma()
    # plotting to neptune
    for name, value in param_dict.items():
        fig = plt.figure()
        plt.plot(value, 's')
        neptune_run[f"{name} / plot"].upload(File.as_image(fig))
        plt.close(fig)
        fig = plt.figure()
        plt.hist(value, edgecolor="black")
        neptune_run[f"{name} / histogram"].upload(File.as_image(fig))
        plt.close(fig)
        if save_numeric_data:
            neptune_run[f"{name} / data"].upload(File.as_pickle(value))
            
# plot uncertainty in classification
def plot_histograms(neptune_run, s1, s2, q1, q2, save_numeric_data=save_numeric_data):

    # seen support
    for i, scores in s1.items():
        if save_numeric_data:
            path = f'exp_1_data/Seen/Support/{i}'
            os.mkdir(path)
        scores = np.transpose(np.array(scores))
        for k, score in enumerate(scores):
            score = np.array(score)
            # print(f"score shape {score.shape}")
            fig = plt.figure()
            plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
            mu = np.mean(score)
            std = np.std(score)
            plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
            neptune_run[f"Seen / Support / {i} / Class {k} histogram"].upload(File.as_image(fig))
            plt.close(fig)
            # save on neptune
            if save_numeric_data:
                neptune_run[f"Seen / Support / {i} / Class {k} data"].upload(File.as_pickle(score))
                filepath = path + f'/Class_{k}_data'
                with open(filepath, 'wb') as f:
                    pickle.dump(score,f)

    # seen query
    for i, scores in q1.items():
        if save_numeric_data:
            path = f'exp_1_data/Seen/Query/{i}'
            os.mkdir(path)
        scores = np.transpose(np.array(scores))
        for k, score in enumerate(scores):
            score = np.array(score)
            fig = plt.figure()
            plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
            mu = np.mean(score)
            std = np.std(score)
            plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
            neptune_run[f"Seen / Query / {i} / Class {k} histogram"].upload(File.as_image(fig))
            plt.close(fig)
            # save on neptune
            if save_numeric_data:
                neptune_run[f"Seen / Query / {i} / Class {k} data"].upload(File.as_pickle(score))
                filepath = path + f'/Class_{k}_data'
                with open(filepath, 'wb') as f:
                    pickle.dump(score,f)

    # unseen support
    for i, scores in s2.items():
        if save_numeric_data:
            path = f'exp_1_data/Unseen/Support/{i}'
            os.mkdir(path)
        scores = np.transpose(np.array(scores))
        for k, score in enumerate(scores):
            score = np.array(score)
            fig = plt.figure()
            plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
            mu = np.mean(score)
            std = np.std(score)
            plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
            neptune_run[f"Unseen / Support / {i} / Class {k} histogram"].upload(File.as_image(fig))
            plt.close(fig)
            if save_numeric_data:
                # save on neptune
                neptune_run[f"Unseen / Support / {i} / Class {k} data"].upload(File.as_pickle(score))
                # save file locally
                filepath = path + f'/Class_{k}_data'
                with open(filepath, 'wb') as f:
                    pickle.dump(score,f)
    # unseen query
    for i, scores in q2.items():
        if save_numeric_data:
            path = f'exp_1_data/Unseen/Query/{i}'
            os.mkdir(path)
        scores = np.transpose(np.array(scores))
        for k, score in enumerate(scores):
            score = np.array(score)
            fig = plt.figure()
            plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
            mu = np.mean(score)
            std = np.std(score)
            plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
            neptune_run[f"Unseen / Query / {i} / Class {k} histogram"].upload(File.as_image(fig))
            plt.close(fig)
            if save_numeric_data:
                # save on neptune
                neptune_run[f"Unseen / Query / {i} / Class {k} data"].upload(File.as_pickle(score))
                # save file locally
                filepath = path + f'/Class_{k}_data'
                with open(filepath, 'wb') as f:
                    pickle.dump(score,f)


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

def initLocalDirectories():
    if path.isdir('exp_1_data'):
        shutil.rmtree('exp_1_data')
    os.mkdir('exp_1_data')
    os.mkdir('exp_1_data/Seen')
    os.mkdir('exp_1_data/Seen/Support')
    os.mkdir('exp_1_data/Seen/Query')
    os.mkdir('exp_1_data/Unseen')
    os.mkdir('exp_1_data/Unseen/Support')
    os.mkdir('exp_1_data/Unseen/Query')

def experiment(params_experiment):
    if save_numeric_data:
        initLocalDirectories()
    num_samples = params_experiment.num_samples
    if params_experiment.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params_experiment.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params_experiment.dataset] + 'base.json'
        val_file = configs.data_dir[params_experiment.dataset] + 'val.json'

    if 'Conv' in params_experiment.model:
        if params_experiment.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    n_query = max(1, int(16 * params_experiment.test_n_way / params_experiment.train_n_way))
    # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

    n_way = params_experiment.n_way
    train_few_shot_params = dict(n_way=n_way, n_support=params_experiment.n_shot, n_query=n_query)
    # base_datamgr = SetDataManager(image_size, **train_few_shot_params)  # n_eposide = 100
    # base_loader = base_datamgr.get_data_loader(base_file, aug=params_experiment.train_aug)

    test_few_shot_params = dict(n_way=n_way, n_support=params_experiment.n_shot, n_query=n_query)
    val_datamgr = SetDataManager(image_size, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    if params_experiment.dataset in ['omniglot', 'cross_char']:
        assert params_experiment.model == 'Conv4' and not params_experiment.train_aug, 'omniglot only support Conv4 without augmentation'

    if params_experiment.method == 'hyper_maml':
        model = HyperMAML(model_dict[params_experiment.model], params=params_experiment,
                          approx=(params_experiment.method == 'maml_approx'),
                          **train_few_shot_params)
        if params_experiment.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            model.n_task = 32
            model.train_lr = 0.1
    elif params_experiment.method == 'hyper_shot':
        model = HyperShot(model_dict[params_experiment.model], params=params_experiment, **train_few_shot_params)
    else:
        raise ValueError('Experiment for hyper_maml only')

    model = model.cuda()

    params_experiment.checkpoint_dir = getCheckpointDir(params_experiment, configs)

    modelfile = get_best_file(params_experiment.checkpoint_dir)  # load best from given model
    print("Using model file", modelfile)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
    else:
        print("[WARNING] Cannot find 'best_file.tar' in: " + str(params_experiment.checkpoint_dir))

    neptune_run = setup_neptune(params_experiment)
    # primary batches for adaptation
    features = []
    labels = []

    for _ in range(params_experiment.num_batches_seen):
        features1, labels1 = next(iter(val_loader))
        if labels:
            while reduce(np.intersect1d, (*labels, labels1)).size > 0:
                features1, labels1 = next(iter(val_loader))
        features.append(features1)
        labels.append(labels1)

    model.n_query = features[0].size(1) - model.n_support
    support_datas1 = []
    query_datas1 = []
    support_datas2 = []
    query_datas2 = []
    model.train()
    # train on 'seen' data
    for i, features1 in enumerate(features):
        _ = model.set_forward_loss(features1, False)
        plot_mu_sigma(neptune_run, model, i)
        features1 = features1.cuda()
        x_var = torch.autograd.Variable(features1)
        support_data = x_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                             *features1.size()[2:])  # support data
        query_data = x_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                           *features1.size()[2:])  # query data
        support_datas1.append(support_data)
        query_datas1.append(query_data)

    # only draw one set from weights distribution
    model.weight_set_num_train = 1
    model.weight_set_num_test = 1

    features_unseen = []
    # new batches for experiment
    for _ in range(params_experiment.num_batches_unseen):
        features2, labels2 = next(iter(val_loader))
        print('finding val batch')
        # if there are repetitions between batches get another batch
        while reduce(np.intersect1d, (*labels, labels2)).size > 0:
            features2, labels2 = next(iter(val_loader))
        print(labels2)
        labels.append(labels2)
        features_unseen.append(features2)

    model.n_query = features[-1].size(1) - model.n_support
    model.eval()
    for i, features2 in enumerate(features_unseen):
        features2 = features2.cuda()
        x2_var = torch.autograd.Variable(features2)
        support_data2 = x2_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                               *features2.size()[2:])  # support data
        query_data2 = x2_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                             *features2.size()[2:])  # query data
        support_datas2.append(support_data2)
        query_datas2.append(query_data2)

    s1 = {}
    q1 = {}
    s2 = {}
    q2 = {}
    model.weight_set_num_train = 1
    model.weight_set_num_test = 1

    for _ in range(num_samples):
        for weight in model.classifier.parameters():
            weight.fast = [reparameterize(weight.mu, weight.logvar)]
        for i, support_data1 in enumerate(support_datas1):
            if i not in s1:
                s1[i] = []
            s1[i].append(F.softmax(model(support_data1), dim=1)[0].clone().data.cpu().numpy())
        for i, query_data1 in enumerate(query_datas1):
            if i not in q1:
                q1[i] = []
            q1[i].append(F.softmax(model(query_data1), dim=1)[0].clone().data.cpu().numpy())
        for i, support_data2 in enumerate(support_datas2):
            if i not in s2:
                s2[i] = []
            s2[i].append(F.softmax(model(support_data2), dim=1)[0].clone().data.cpu().numpy())
        for i, query_data2 in enumerate(query_datas2):
            if i not in q2:
                q2[i] = []
            q2[i].append(F.softmax(model(query_data2), dim=1)[0].clone().data.cpu().numpy())

    plot_histograms(neptune_run, s1, s2, q1, q2)


def main():
    # params_experiment = parse_args('train')
    params_experiment = parse_args('experiment1')
    experiment(params_experiment=params_experiment)


if __name__ == '__main__':
    main()
