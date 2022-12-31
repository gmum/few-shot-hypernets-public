import json
import sys
from collections import defaultdict
from typing import Type, List, Union, Dict, Optional
from copy import deepcopy

import numpy as np
import torch
import random
from neptune.new import Run
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import os

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.DKT import DKT
from methods.hypernets.hypernet_poc import HyperNetPOC
from methods.hypernets import hypernet_types
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.hypernets.hypermaml import HyperMAML
from io_utils import model_dict, parse_args, get_resume_file, setup_neptune

import matplotlib.pyplot as plt
from neptune.new.types import File
from pathlib import Path

from save_features import do_save_fts
from test import perform_test

def upload_images(neptune_run, hist_data, tag):
    if hist_data:
        if hist_data["mu_weight"] != []:
            # mu weight 
            fig = plt.figure()
            plt.hist(hist_data["mu_weight"], edgecolor="black", bins=20)
            neptune_run[f"mu_weight @ {tag} / histogram"].upload(File.as_image(fig))
            plt.close(fig)

            fig = plt.figure()
            plt.violinplot(hist_data["mu_weight"])
            neptune_run[f"mu_weight @ {tag} / violinplot"].upload(File.as_image(fig))
            plt.close(fig)

        if hist_data["mu_bias"] != []:
            # mu bias
            fig = plt.figure()
            plt.hist(hist_data["mu_bias"], edgecolor="black", bins=20)
            neptune_run[f"mu_bias @ {tag} / histogram"].upload(File.as_image(fig))
            plt.close(fig)

            fig = plt.figure()
            plt.violinplot(hist_data["mu_bias"])
            neptune_run[f"mu_bias @ {tag} / violinplot"].upload(File.as_image(fig))
            plt.close(fig)

        if hist_data["sigma_weight"] != []:
            # sigma weight
            fig = plt.figure()
            plt.hist(hist_data["sigma_weight"], edgecolor="black", bins=20)
            neptune_run[f"sigma_weight @ {tag} / histogram"].upload(File.as_image(fig))
            plt.close(fig)

            fig = plt.figure()
            plt.violinplot(hist_data["sigma_weight"])
            neptune_run[f"sigma_weight @ {tag} / violinplot"].upload(File.as_image(fig))
            plt.close(fig)

        if hist_data["sigma_bias"] != []:
            # sigma bias
            fig = plt.figure()
            plt.hist(hist_data["sigma_bias"], edgecolor="black", bins=20)
            neptune_run[f"sigma_bias @ {tag} / histogram"].upload(File.as_image(fig))
            plt.close(fig)

            fig = plt.figure()
            plt.violinplot(hist_data["sigma_bias"])
            neptune_run[f"sigma_bias @ {tag} / violinplot"].upload(File.as_image(fig))
            plt.close(fig)

def _set_seed(seed, verbose=True):
    if (seed != 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if (verbose): print("[INFO] Setting SEED: " + str(seed))
    else:
        if (verbose): print("[INFO] Setting SEED: None")


def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, *,
          neptune_run: Optional[Run] = None):
    print("Tot epochs: " + str(stop_epoch))
    if optimization == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif optimization == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    else:
        raise ValueError(f'Unknown optimization {optimization}, please define by yourself')

    max_acc = 0
    max_train_acc = 0
    max_acc_adaptation_dict = {}

    if params.hm_set_forward_with_adaptation:
        max_acc_adaptation_dict = {}
        for i in range(params.hn_val_epochs + 1):
            if i != 0:
                max_acc_adaptation_dict[f"accuracy/val_support_max@-{i}"] = 0
            max_acc_adaptation_dict[f"accuracy/val_max@-{i}"] = 0

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    if (Path(params.checkpoint_dir) / "metrics.json").exists() and params.resume:
        with (Path(params.checkpoint_dir) / "metrics.json").open("r") as f:
            try:
                metrics_per_epoch = defaultdict(list, json.load(f))
                try:
                    max_acc = metrics_per_epoch["accuracy/val_max"][-1]
                    max_train_acc = metrics_per_epoch["accuracy/train_max"][-1]

                    if params.hm_set_forward_with_adaptation:
                        for i in range(params.hn_val_epochs + 1):
                            if i != 0:
                                max_acc_adaptation_dict[f"accuracy/val_support_max@-{i}"] = \
                                metrics_per_epoch[f"accuracy/val_support_max@-{i}"][-1]
                            max_acc_adaptation_dict[f"accuracy/val_max@-{i}"] = \
                            metrics_per_epoch[f"accuracy/val_max@-{i}"][-1]
                except:
                    max_acc = metrics_per_epoch["accuracy_val_max"][-1]
                    max_train_acc = metrics_per_epoch["accuracy_train_max"][-1]
            except:
                metrics_per_epoch = defaultdict(list)

    else:
        metrics_per_epoch = defaultdict(list)

    scheduler = get_scheduler(params, optimizer, stop_epoch)

    print("Starting training")
    print("Params accessed until this point:")
    print("\n\t".join(sorted(params.history)))
    print("Params ignored until this point:")
    print("\n\t".join(params.get_ignored_args()))

    delta_params_list = []

    for epoch in range(start_epoch, stop_epoch):
        if epoch >= params.es_epoch:
            if max_acc < params.es_threshold:
                print("Breaking training at epoch", epoch, "because max accuracy", max_acc, "is lower than threshold",
                      params.es_threshold)
                break
        model.epoch = epoch
        model.start_epoch = start_epoch
        model.stop_epoch = stop_epoch

        model.epoch_state_dict["hn_warmup"] = params.hn_warmup
        model.epoch_state_dict["cur_epoch"] = epoch 
        model.epoch_state_dict["from_epoch"] = params.hn_warmup_start_epoch
        model.epoch_state_dict["to_epoch"] = params.hn_warmup_stop_epoch

        model.train()
        metrics, hist_data = model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return

        if epoch % 100 == 0:
            upload_images(neptune_run, hist_data, epoch)

        scheduler.step()
        model.eval()

        delta_params = metrics.pop('delta_params', None)
        if delta_params is not None:
            delta_params_list.append(delta_params)

        if (epoch % params.eval_freq == 0) or epoch in [
            params.es_epoch - 1,
            stop_epoch - 1
        ]:
            try:
                acc, test_loop_metrics, bnn_dict = model.test_loop(val_loader, epoch=epoch)
            except:
                acc, bnn_dict = model.test_loop(val_loader, epoch=epoch)
                test_loop_metrics = dict()
            print(
                f"Epoch {epoch}/{stop_epoch}  | Max test acc {max_acc:.2f} | Test acc {acc:.2f} | Metrics: {test_loop_metrics}")

            print(bnn_dict.keys())
            if bnn_dict:
                fig = plt.figure()
                plt.hist(bnn_dict[f"mu_weight_test_mean@{epoch}"], edgecolor="black", bins=20)
                neptune_run[f"mu_weight_test_mean@{epoch}/train"].upload(File.as_image(fig))
                plt.close(fig)

                fig = plt.figure()
                plt.hist(bnn_dict[f"mu_bias_test_mean@{epoch}"], edgecolor="black", bins=20)
                neptune_run[f"mu_bias_test_mean@{epoch}/train"].upload(File.as_image(fig))
                plt.close(fig)

                fig = plt.figure()
                plt.hist(bnn_dict[f"sigma_weight_test_mean@{epoch}"], edgecolor="black", bins=20)
                neptune_run[f"sigma_weight_test_mean@{epoch}/train"].upload(File.as_image(fig))
                plt.close(fig)

                fig = plt.figure()
                plt.hist(bnn_dict[f"sigma_bias_test_mean@{epoch}"], edgecolor="black", bins=20)
                neptune_run[f"sigma_bias_test_mean@{epoch}/train"].upload(File.as_image(fig))
                plt.close(fig)

                fig = plt.figure()
                plt.hist(bnn_dict[f"mu_weight_test_std@{epoch}"], edgecolor="black", bins=20)
                neptune_run[f"mu_weight_test_std@{epoch}/train"].upload(File.as_image(fig))
                plt.close(fig)

                fig = plt.figure()
                plt.hist(bnn_dict[f"mu_bias_test_std{epoch}"], edgecolor="black", bins=20)
                neptune_run[f"mu_bias_test_std@{epoch}/train"].upload(File.as_image(fig))
                plt.close(fig)

                fig = plt.figure()
                plt.hist(bnn_dict[f"sigma_weight_test_std@{epoch}"], edgecolor="black", bins=20)
                neptune_run[f"sigma_weight_test_std@{epoch}/train"].upload(File.as_image(fig))
                plt.close(fig)

                fig = plt.figure()
                plt.hist(bnn_dict[f"sigma_bias_test_std@{epoch}"], edgecolor="black", bins=20)
                neptune_run[f"sigma_bias_test_std@{epoch}/train"].upload(File.as_image(fig))
                plt.close(fig)

            metrics = metrics or dict()
            metrics["lr"] = scheduler.get_lr()
            metrics["accuracy/val"] = acc
            metrics["accuracy/val_max"] = max_acc
            metrics["accuracy/train_max"] = max_train_acc
            metrics["reparam_scaling"] = min(1,(epoch-params.hn_warmup_start_epoch) / (params.hn_warmup_stop_epoch-params.hn_warmup_start_epoch)) if epoch >= params.hn_warmup_start_epoch else 0
            metrics = {
                **metrics,
                **test_loop_metrics,
                **max_acc_adaptation_dict
            }

            if params.hm_set_forward_with_adaptation:
                for i in range(params.hn_val_epochs + 1):
                    if i != 0:
                        metrics[f"accuracy/val_support_max@-{i}"] = max_acc_adaptation_dict[
                            f"accuracy/val_support_max@-{i}"]
                    metrics[f"accuracy/val_max@-{i}"] = max_acc_adaptation_dict[f"accuracy/val_max@-{i}"]

            if metrics["accuracy/train"] > max_train_acc:
                max_train_acc = metrics["accuracy/train"]

            if params.hm_set_forward_with_adaptation:
                for i in range(params.hn_val_epochs + 1):
                    if i != 0 and metrics[f"accuracy/val_support_acc@-{i}"] > max_acc_adaptation_dict[
                        f"accuracy/val_support_max@-{i}"]:
                        max_acc_adaptation_dict[f"accuracy/val_support_max@-{i}"] = metrics[
                            f"accuracy/val_support_acc@-{i}"]

                    if metrics[f"accuracy/val@-{i}"] > max_acc_adaptation_dict[f"accuracy/val_max@-{i}"]:
                        max_acc_adaptation_dict[f"accuracy/val_max@-{i}"] = metrics[f"accuracy/val@-{i}"]

            if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                print("--> Best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

                upload_images(neptune_run, hist_data, "best")

                if params.maml_save_feature_network and params.method in ['maml', 'hyper_maml']:
                    outfile = os.path.join(params.checkpoint_dir, 'best_feature_net.tar')
                    torch.save({'epoch': epoch, 'state': model.feature.state_dict()}, outfile)

            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

            if params.maml_save_feature_network and params.method in ['maml', 'hyper_maml']:
                outfile = os.path.join(params.checkpoint_dir, 'last_feature_net.tar')
                torch.save({'epoch': epoch, 'state': model.feature.state_dict()}, outfile)

            if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

            if metrics is not None:
                for k, v in metrics.items():
                    metrics_per_epoch[k].append(v)

            with (Path(params.checkpoint_dir) / "metrics.json").open("w") as f:
                json.dump(metrics_per_epoch, f, indent=2)

            if neptune_run is not None:
                for m, v in metrics.items():
                    neptune_run[m].log(v, step=epoch)

    neptune_run["best_model"].track_files(os.path.join(params.checkpoint_dir, 'best_model.tar'))
    neptune_run["last_model"].track_files(os.path.join(params.checkpoint_dir, 'last_model.tar'))

    if params.maml_save_feature_network:
        neptune_run["best_feature_net"].track_files(os.path.join(params.checkpoint_dir, 'best_feature_net.tar'))
        neptune_run["last_feature_net"].track_files(os.path.join(params.checkpoint_dir, 'last_feature_net.tar'))

    if len(delta_params_list) > 0 and params.hm_save_delta_params:
        with (Path(params.checkpoint_dir) / f"delta_params_list_{len(delta_params_list)}.json").open("w") as f:
            json.dump(delta_params_list, f, indent=2)

    return model


def plot_metrics(metrics_per_epoch: Dict[str, Union[List[float], float]], epoch: int, fig_dir: Path):
    for m, values in metrics_per_epoch.items():
        plt.figure()
        if "accuracy" in m:
            plt.ylim((0, 100))
        plt.errorbar(
            list(range(len(values))),
            [
                np.mean(v) if isinstance(v, list) else v for v in values
            ],
            [
                np.std(v) if isinstance(v, list) else 0 for v in values
            ],
            ecolor="black",
            fmt="o",
        )
        plt.grid()
        plt.title(f"{epoch}- {m}")
        plt.savefig(fig_dir / f"{m}.png")
        plt.close()


def get_scheduler(params, optimizer, stop_epoch=None) -> lr_scheduler._LRScheduler:
    if params.lr_scheduler == "multisteplr":
        if params.milestones is not None:
            milestones = params.milestones
        else:
            milestones = list(range(0, params.stop_epoch, params.stop_epoch // 4))[1:]

        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                        gamma=0.3)
    elif params.lr_scheduler == "none":
        return lr_scheduler.MultiStepLR(optimizer,
                                        milestones=list(range(0, params.stop_epoch, params.stop_epoch // 4))[1:],
                                        gamma=1)

    elif params.lr_scheduler == "cosine":
        T_0 = stop_epoch if stop_epoch is not None else params.stop_epoch // 4
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0
        )

    raise TypeError(params.lr_scheduler)


if __name__ == '__main__':
    params = parse_args('train')
    _set_seed(params.seed)
    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        # params.model = 'Conv4S'
        # no need for this, since omniglot is loaded as RGB

    # optimization = 'Adam'
    optimization = params.optim

    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400  # default
        else:  # meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600  # default

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=16)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        if params.method == 'baseline':
            model = BaselineTrain(model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')

    elif params.method in ['DKT', 'protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml',
                           'maml_approx', 'hyper_maml'] + list(hypernet_types.keys()):
        n_query = max(1, int(
            16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        print("n_query", n_query)
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot, n_query=n_query)
        base_datamgr = SetDataManager(image_size, **train_few_shot_params)  # n_eposide=100
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=n_query)

        val_datamgr = SetDataManager(image_size, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if (params.method == 'DKT'):
            dkt_train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
            model = DKT(model_dict[params.model], **dkt_train_few_shot_params)
            model.init_summary()
        elif params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model](flatten=False)
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['maml', 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(model_dict[params.model], params=params, approx=(params.method == 'maml_approx'),
                         **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1

        elif params.method in hypernet_types.keys():
            hn_type: Type[HyperNetPOC] = hypernet_types[params.method]
            model = hn_type(model_dict[params.model], params=params, **train_few_shot_params)
        elif params.method == "hyper_maml":
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = HyperMAML(model_dict[params.model], params=params, approx=(params.method == 'maml_approx'),
                              **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    if params.checkpoint_suffix != "":
        params.checkpoint_dir = params.checkpoint_dir + "_" + params.checkpoint_suffix
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params.checkpoint_dir)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method in ['maml', 'maml_approx', 'hyper_maml']:
        stop_epoch = params.stop_epoch * model.n_task  # maml use multiple tasks in one update

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        print(resume_file)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
            print("Resuming training from", resume_file, "epoch", start_epoch)

    elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
            configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.",
                                         "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    args_dict = vars(params.params)
    with (Path(params.checkpoint_dir) / "args.json").open("w") as f:
        json.dump(
            {
                k: v if isinstance(v, (int, str, bool, float)) else str(v)
                for (k, v) in args_dict.items()
            },
            f,
            indent=2,
        )

    with (Path(params.checkpoint_dir) / "rerun.sh").open("w") as f:
        print("python", " ".join(sys.argv), file=f)

    neptune_run = setup_neptune(params)

    if neptune_run is not None:
        neptune_run["model"] = str(model)

    if not params.evaluate_model:
        model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params,
                      neptune_run=neptune_run)

    params.split = "novel"
    params.save_iter = -1

    try:
        do_save_fts(params)
    except Exception as e:
        print("Cannot save features bc of", e)

    val_datasets = [params.dataset]
    if params.dataset in ["cross", "miniImagenet"]:
        val_datasets = ["cross", "miniImagenet"]

    for idx, d in enumerate(val_datasets):
        print("Evaluating on", d)
        params.dataset = d
        for hn_val_epochs in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 200]:
            params.hn_val_epochs = hn_val_epochs
            params.hm_set_forward_with_adaptation = True
            # add default test params
            params.adaptation = True
            params.repeat = 5

            print(f"Testing with {hn_val_epochs=}")
            test_results, bayesian_dicts = perform_test(params)
            if neptune_run is not None:
                neptune_run[f"full_test/{d}/metrics @ {hn_val_epochs}"] = test_results

            for bayesian_dict in bayesian_dicts:
                if bayesian_dict:
                    for key in bayesian_dict.keys():
                        fig = plt.figure()
                        plt.hist(bayesian_dict[key], edgecolor="black", bins=20)
                        neptune_run[key + f"/test_val_epochs@{hn_val_epochs}_val_dataset@{idx}"].upload(File.as_image(fig))
                        plt.close(fig)


