# Few-shot Hypernets

The official PyTorch implementation of papers: 

* *[HyperShot: Few-Shot Learning by Kernel HyperNetworks](https://openaccess.thecvf.com/content/WACV2023/html/Sendera_HyperShot_Few-Shot_Learning_by_Kernel_HyperNetworks_WACV_2023_paper.html)* (2023) 
Sendera M., Przewięźlikowski M., Karanowski K., Zięba M. Tabor J., Spurek P. - WACV 2023.

* *[The general framework for few-shot learning by kernel HyperNetworks
](https://link.springer.com/article/10.1007/s00138-023-01403-4)* (2023) Sendera M., Przewięźlikowski M., Miksa J., Rajski M., Karanowski K., Zięba M. Tabor J., Spurek P. -  Machine Vision and Applications Volume 34, article number 53

* *[HyperMAML: Few-Shot Adaptation of Deep Models with Hypernetworks](https://arxiv.org/abs/2205.15745)* (2022)
Przewięźlikowski M., Przybysz P. , Tabor J., Zięba M., Spurek P. - preprint.

  
* *[Hypernetwork approach to Bayesian MAML](https://arxiv.org/abs/2210.02796)* (2022)
Borycki P., Kubacki P., Przewięźlikowski M., Kuśmierczyk T., Tabor J., Spurek P., preprint.


## Overview

### HyperShot

Few-shot models aim at making predictions using a minimal number of labeled examples from a given task. The main challenge
in this area is the one-shot setting where only one element represents each class. We propose HyperShot - the fusion of 
kernels and hypernetwork paradigm. Compared to reference approaches that apply a gradient-based adjustment of the parameters, our
model aims to switch the classification module parameters depending on the task's embedding. In practice, we utilize a 
hypernetwork, which takes the aggregated information from support data and returns the classifier's parameters handcrafted 
for the considered problem. Moreover, we introduce the kernel-based representation of the support examples delivered to 
hypernetwork to create the parameters of the classification module. Consequently, we rely on relations between embeddings
of the support examples instead of direct feature values provided by the backbone models. Thanks to this approach, our model
can adapt to highly different tasks.

### BayesianHyperShot

While HyperShot obtains very good results, it is limited by typical problems such as poorly quantified uncertainty 
due to limited data size.
We further show that incorporating Bayesian neural networks into our general framework, 
an approach we call BayesHyperShot, solves this issue.

### HyperMAML
The aim of Few-Shot learning methods is to train models which can easily adapt to previously unseen tasks, based on small 
amounts of data. One of the most popular and elegant Few-Shot learning approaches is Model-Agnostic Meta-Learning (MAML).
The main idea behind this method is to learn the general weights of the meta-model, which are further adapted to specific 
problems in a small number of gradient steps. However, the model's main limitation lies in the fact that the update procedure 
is realized by gradient-based optimisation. In consequence, MAML cannot always modify weights to the essential level in one or 
even a few gradient iterations. On the other hand, using many gradient steps results in a complex and time-consuming optimization
procedure, which is hard to train in practice, and may lead to overfitting. In this paper, we propose HyperMAML, a novel 
generalization of MAML, where the training of the update procedure is also part of the model. Namely, in HyperMAML, instead
of updating the weights with gradient descent, we use for this purpose a trainable Hypernetwork. Consequently, in this 
framework, the model can generate significant updates whose range is not limited to a fixed number of gradient steps. 
Experiments show that HyperMAML consistently outperforms MAML and performs comparably to other state-of-the-art techniques
in a number of standard Few-Shot learning benchmarks.

### BayesHMAML
The main goal of Few-Shot learning algorithms is to enable learning from small
amounts of data. One of the most popular and elegant Few-Shot learning ap-
proaches is Model-Agnostic Meta-Learning (MAML). The main idea behind this
method is to learn shared universal weights of a meta-model, which then are
adapted for specific tasks. However, due to limited data size, the method suffers
from over-fitting and poorly quantifies uncertainty. Bayesian approaches could, in
principle, alleviate these shortcomings by learning weight distributions in place of
point-wise weights. Unfortunately, previous Bayesian modifications of MAML are
limited in a way similar to the classic MAML, e.g., task-specific adaptations must
share the same structure and can not diverge much from the universal meta-model.
Additionally, task-specific distributions are considered as posteriors to the universal
distributions working as priors and optimizing them jointly with gradients is hard
and poses a risk of getting stuck in local optima.
In this paper, we propose BayesHMAML, a novel generalization of Bayesian
MAML, which employs Bayesian principles along with Hypernetworks for MAML.
We achieve better convergence than the previous methods by classically learning
universal weights. Furthermore, Bayesian treatment of the specific tasks enables
uncertainty quantification, and high flexibility of task adaptations is achieved using
Hypernetworks instead of gradient-based updates. Consequently, the proposed
approach not only improves over the previous methods, both classic and Bayesian
MAML in several standard Few-Shot learning benchmarks but also benefits from
the properties of the Bayesian framework.

## Requirements

1. Python >= 3.7
2. Numpy >= 1.19
3. [pyTorch](https://pytorch.org/) >= 1.11
4. [GPyTorch](https://gpytorch.ai/) >= 1.5.1
5. (optional) [neptune-client](https://neptune.ai/) for logging traning results into your Neptune project.
 

### Installation

```
pip install numpy torch torchvision gpytorch h5py pillow
```


## Code of our method

* (Bayesian) HyperShot: [hypernet_kernel.py](./methods/hypernets/hypernet_kernel.py)
* HyperMAML: [hypermaml.py](./methods/hypernets/hypermaml.py)
* BayesHMAML [bayeshmaml.py](./methods/hypernets/bayeshmaml.py)



## Running the code

You can run 
```
python train.py --h to list all the possible arguments.
```

The [train.py](./train.py) script performs the whole training, evaluation and final test procedure.

For re-running our experiments, please refer to the [commands.sh](./commands.sh) file.

Check out [commands.sh](./commands.sh) for our best grid test arguments

### Methods

This repository provides implementations of several few-shot learning methods:
* `hyper_shot` - [HyperShot: Few-Shot Learning by Kernel HyperNetworks](https://arxiv.org/abs/2203.11378) /  [The general framework for few-shot learning by kernel HyperNetworks
](https://link.springer.com/article/10.1007/s00138-023-01403-4)
* `hyper_maml` - [HyperMAML: Few-Shot Adaptation of Deep Models with Hypernetworks](https://arxiv.org/abs/2205.15745)
* `bayes_hmaml` - [Hypernetwork approach to Bayesian MAML](https://arxiv.org/abs/2210.02796)
* `hn_ppa` - [Few-Shot Image Recognition by Predicting Parameters from Activations
](https://arxiv.org/abs/1706.03466)
* `DKT` - [Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels
](https://arxiv.org/abs/1910.05199)
* `maml`, `maml_approx` - [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
](https://arxiv.org/abs/1703.03400)
* `protonet` - [Prototypical Networks for Few-shot Learning
](https://arxiv.org/abs/1703.05175)
* `relationnet` - [Learning to Compare: Relation Network for Few-Shot Learning
](https://arxiv.org/abs/1711.06025)
* `matchingnet` - [Matching Networks for One Shot Learning
](https://arxiv.org/abs/1606.04080)
* `baseline++` - [A Closer Look at Few-Shot Classification](https://arxiv.org/abs/1904.04232)
* `baseline` - Feature Transfer

You must use those exact strings at training and test time when you call the script (see below). 

### Datasets


This is an example of how to download and prepare a dataset for training/testing. Here we assume the current directory is the project root folder:

```
cd filelists/DATASET_NAME/
sh download_DATASET_NAME.sh
```
Replace `DATASET_NAME` with one of the following: `omniglot`, `CUB`, `miniImagenet`, `emnist`, `QMUL`. Notice that mini-ImageNet is a large dataset that requires substantial storage, therefore you can save the dataset in another location and then change the entry in `configs.py` in accordance.

These are the instructions to train and test the methods reported in the paper in the various conditions.

In addition, you can select `cross_char`  and `cross` datasets for **cross-domain classification** of 
**Omnglot &rarr; EMNIST** and **mini-ImageNet &rarr; CUB**, respectively.

### Backbones

The script allows training and testing on different backbone networks. By default the script will use the same backbone used in our experiments (`Conv4`). Check the file `backbone.py` for the available architectures, and use the parameter `--model=BACKBONE_STRING` where `BACKBONE_STRING` is one of the following: `Conv4`, `Conv6`, `ResNet10|18|34|50|101`.

### Neptune

We provide logging the training / validation metrics and details to [Neptune](https://neptune.ai/). To do so, one must export the following env variables before running `train.py`.

```bash
export NEPTUNE_PROJECT=...
export NEPTUNE_API_TOKEN=...
```


Acknowledgements
---------------

This repository is a fork of: [https://github.com/BayesWatch/deep-kernel-transfer](https://github.com/BayesWatch/deep-kernel-transfer), which in turn is a fork of [https://github.com/wyharveychen/CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).

## Bibtex citations

If you find our work useful, please consider citing it:

```bibtex

@InProceedings{sendera2023hypershot,
    author    = {Sendera, Marcin and Przewi\k{e}\'zlikowski, Marcin and Karanowski, Konrad and Zi\k{e}ba, Maciej and Tabor, Jacek and Spurek, Przemys{\l}aw},
    title     = {HyperShot: Few-Shot Learning by Kernel HyperNetworks},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {2469-2478}
}
```
```bibtex
@article{sendera2023thegeneral,
author = {Sendera, Marcin and Przewięźlikowski, Marcin and Miksa, Jan and Rajski, Mateusz and Karanowski, Konrad and Zięba, Maciej and Tabor, Jacek and Spurek, Przemysław},
year = {2023},
month = {05},
pages = {},
title = {The general framework for few-shot learning by kernel HyperNetworks},
volume = {34},
journal = {Machine Vision and Applications},
doi = {10.1007/s00138-023-01403-4}
}
```
```bibtex
@misc{przewiezlikowski2022hypermaml,
  doi = {10.48550/ARXIV.2205.15745},
  url = {https://arxiv.org/abs/2205.15745},
  author = {Przewięźlikowski, M. and Przybysz, P. and Tabor, J. and Zięba, M. and Spurek, P.},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {HyperMAML: Few-Shot Adaptation of Deep Models with Hypernetworks},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
```bibtex
@misc{borycki2023hypernetwork,
      title={Hypernetwork approach to Bayesian MAML}, 
      author={Piotr Borycki and Piotr Kubacki and Marcin Przewięźlikowski and Tomasz Kuśmierczyk and Jacek Tabor and Przemysław Spurek},
      year={2023},
      eprint={2210.02796},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```