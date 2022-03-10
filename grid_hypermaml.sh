#!/usr/bin/env bash

CMD="python train.py --dataset cross_char --num_classes 4112 --train_n_way 5 \
 --seed 1 --method hyper_maml --stop_epoch 60 --model Conv4 --n_shot 4 \
 --es_threshold 15 --hn_hidden_size 256 --hn_enhance_embeddings True \
 --hn_save_delta_params True --hn_use_class_batch_input --lr 1e-3 \
 --lr_scheduler multisteplr \
 --checkpoint_suffix FEW_483"

$CMD
