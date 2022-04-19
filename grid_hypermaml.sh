#!/usr/bin/env bash

CMD_HYPERMAML="python train.py --dataset cross_char --num_classes 4112 --train_n_way 5 \
 --seed 1 --method hyper_maml --stop_epoch 80 --model Conv4 --n_shot 4 \
 --es_threshold 15 --hn_hidden_size 256 --hn_enhance_embeddings True \
 --hn_use_class_batch_input --lr 1e-3 \
 --hm_maml_warmup --hm_maml_warmup_epochs 50 --hm_maml_warmup_switch_epochs 100 \
 --checkpoint_suffix FEW_500"

CMD_MAML="python train.py --dataset cross_char --num_classes 4112 --train_n_way 5 \
 --seed 1 --method maml --stop_epoch 80 --model Conv4 --n_shot 4 \
 --es_threshold 15 --lr 1e-3"

CMD_HYPERMAML_REF_RUN="python train.py --dataset cross_char --num_classes 4112 \
 --train_n_way 5 --seed 1 --lr 1e-4 --method hyper_maml --stop_epoch 60 \
 --model Conv4 --n_shot 4 --hn_hidden_size 256 --es_threshold 15 --hn_enhance_embeddings True
 --checkpoint_suffix FEW_500"

$CMD_HYPERMAML