#!/usr/bin/env bash

CMD="python train.py --dataset cross_char --num_classes 4112 --train_n_way 5 \
  --seed 1 --lr 1e-4 --method hyper_maml --stop_epoch 40 --model Conv4 --n_shot 4 --es_threshold 15 \
  --hn_hidden_size 256 --hn_enhance_embeddings True --hn_activation sigmoid --hn_save_delta_params True --lr_scheduler multisteplr"
# CMD2="python train.py --dataset cross_char --num_classes 4112 --train_n_way 5 \
#   --seed 1 --method hyper_maml --stop_epoch 60 --model Conv4 --n_shot 4 --es_threshold 15 \
#   --hn_hidden_size 128 --hn_enhance_embeddings True --hn_save_delta_params True"

for LR in 0.0001;
do
    SUFFIX="${LR}_lr"
    $CMD --lr $LR --checkpoint_suffix $SUFFIX
done
