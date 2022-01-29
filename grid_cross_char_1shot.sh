#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate uj

set -ex


which python

CMD="python train.py --dataset cross_char --hn_detach_ft_in_hn 100000 --hn_detach_ft_in_tn 100000 \
  --hn_taskset_print_every 20 \
  --hn_taskset_repeats 10:10-20:5-30:2 --hn_taskset_size 1 --hn_tn_depth 2  \
  --lr 0.001 --method hn_poc_sup_sup_kernel \
  --model Conv4 --num_classes 4112 \
  --optim adam --resume --save_freq 500 --start_epoch 0 --stop_epoch 2000 \
  --test_n_way 5 --train_n_way 5 --hn_val_epochs 0 --n_shot 1 --es_threshold 60 --seed 1 --use_cosine_distance"


for HN_HEAD_LEN in 2 1;
do
  for HN_NECK_LEN in 0 1;
  do
    for HN_HIDDEN_SIZE in 128 256 512;
    do
      for HN_TN_HIDDEN_SIZE in 64 128;
      do
        SUFFIX="${HN_HEAD_LEN}_${HN_NECK_LEN}_${HN_HIDDEN_SIZE}_${HN_TN_HIDDEN_SIZE}_cosine"
        $CMD --hn_head_len $HN_HEAD_LEN --hn_neck_len $HN_NECK_LEN --hn_hidden_size $HN_HIDDEN_SIZE \
          --hn_tn_hidden_size $HN_TN_HIDDEN_SIZE --checkpoint_suffix $SUFFIX
      done
    done
  done
done
