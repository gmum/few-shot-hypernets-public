#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate few_shot_hypernets

set -xe

which python

CMD="python train.py --dataset cross --n_shot 1 --test_n_way 5 --num_classes 200 --method hn_poc_sup_sup_kernel --train_n_way 5 --seed 1 --resume --train_aug --use_cosine_nn_kernel"

for HN_KERNEL_OUT_SIZE in 400 50 100;
do
  for HN_KERNEL_LAYERS_NUM in 2 1 4;
  do
    for HN_KERNEL_HIDDEN_DIM in 400 50 100;
    do
      for DETACH_EPOCH in 100000;
      do
        for HN_HIDDEN_SIZE in 4096;
        do
          for HN_TN_HIDDEN_SIZE in 1024;
          do
            for HN_TN_DEPTH in 1;
            do
              for HN_NECK_LEN in 2;
              do
                for HN_HEAD_LEN in 3;
                do
                  for LR in 1e-3;
                  do
                    SUFFIX="cosine_nn_kernel_1600-${HN_KERNEL_LAYERS_NUM}x${HN_KERNEL_HIDDEN_DIM}-${HN_KERNEL_OUT_SIZE}"
                    $CMD --hn_hidden_size $HN_HIDDEN_SIZE \
                      --hn_kernel_out_size $HN_KERNEL_OUT_SIZE \
                      --hn_tn_hidden_size $HN_TN_HIDDEN_SIZE \
                      --hn_taskset_size 1 \
                      --hn_neck_len $HN_NECK_LEN \
                      --hn_head_len $HN_HEAD_LEN \
                      --hn_tn_depth $HN_TN_DEPTH \
                      --hn_detach_ft_in_hn $DETACH_EPOCH \
                      --hn_detach_ft_in_tn $DETACH_EPOCH \
                      --stop_epoch 1001 \
                      --es_epoch 10000 \
                      --save_freq 500 \
                      --checkpoint_suffix $SUFFIX
                  done
                  wait
                done
              done
            done
          done
        done
      done
    done
  done
done