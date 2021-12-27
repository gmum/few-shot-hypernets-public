#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate few_shot_hypernets

set -xe

which python

CMD="python train.py --dataset cross_char --num_classes 4112 --method hn_poc_sup_sup_kernel --train_n_way 5 --seed 1 --resume --no_self_relations"



for DETACH_EPOCH in 100000;
do
  for HN_HIDDEN_SIZE in 2048 1024;
  do
    for HN_TN_HIDDEN_SIZE in 1024 512 256;
    do
      for HN_KERNEL_LAYERS_NUM in 4 2;
      do
        for HN_KERNEL_HIDDEN_DIM in 64 128 256;
        do
          for HN_TN_DEPTH in 1 2;
          do
            for HN_NECK_LEN in 1 0;
            do
              for HN_HEAD_LEN in 1;
              do
                for LR in 1e-4;
                do
                  SUFFIX="${HN_HIDDEN_SIZE}-${HN_TN_HIDDEN_SIZE}-${HN_TASKSET_SIZE}-${HN_NECK_LEN}-${HN_HEAD_LEN}_${HN_TN_DEPTH}_${LR}-${HN_KERNEL_HIDDEN_DIM}-${HN_KERNEL_LAYERS_NUM}_sup_sup_kernel_no_self_relations"
                  $CMD --hn_hidden_size $HN_HIDDEN_SIZE \
                    --hn_tn_hidden_size $HN_TN_HIDDEN_SIZE \
                    --hn_taskset_size 1 \
                    --hn_neck_len $HN_NECK_LEN \
                    --hn_head_len $HN_HEAD_LEN \
                    --hn_tn_depth $HN_TN_DEPTH \
                    --hn_detach_ft_in_hn $DETACH_EPOCH \
                    --hn_detach_ft_in_tn $DETACH_EPOCH \
                    --stop_epoch 1001 \
                    --checkpoint_suffix $SUFFIX \
                    --hn_kernel_layers_no $HN_KERNEL_LAYERS_NUM \
                    --hn_kernel_hidden_dim $HN_KERNEL_HIDDEN_DIM
                done
                wait
                rm save/checkpoints/cross_char/Conv4_hn_poc_*/*.tar
                wait
              done
            done
          done
        done
      done
    done
  done
done
