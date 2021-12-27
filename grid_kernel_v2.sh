#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate few_shot_hypernets

set -xe

which python

export NEPTUNE_PROJECT=''
export NEPTUNE_API_TOKEN=""

CMD="python train.py --dataset cross_char --num_classes 4112 --method hn_poc_kernel --train_n_way 5 --seed 1 --resume"


for HN_KERNEL_LAYERS_NUM in 2 4 1;
do
  for HN_KERNEL_HIDDEN_DIM in 128 256 512;
  do
    for K_TRANSFORMER_LAYERS_NUM in 1;
    do
      for K_TRANSFORMER_HEADS_NUM in 1;
      do
        for K_TRANSFORMER_FF_DIM in 512 1024 2048;
        do
          for DETACH_EPOCH in 100000 250;
          do
            for HN_HIDDEN_SIZE in 4096 2048 1024;
            do
              for HN_TN_HIDDEN_SIZE in 2048 1024 512;
              do
                for HN_TN_DEPTH in 1 2 3;
                do
                  for HN_NECK_LEN in 1 0;
                  do
                    for HN_HEAD_LEN in 1;
                    do
                      for LR in 1e-4 1e-3;
                      do
                        SUFFIX="${HN_HIDDEN_SIZE}-${HN_TN_HIDDEN_SIZE}-${HN_TASKSET_SIZE}-${HN_NECK_LEN}-${HN_HEAD_LEN}_${HN_TN_DEPTH}_${LR}_k-transformer_-${K_TRANSFORMER_HEADS_NUM}-${K_TRANSFORMER_LAYERS_NUM}-${K_TRANSFORMER_FF_DIM}_kernel_${HN_KERNEL_LAYERS_NUM}-${HN_KERNEL_HIDDEN_DIM}_long_kernel"
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
                          --hn_kernel_hidden_dim $HN_KERNEL_HIDDEN_DIM \
                          --kernel_transformer_layers_no $K_TRANSFORMER_LAYERS_NUM \
                          --kernel_transformer_heads_no $K_TRANSFORMER_HEADS_NUM \
                          --kernel_transformer_feedforward_dim $K_TRANSFORMER_FF_DIM
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
    done
  done
done