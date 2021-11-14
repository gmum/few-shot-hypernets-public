#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate uj

set -xe


which python

CMD="python train.py --dataset cross_char --num_classes 4112 --method hn_poc --train_n_way 5 --seed 1 --resume"


for HN_HIDDEN_SIZE in 512;
do
  for HN_TN_HIDDEN_SIZE in 256 128;
  do
    for HN_TASKSET_SIZE in 8 4 1;
    do
      for HN_NECK_LEN in 1 0;
      do
        for HN_HEAD_LEN in 2 1;
        do
          for LR in 1e-2 1e-3 1e-4;
          do
            SUFFIX="${HN_HIDDEN_SIZE}-${HN_TN_HIDDEN_SIZE}-${HN_TASKSET_SIZE}-${HN_NECK_LEN}-${HN_HEAD_LEN}_${LR}_long"
            $CMD --hn_hidden_size $HN_HIDDEN_SIZE \
              --hn_tn_hidden_size $HN_TN_HIDDEN_SIZE \
              --hn_taskset_size $HN_TASKSET_SIZE \
              --hn_neck_len $HN_NECK_LEN \
              --hn_head_len $HN_HEAD_LEN \
              --stop_epoch 1000
              --checkpoint_suffix $SUFFIX &
          done
          wait
        done

      done
    done
  done
done
