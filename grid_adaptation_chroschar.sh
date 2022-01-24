#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate uj

set -x


which python

CMD="python train.py --dataset cross_char --num_classes 4112 --method hn_poc --train_n_way 5 --seed 1 --resume --save_freq 249 --stop_epoch 1001 --eval_freq 1 --es_threshold 77 --n_val_perms 1 "

for HN_TASKSET_SIZE in 1;
do
for DETACH_EPOCH in  250 100000;
do
  for HN_HIDDEN_SIZE in 2048 1024;
  do
    for HN_TN_HIDDEN_SIZE in 256 128 512;
    do
      for HN_TN_DEPTH in 2 1 3;
      do
        for HN_NECK_LEN in 0 1;
        do
          for HN_HEAD_LEN in 1 2;
          do
            for LR in 1e-4;
	    do
              for HN_VAL_EPOCHS in 0;
	      do
		for HN_VAL_LR in 1e-4;
              	do
		  for LR_SCHED in none; #cosine ;
		  do
		  for HN_AGG in concat mean sum;
		  do
              	SUFFIX="${HN_HIDDEN_SIZE}-${HN_TN_HIDDEN_SIZE}-${HN_TASKSET_SIZE}-${HN_NECK_LEN}-${HN_HEAD_LEN}_${HN_TN_DEPTH}_${LR}_${LR_SCHED}_adapt_${HN_VAL_EPOCHS}_${HN_VAL_LR}_val_ens_v3_eval_freq_1_agg_${HN_AGG}"
              	$CMD --hn_hidden_size $HN_HIDDEN_SIZE \
                	--hn_tn_hidden_size $HN_TN_HIDDEN_SIZE \
                	--hn_taskset_size $HN_TASKSET_SIZE \
                	--hn_neck_len $HN_NECK_LEN \
               		 --hn_head_len $HN_HEAD_LEN \
                	--hn_tn_depth $HN_TN_DEPTH \
                	--hn_detach_ft_in_hn $DETACH_EPOCH \
                	--hn_detach_ft_in_tn $DETACH_EPOCH \
			--lr $LR \
			--hn_val_epochs $HN_VAL_EPOCHS \
			--hn_val_lr $HN_VAL_LR \
			--lr_scheduler $LR_SCHED \
                        --hn_sup_aggregation $HN_AGG \
                	--checkpoint_suffix $SUFFIX
		done
		done
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
