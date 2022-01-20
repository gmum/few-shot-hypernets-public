#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate uj

set -ex


which python

CMD="python train.py --dataset cross_char --hn_detach_ft_in_hn 100000 --hn_detach_ft_in_tn 100000 --hn_head_len 1 --hn_hidden_size 512 --hn_kernel_convolution_output_dim 256 --hn_kernel_hidden_dim 64 --hn_kernel_invariance_type attention --hn_kernel_layers_no 4 --hn_neck_len 0 --hn_taskset_print_every 20 --hn_taskset_repeats 10:10-20:5-30:2 --hn_taskset_size 1 --hn_tn_depth 2 --hn_tn_hidden_size 1024 --hn_transformer_feedforward_dim 512 --hn_transformer_heads_no 1 --hn_transformer_layers_no 1 --kernel_transformer_feedforward_dim 512 --kernel_transformer_heads_no 1 --kernel_transformer_layers_no 1 --lr 0.001 --method hn_poc_sup_sup_kernel --model Conv4 --num_classes 4112 --optim adam --resume --save_freq 500 --start_epoch 0 --stop_epoch 2000 --test_n_way 5 --train_n_way 5 --hn_val_epochs 0 --n_shot 1 --es_threshold 20"

for SEED in 1 2 3;
do
  SUFFIX="512-1024--0-1_2_1e-4-64-4_sup_sup_kernel_rerun_v12_no_ft_s_${SEED}"
  $CMD --seed $SEED --checkpoint_suffix $SUFFIX
done
