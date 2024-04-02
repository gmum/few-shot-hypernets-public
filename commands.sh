# HyperShot

## cross_char

### 1-shot
python train.py --method hyper_shot --model Conv4 --dataset cross_char --num_classes 4112 \
  --n_shot 1 --test_n_way 5 --train_n_way 5 \
  --stop_epoch 2000 --hn_val_epochs 0 \
  --hn_head_len 2 --hn_neck_len 1 --hn_hidden_size 512 --hn_tn_hidden_size 128 --hn_tn_depth 2 \
  --hn_use_cosine_distance

### 5-shot
python train.py --method hyper_shot --model Conv4 --dataset cross_char --num_classes 4112 \
  --n_shot 5 --test_n_way 5 --train_n_way 5 \
  --stop_epoch 2000 --hn_val_epochs 0 \
  --hn_head_len 2 --hn_neck_len 1 --hn_hidden_size 512 --hn_tn_hidden_size 128 --hn_tn_depth 2 \
  --hn_sup_aggregation mean --hn_use_cosine_distance

## CUB

### 1-shot

python train.py --method hyper_shot --model Conv4 --dataset CUB --num_classes 200 \
 --n_shot 1 --test_n_way 5 --train_n_way 5 --train_aug \
 --stop_epoch 10000 --hn_val_epochs 0 --es_threshold 20 \
 --hn_tn_depth 2 --hn_head_len 2 --hn_neck_len 0 --hn_hidden_size 128 --hn_tn_hidden_size 64 \
 --hn_use_cosine_distance

### 5-shot

python train.py --method hyper_shot --model Conv4 --dataset CUB --num_classes 200 \
  --n_shot 5 --test_n_way 5 --train_n_way 5 --train_aug \
  --stop_epoch 10000 --hn_val_epochs 0 --es_threshold 20 \
  --hn_tn_depth 2  --hn_head_len 2 --hn_neck_len 0 --hn_hidden_size 128 --hn_tn_hidden_size 64 \
  --hn_sup_aggregation mean --hn_use_cosine_distance

# Bayesian HyperShot

## cross_char
### 1-shot

python train.py --method hyper_shot --hn_bayesian_model  --hn_bayesian_test --model Conv4 --dataset cross_char --num_classes 4112 \
  --n_shot 1 --test_n_way 5 --train_n_way 5--stop_epoch 2000 --hn_val_epochs 0 --hn_val_lr 0.01 \
  --hn_head_len 2 --hn_neck_len 1 --hn_hidden_size 512 --hn_tn_hidden_size 128 --hn_tn_depth 2  \
  --hn_use_kld --hn_S=3 --hn_use_cosine_distance --hn_kld_const_scaler -4 --hn_S_test=3 --hn_use_mu_in_kld


## cross
### 5-shot

python train.py --dataset miniImagenet --n_shot 5 --test_n_way 5 --num_classes 200 --method hyper_shot\
 --train_n_way 5 --seed 1 --resume --train_aug --hn_use_cosine_distance --hn_hidden_size 4096 --hn_tn_hidden_size 1024\
  --hn_taskset_size 1 --hn_neck_len 2 --hn_head_len 3 --hn_tn_depth 1 --hn_detach_ft_in_hn 100000\
   --hn_detach_ft_in_tn 100000 --stop_epoch 10001 --es_epoch 10000 --save_freq 500 --hn_val_epochs 0 --hn_use_kld --hn_S 5\
     --hn_kld_const_scaler 0 --hn_S_test 10 --hn_bayesian_model --hn_bayesian_test --maybe_terminate\
      --detect_anomaly --hn_use_kld_scheduler --hn_kld_start_val -24 --hn_use_mu_in_kld --hn_kld_stop_val -3

# HyperMAML

## cross_char

### 1-shot
python train.py --method hyper_maml --model Conv4 --dataset cross_char --num_classes 4112 \
  --n_shot 1 --test_n_way 5 --train_n_way 5 \
  --stop_epoch 64 --lr_scheduler multisteplr --lr 1e-2 \
  --hm_maml_warmup --hm_maml_warmup_epochs 50 --hm_maml_warmup_switch_epochs 500 --milestones 51 550 \
  --hn_head_len 3 --hn_hidden_size 512 --hm_enhance_embeddings True --hm_use_class_batch_input


### 5-shot
python train.py --method hyper_maml --model Conv4 --dataset cross_char --num_classes 4112 \
  --n_shot 5 --test_n_way 5 --train_n_way 5 \
  --stop_epoch 64 --lr_scheduler multisteplr --lr 1e-2 \
  --hm_maml_warmup --hm_maml_warmup_epochs 50 --hm_maml_warmup_switch_epochs 500 --milestones 51 550 \
  --hn_head_len 3 --hn_hidden_size 512 --hm_enhance_embeddings True --hm_use_class_batch_input --hn_sup_aggregation mean


## CUB

### 1-shot
python train.py --method hyper_maml --model Conv4Pool --dataset CUB --num_classes 200 \
  --n_shot 1 --test_n_way 5 --train_n_way 5 --train_aug \
  --stop_epoch 1000  --es_threshold 20 --lr 1e-3 --lr_scheduler multisteplr \
  --hm_maml_warmup --hm_maml_warmup_epochs 100  --hm_maml_warmup_switch_epochs 1000 --milestones 101 1100 \
  --hn_head_len 3 --hn_hidden_size 256 --hm_enhance_embeddings True --hm_use_class_batch_input

### 5-shot
python train.py --method hyper_maml --model Conv4Pool --dataset CUB --num_classes 200 \
  --n_shot 5 --test_n_way 5 --train_n_way 5 --train_aug \
  --stop_epoch 1000  --es_threshold 20 --lr 1e-3 --lr_scheduler multisteplr \
  --hm_maml_warmup --hm_maml_warmup_epochs 100  --hm_maml_warmup_switch_epochs 1000 --milestones 101 1100 \
  --hn_head_len 3 --hn_hidden_size 256 --hm_enhance_embeddings True --hm_use_class_batch_input --hn_sup_aggregation mean


## miniImagenet

### 1-shot
python train.py --method hyper_maml --model Conv4Pool --dataset miniImagenet --num_classes 200 \
  --n_shot 1 --test_n_way 5 --train_n_way 5 --train_aug \
  --stop_epoch 1000  --es_threshold 20 --lr 1e-3 --lr_scheduler multisteplr \
  --hm_maml_warmup --hm_maml_warmup_epochs 100  --hm_maml_warmup_switch_epochs 1000 --milestones 101 1100 \
  --hn_head_len 3 --hn_hidden_size 256 --hm_enhance_embeddings True --hm_use_class_batch_input

### 5-shot
python train.py --method hyper_maml --model Conv4Pool --dataset miniImagenet --num_classes 200 \
  --n_shot 5 --test_n_way 5 --train_n_way 5 --train_aug \
  --stop_epoch 1000  --es_threshold 20 --lr 1e-3 --lr_scheduler multisteplr \
  --hm_maml_warmup --hm_maml_warmup_epochs 100  --hm_maml_warmup_switch_epochs 1000 --milestones 101 1100 \
  --hn_head_len 3 --hn_hidden_size 256 --hm_enhance_embeddings True --hm_use_class_batch_input --hn_sup_aggregation mean

# BayesHMAML

## cross_char

### 1-shot
python train.py --hm_weight_set_num_test 0 --es_threshold 10 --dataset cross_char --num_classes 4112 --train_n_way 5 \
  --seed 1 --method bayes_hmaml --stop_epoch 64 --model Conv4 --hm_enhance_embeddings True --hm_use_class_batch_input \
  --lr_scheduler multisteplr --n_shot 1 --hm_maml_warmup --lr 0.01 --hm_maml_warmup_epochs 50 --hm_maml_warmup_switch_epochs 500 \
  --hn_head_len 3 --hn_hidden_size 512 --milestones 51 550 --kl_stop_val 0.001 --kl_scale 1e-24 --hm_weight_set_num_train 5

### 5-shot
python train.py --model Conv4 --dataset cross_char --num_classes 4112 --n_shot 5 --train_n_way 5 --method bayes_hmaml \
  --stop_epoch 64 --lr_scheduler multisteplr --hm_maml_warmup --hm_maml_warmup_epochs 50 --hm_maml_warmup_switch_epochs 500 \
  --milestones 51 550 --hn_head_len 3 --hn_hidden_size 512 --hm_enhance_embeddings True --hm_use_class_batch_input \
  --hn_sup_aggregation mean --hm_weight_set_num_test 0 --es_threshold 0.0 --kl_scale=1e-24  \
  --hm_weight_set_num_train 5 --kl_stop_val 0.001 --lr 0.01

## CUB

### 1-shot
python train.py --hm_weight_set_num_test 0 --method bayes_hmaml --model Conv4Pool --dataset CUB --num_classes 200 --n_shot 1 \
  --test_n_way 5 --train_n_way 5 --train_aug --stop_epoch 1000 --es_threshold 20 --lr 0.01 --lr_scheduler multisteplr --hm_maml_warmup  \
  --hm_maml_warmup_epochs 100 --hm_maml_warmup_switch_epochs 1000 --milestones 101 1100 --hn_head_len 3 --hn_hidden_size 256  \
  --hm_enhance_embeddings True --hm_use_class_batch_input --kl_scale 1e-24 --kl_stop_val 0.0001 --hm_weight_set_num_train 5

### 5-shot
python train.py --method bayes_hmaml --model Conv4Pool --dataset CUB --num_classes 200 --n_shot 5 --test_n_way 5 --train_n_way 5 \
  --train_aug --stop_epoch 1000 --es_threshold 20 --lr 1e-3 --lr_scheduler multisteplr --hm_maml_warmup --hm_maml_warmup_epochs 100 \
  --hm_maml_warmup_switch_epochs 1000 --milestones 101 1100 --hn_head_len 3 --hn_hidden_size 256 --hm_enhance_embeddings True \
  --hm_use_class_batch_input --hn_sup_aggregation mean --hm_weight_set_num_test 0 \
  --hm_weight_set_num_train 5 --kl_stop_val 1e-5

## miniImageNet

### 1-shot
python train.py --method bayes_hmaml --hm_weight_set_num_test 0 --model Conv4Pool --dataset miniImagenet --num_classes 200 --n_shot 1  \
  --test_n_way 5 --train_n_way 5 --train_aug --stop_epoch 1000 --es_threshold 20 --lr 0.001 --lr_scheduler multisteplr --hm_maml_warmup  \
  --hm_maml_warmup_epochs 100 --hm_maml_warmup_switch_epochs 1000 --milestones 101 1100 --hn_head_len 3 --hn_hidden_size 256  \
  --hm_enhance_embeddings True --hm_use_class_batch_input --kl_scale 1e-24 --kl_stop_val 0.0001 --hm_weight_set_num_train 5

### 5-shot
python train.py --method bayes_hmaml --model Conv4Pool --dataset miniImagenet --num_classes 200 --n_shot 5 --test_n_way 5 \
  --train_n_way 5 --train_aug --stop_epoch 1000 --es_threshold 20 --lr 1e-3 --lr_scheduler multisteplr --hm_maml_warmup \
  --hm_maml_warmup_epochs 100 --hm_maml_warmup_switch_epochs 1000 --milestones 101 1100 --hn_head_len 3 --hn_hidden_size 256 \
  --hm_enhance_embeddings True --hm_use_class_batch_input --hn_sup_aggregation mean --resume --hm_weight_set_num_test 0 \
  --hm_weight_set_num_train 5 --kl_stop_val 1e-5
