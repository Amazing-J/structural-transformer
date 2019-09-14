#!/bin/bash

data_prefix='./workspace/data/self_no_indirect_data/gq'
model_dir='./workspace/self_no_indirect_model2015/'
if [ ! -d "$model_dir" ]; then mkdir -p "$model_dir"; fi

CUDA_VISIBLE_DEVICES=0 nohup python3 train.py \
                        -data $data_prefix \
                        -save_model $model_dir \
                        -world_size 1 \
                        -gpu_ranks 0  \
                        -save_checkpoint_steps 5000 \
                        -valid_steps 5000 \
                        -report_every 20 \
                        -keep_checkpoint 50 \
                        -seed 3435 \
                        -train_steps 300000 \
                        -warmup_steps 16000 \
                        --share_decoder_embeddings \
                        -share_embeddings \
                        --position_encoding \
                        --optim adam \
                        -adam_beta1 0.9 \
                        -adam_beta2 0.98 \
                        -decay_method noam \
                        -learning_rate 0.5 \
                        -max_grad_norm 0.0 \
                        -batch_size 4096 \
                        -batch_type tokens \
                        -normalization tokens \
                        -dropout 0.3 \
                        -label_smoothing 0.1 \
                        -max_generator_batches 100 \
                        -param_init 0.0 \
                        -param_init_glorot \
                        -valid_batch_size 4 \
                        -accum_count 1  > no_indirect_self_2015 2>&1 & 



