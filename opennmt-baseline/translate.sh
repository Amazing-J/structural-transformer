#!/bin/bash

train_test_data_dir='/home/fanjy/jzhu/corpus/LDC2015E86/baseline'
model_file='./workspace/model3/_step_280000.pt'


CUDA_VISIBLE_DEVICES=0  python3 ./translate.py \
                        -model $model_file  -src $train_test_data_dir/dev_source_bpe \
                        -output ./workspace/translate-result/test_target.tran   \
                        -beam_size 5 \
                        -share_vocab \
                        -gpu 0

