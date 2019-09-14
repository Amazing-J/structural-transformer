#!/bin/bash

train_test_data_dir='/home/jzhu/corpus/LDC2015E86/all_path'

data_dir='./workspace/data/all_path_data'
if [ ! -d "$data_dir" ]; then mkdir -p "$data_dir"; fi
data_prefix="$data_dir/gq"

python3 ./preprocess.py -train_src /home/jzhu/corpus/LDC2015E86/train_concept_no_EOS_bpe \
                        -train_tgt /home/jzhu/corpus/LDC2015E86/train_target_token_bpe \
                        -train_structure  /home/jzhu/corpus/LDC2015E86/all_path/train_edge_all  \
                        -valid_src /home/jzhu/corpus/LDC2015E86/dev_concept_no_EOS_bpe  \
                        -valid_tgt /home/jzhu/corpus/LDC2015E86/dev_target_token_bpe \
                        -valid_structure /home/jzhu/corpus/LDC2015E86/all_path/dev_edge_all   \
                        -save_data $data_prefix \
                        -src_vocab_size 10000  \
                        -tgt_vocab_size 10000 \
                        -structure_vocab_size 5000 \
                        -src_seq_length 10000 \
                        -tgt_seq_length 10000 \
                        -share_vocab





