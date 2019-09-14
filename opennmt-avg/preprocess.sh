#!/bin/bash

train_test_data_dir='/home/fanjy/jzhu/corpus/LDC2015E86'

data_dir='./workspace/data/path_data'
if [ ! -d "$data_dir" ]; then mkdir -p "$data_dir"; fi
data_prefix="$data_dir/gq"

python3 ./preprocess.py -train_src              $train_test_data_dir/train_concept_no_EOS_bpe \
                       -train_tgt               $train_test_data_dir/train_target_token_bpe \
                       -train_structure1        $train_test_data_dir/five_path/train_edge_all_bpe_1  \
                       -train_structure2        $train_test_data_dir/five_path/train_edge_all_bpe_2  \
                       -train_structure3        $train_test_data_dir/five_path/train_edge_all_bpe_3  \
                       -train_structure4        $train_test_data_dir/five_path/train_edge_all_bpe_4  \
                       -train_structure5        $train_test_data_dir/five_path/train_edge_all_bpe_5  \
                       -valid_src               $train_test_data_dir/dev_concept_no_EOS_bpe  \
                       -valid_tgt               $train_test_data_dir/dev_target_token_bpe \
                       -valid_structure1        $train_test_data_dir/five_path/dev_edge_all_bpe_1   \
                       -valid_structure2        $train_test_data_dir/five_path/dev_edge_all_bpe_2   \
                       -valid_structure3        $train_test_data_dir/five_path/dev_edge_all_bpe_3   \
                       -valid_structure4        $train_test_data_dir/five_path/dev_edge_all_bpe_4   \
                       -valid_structure5        $train_test_data_dir/five_path/dev_edge_all_bpe_5   \
                       -save_data $data_prefix \
                       -src_vocab_size 20000  \
                       -tgt_vocab_size 20000 \
                       -structure_vocab_size 20000 \
                       -src_seq_length 10000 \
                       -tgt_seq_length 10000 \
                       -share_vocab





