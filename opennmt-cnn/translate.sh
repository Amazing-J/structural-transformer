#!/bin/bash

train_test_data_dir='/home/fanjy/jzhu/corpus/LDC2017T10/no_indirect_corpus'
model_file='./workspace/no_indirct_model/_step_290000.pt'
output_dir='./workspace/translate-result'




CUDA_VISIBLE_DEVICES=1  python3  ./translate.py  -model       $model_file   \
                                                 -src         $train_test_data_dir/test_concept_no_EOS_bpe \
                                                 -structure1  $train_test_data_dir/test_edge_all_bpe_1  \
                                                 -structure2  $train_test_data_dir/test_edge_all_bpe_2  \
                                                 -structure3  $train_test_data_dir/test_edge_all_bpe_3  \
                                                 -structure4  $train_test_data_dir/test_edge_all_bpe_4  \
                                                 -structure5  $train_test_data_dir/test_edge_all_bpe_5  \
                                                 -structure6  $train_test_data_dir/test_edge_all_bpe_6  \
                                                 -structure7  $train_test_data_dir/test_edge_all_bpe_7  \
                                                 -structure8  $train_test_data_dir/test_edge_all_bpe_8  \
                                                 -output      $output_dir/test_target.tran  -share_vocab  \
                                                 -gpu 0




