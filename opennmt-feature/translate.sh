#!/bin/bash

model_file='/home/jzhu/opennmt-structure/workspace/model/_step_300000.pt'
output_dir='./workspace/translate-result'




CUDA_VISIBLE_DEVICES=2  python3  ./translate.py  -model      $model_file \
                                                 -src        /home/jzhu/corpus/LDC2015E86/dev_concept_no_EOS_bpe \
                                                 -structure  /home/jzhu/corpus/LDC2015E86/all_path/dev_edge_all  \
                                                 -output     $output_dir/test_target.tran \
                                                 -beam_size 5 \
                                                 -share_vocab  \
                                                 -gpu 0




