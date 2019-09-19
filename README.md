# Structural Transformer Model

This repository contains the code for our paper ["Modeling Graph Structure in Transformer for Better AMR-to-Text Generation"](https://arxiv.org/abs/1909.00136) in EMNLP-IJCNLP 2019

The code is developed under Pytorch 1.0 Due to the compitibility reason of Pytorch, it may not be loaded by some lower version (such as 0.4.0).

Please create issues if there are any questions! This can make things more tractable.

## About AMR 
AMR is a graph-based semantic formalism, which can unified representations for several sentences of the same meaning. Comparing with other structures, such as dependency and semantic roles, the AMR graphs have several key differences:

* AMRs only focus on concepts and their relations, so no function words are included. Actually the edge labels serve the role of function words.
* Inflections are dropped when converting a noun, a verb or named entity into a AMR concept. Sometimes a synonym is used instead of the original word. This makes more unified AMRs so that each AMR graph can represent more sentences.
* Relation tags (edge labels) are predefined and are not extracted from text (like the way OpenIE does). More details are in the official AMR page [AMR website@ISI](https://amr.isi.edu/download.html), where you can download the public-available AMR bank: [little prince](https://amr.isi.edu/download/amr-bank-struct-v1.6.txt). Try it for fun!

## Data precrocessing 
### Baseline Input 
Our baseline use the depth-first traversal strategy as in [Konstas et al.](https://github.com/sinantie/NeuralAmr) to linearize AMR graphs to obtail simplified AMRs. We remove variables, wiki links and sense tags before linearization.

```
(a / and
            :op1 (b / begin-01
                  :ARG1 (i / it)
                  :ARG2 (t / thing :wiki "Massachusetts_health_care_reform"
                        :name (n / name :op1 "Romneycare")))
            :op2 (e / end-01
                 :ARG1 i
                 :ARG2 t))
```

need to be simplified as:

`
and :op1 ( begin :arg1 it :arg2 ( thing :name ( name :op1 romneycare )  )  ) :op2 ( end :arg1 it :arg2 thing )
`

Of course, after the transformation is complete, you still need to do [Byte Pair Encoding (BPE)](https://github.com/rsennrich/subword-nmt) on it. As for the target end, we use the [PTB_tokenizer](https://nlp.stanford.edu/software/tokenizer.shtml) from Stanford corenlp to preprocess our data. We also provide sample input for baseline ([./corpus_sample/baseline_corpus](https://github.com/Amazing-J/structural-transformer/tree/master/corpus_sample/baseline_corpus)).

### Structural Transformer Input

**Structure-Aware Self-Attention:**

$$e_{ij} = \frac{\left(x_iW^Q\right)\left(x_jW^K + r_{ij}W^{R}\right)^{T}}{\sqrt{d_z}}$$

Note that the relation $$r_{ij}$$ is **the vector representation** for element pair ($$x_i$$, $$x_j$$).

We also use the depth-first traversal strategy to linearize AMR graphs to obtain simplified AMRs which only consist of concepts. As show below, the input sequence is much shorter than the input sequence in the baseline.
	**-train_src** For example: [corpus_sample/.../train_concept_no_EOS_bpe](https://github.com/Amazing-J/structural-transformer/blob/master/corpus_sample/five_path_corpus/train_concept_no_EOS_bpe)
	
` and  begin  it  thing  name  romneycare  end  it  thing `

Besides, we also obtain a matrix which records the graph structure between every concept pairs, which implies their semantic relationship.

**Learning Graph Structure Representation for Concept Pairs**

The above structure-aware self-attention is capable of incorporating graph structure between concept pairs. We use a sequence of edge labels, along the path from $$x_i$$ to $$x_j$$ to indicate the AMR graph structure between concepts $$x_i$$ and $$x_j$$. In order to distinguish the edge direction, we add a direction symbol to each label with $$\uparrow$$ for climbing up along the path, and $$\downarrow$$ for going down. Specifically, for the special case of $$i==j$$, we use ***None*** as the path. 

**Feature-based Method**

A natural way to represent the structural path is to view it as a string feature. To this end, we combine the labels in the structural path into a string. The model used is [./opennmt-feature](https://github.com/Amazing-J/structural-transformer/tree/master/opennmt-feature).
The parameter **-train_structure** represents the structural relationship in the AMR graph. We give the corresponding corpus sample [corpus_sample/all_path_corpus](https://github.com/Amazing-J/structural-transformer/blob/master/corpus_sample/all_path_corpus/train_edge_all_bpe). 
Each line in the corpus represents the structural relationship between all nodes in an AMR graph. Assuming $$n$$ concept nodes are input( **-train_src** ), there will be $${(n+1)}^2$$ tokens in this line, each token representing a path relationship ( There's also an **EOS** token at the end of the input sequence, so it is $${(n+1)}^2$$. 

**Avg\Sum\CNN\SA-based Method**

To overcome the data sparsity in the above feature-based method, we view the structural path as a label sequence. We give the corresponding corpus sample [corpus_sample/five_path_corpus](https://github.com/Amazing-J/structural-transformer/tree/master/corpus_sample/five_path_corpus) .
We split the **-train_structure** file in the above feature-based method into several corpus, which are **-train_edge_all_bpe_1**, **-train_edge_all_bpe_2**, and so on. For example, **-train_edge_all_bpe_1** only contains the first token of each structure path, **-train_edge_all_bpe_2** only contains the second token of each structure path, and so on. (In our experiment, it is optimal to set the length to 4, which means that we only use the first four corpus.)

After the corresponding corpus is prepared, modify the PATH within "preprocess.sh". You should pay attention to the field "data_dir", which a directory of pre-processed data that will be used during training. We usually use the experiment setting, such as "./workspace/data". Finally, execute the corresponding script file, such as ```bash preprocess.sh```. 
Data preprocessing is completed.

## Training 

First, modify the PATH within "train.sh". "data_prefix" is the preprocessing directory we mentioned above. Note the prefix gq. For example "./workspace/data/gq". Finally, execute the corresponding script file, such as ```bash train.sh```.

## Decoding 

All you need to do is change the PATH in the "translate.sh" accordingly, and then execute ```bash translate.sh```.

## Cite 

If you like our paper, please cite
```
@inproceedings{zhu2019structural-transformer,
  title={Modeling Graph Structure in Transformer for Better AMR-to-Text Generation},
  author={Zhu, Jie and Li, Junhui and Zhu, Muhua and Qian, Longhua and Zhang, Min and Zhou, Guodong},
  booktitle={Proceedings of the EMNLP-IJCNLP 2019},
  pages={},
  year={2019}
}
```



