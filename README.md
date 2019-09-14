# Structural Transformer Model

This repository contains the code for our paper ["Modeling Graph Structure in Transformer for Better AMR-to-Text Generation"](https://arxiv.org/abs/1909.00136)in EMNLP 2019

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
```
and :op1 ( begin :arg1 it :arg2 ( thing :name ( name :op1 romneycare )  )  ) :op2 ( end :arg1 it :arg2 thing )
```
Of course, after the transformation is complete, you still need to do [Byte Pair Encoding (BPE)](https://github.com/rsennrich/subword-nmt) on it. As for the target end, we use the [PTB_tokenizer](https://nlp.stanford.edu/software/tokenizer.shtml) from Stanford corenlp to preprocess our data. We also provide sample input for baseline ([./corpus_sample/baseline_corpus](https://github.com/Amazing-J/structural-transformer/tree/master/corpus_sample/baseline_corpus)).

### Structural Transformer Input
We also use the depth-first traversal strategy to linearize AMR graphs to obtain simplified AMRs which only consist of concepts. 
```
 and  begin  it  thing  name  romneycare  end  it  thing
```
As show above, the input sequence is much shorter than the input sequence in the baseline. Besides, we also obtain a matrix which records the graph structure between every concept pairs, which implies their semantic relationship.



