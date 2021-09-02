# RNNs with Dynamic Embeddings for Source Code Processing

The official PyTorch implementation of:
* __On the Embeddings of Variables in Recurrent Neural Networks for Source Code__ [[arxiv](https://arxiv.org/abs/2010.12693)] (accepted to [NAACL'21](https://2021.naacl.org/))

## Repository structure
* `code_completion`: code for the code completion task (additional preprocessing, models, training etc)
* `var_misuse`: code for the variable misuse task (additional preprocessing, models, training etc)

Please refer to these subfolders for each task's instructions.

## Data
The experiments were conducted on the [Python150k](https://www.sri.inf.ethz.ch/py150) and [JavaScript150k](https://www.sri.inf.ethz.ch/js150) datasets, resplitted according to https://github.com/bayesgroup/code_transformers. Please follow [this instruction](https://github.com/bayesgroup/code_transformers/tree/main/data_utils) to obtain data. 

## Run

The experiments were run on a system with Linux 3.10.0 using Tesla V100 GPU. The implementation is based on PyTorch>=1.5.

Running experiments:
1. Download and resplit data, see [this instruction](https://github.com/bayesgroup/code_transformers/tree/main/data_utils for details;
2. Preprocess data for a task you are interested in, see `code_completion` or `var_misuse` for details;
3. Run the experiment you are interested in, see `code_completion` or `var_misuse` for details.

## Attribution

Parts of this code are based on the following repositories:
* [A Transformer-based Approach for Source Code Summarization](https://github.com/wasiahmad/NeuralCodeSum) 
* [OpenNMT](https://github.com/OpenNMT/OpenNMT-py)
* [DrQA](https://github.com/facebookresearch/DrQA)
* https://github.com/oleges1/code-completion

## Citation

If you found this code useful, please cite our paper

```
@inproceedings{chirkova2021embeddings,
      title={On the Embeddings of Variables in Recurrent Neural Networks for Source Code}, 
      author={Nadezhda Chirkova},
      booktitle={North American Chapter of the Association for Computational Linguistics}
      year={2021}, 
}
```

