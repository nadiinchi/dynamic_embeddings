# Variable misuse task

## Dependencies
The implementation uses the following libraries:
* torch (we used version 1.7.1, you can use versions >=1.5 and it should work, but we did not test the code on other versions)
* numpy (we used version 1.16.4)
* pandas
* tqdm
* tabulate

## Running experiments
1. Download and resplit data, see https://github.com/bayesgroup/code_transformers/tree/main/data_utils for details. After this step, you will have `data` folder containing files `*.dedup.json`.
2. Preprocess data: follow instructions [here](https://github.com/bayesgroup/code_transformers/tree/main/vm_fn#step-1-basic-preprocessing): "Step 1: basic preprocessing". After this step, you will have `preprocessed_data_vm` folder.
3. Run experiments. Script `runall.py` generates commands for the experiments in the [paper](https://arxiv.org/abs/2010.12693). 
```(bash)
python runall.py [--test]
```
Use `--test` flag to check the generated commands (they will be just printed) and remove flag to run commands. Make sure you have done data preprocessing (see above).

Additional options for `runall.py` script:
* `--num_runs`: number from experiments runs
* `--label` and `--comment_add`: the logs and models are saved to a folder named `logs/{label}/{exp_group}/{exp_folder}{comment_add}`, and you can specify a general label for your current experiments (default label if `run`) and an additional comment for a partiular run (default empty)

## Attribution
The code is partly based on the following repositories:
* [A Transformer-based Approach for Source Code Summarization](https://github.com/wasiahmad/NeuralCodeSum) 
* [OpenNMT](https://github.com/OpenNMT/OpenNMT-py)
* [DrQA](https://github.com/facebookresearch/DrQA)