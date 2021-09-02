# Code completion task

## Dependencies
The implementation uses the following libraries:
* torch (we used version 1.7.1, you can use versions >=1.5 and it should work, but we did not test the code on other versions)
* numpy (we used version 1.16.4)
* scikit-learn
* pandas
* tqdm
* tabulate
* six

## Running experiments
1. Download and resplit data, see https://github.com/bayesgroup/code_transformers/tree/main/data_utils for details. After this step, you will have `data` folder containing files `*.dedup.json`.
2. Preprocess data:
```(bash)
python preprocess.py {PY|JS} {optional: src_dir}
```
`src_dir` specifies path to the `data` folder obtained after step 1. By default, `../data/` is used. PY or JS specifies the dataset language. After this step, you will have `pickle_data` folder. 

3. Run experiments. Script `runall.py` generates commands for the experiments in the [paper](https://arxiv.org/abs/2010.12693). 
```(bash)
python runall.py --dataset {py|js} --model {fulldata_static|fulldata_dynamic|anonym_static|anonym_dynamic} [--test]
```
Script `run_all.py` generates commands of kind `... python3 train.py ...` with arguments depending on the chosen dataset-model pair. Use `--test` flag to check the generated command (it will be just printed) and remove flag to run the command. Make sure you have done data preprocessing (see above).

Additional options for `runall.py` script:
* `--num_runs`: number from experiments runs
* `--label` and `--comment_add`: the logs and models are saved to a folder named `logs/{label}/{exp_group}/{exp_folder}{comment_add}`, and you can specify a general label for your current experiments (default label if `run`) and an additional comment for a partiular run (default empty)

## Attribution
The code is partly based on https://github.com/oleges1/code-completion.
