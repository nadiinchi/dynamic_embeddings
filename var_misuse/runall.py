import argparse
import os
import numpy as np
import pandas as pd
import datetime
import time

parser = argparse.ArgumentParser(description='Managing experiments')
parser.add_argument('--num_runs', type=int, default=1, metavar='MODEL', \
                    required=False, help='number of runs of each model')
parser.add_argument('--test', action='store_true',
                        help='what to do with generated commands: print (to check commands) or os.system (to run comands)')
parser.add_argument('--label', type=str, default="run", metavar='MODEL', required=False, help='label used in naming log folders')
parser.add_argument('--comment_add', type=str, default="", metavar='MODEL', required=False, help='if you wish to add anyth to log folder name')

args = parser.parse_args()
    
if args.test:
    action = print
else:
    action = os.system
 
commands = []
for lang in ["py", "js"]:
    for comment, data_type, model_type in [("static", "full", "rnn"), \
                                      ("dynamic", "full", "rnn_dynemb_mixed"), \
                                      ("static", "ano", "rnn"), \
                                      ("dynamic", "ano", "rnn_dynemb")]:
        dataname = "python" if lang=="py" else "js"
        data_label = "values" if data_type=="full" else "anovalues"
        data_options = " --train_src traverse_%s_train.txt --dev_src traverse_%s_test.txt "%(data_label, data_label) # types and targets are specified in default values in train.py
        emsize = 500 if comment == "dynamic" else 1200
        command = "train.py --dir logs_rnn/"+args.label+"/"+lang+"_"+data_type+"data"+args.comment_add+"/ --data_dir preprocessed_data_vm/ "+data_options+" --comment "+comment+" --max_src_len 250 --print_fq 1 --checkpoint True --learning_rate 0.0001 --grad_clipping 1000 --lr_decay 0.6 --num_epochs 10 --dataset_name "+dataname+" --model_type "+model_type+" --bidirection True --emsize "+str(emsize)+" --emsize_type 300"
        commands.append(command)
    
def get_run_command(command):
    ### add everything you need to run training, e. g. set CUDA_VISIBLE_DEVICES or use sbatch
    return "python3 " + command    

for command in commands:
    for _ in range(1 if args.test else args.num_runs):
        action(get_run_command(command))
        

