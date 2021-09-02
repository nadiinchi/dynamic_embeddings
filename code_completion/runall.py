import os
import time
import sys
import argparse
import string
import random
    
parser = argparse.ArgumentParser(description='Run a lot of experiments')
parser.add_argument('--dataset', type=str, default='py', metavar='DATASET', required=True, help='dataset name (py, js)')
parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True, help='model name: fulldata_static|fulldata_dynamic|anonym_static|anonym_dynamic|vocab1')
parser.add_argument('--num_runs', type=int, default=1, metavar='MODEL', required=False, help='number of runs of each model')
parser.add_argument('--test', action='store_true',
                        help='print (test) or os.system (run)')
parser.add_argument('--label', type=str, default="nov", metavar='MODEL', help='logs and models are saved to logs/{label}/exp_group_comment[folder_add]/exp_comment[comment_add]')
parser.add_argument('--comment_add', type=str, default="", metavar='TEXT', required=False, help='logs and models are saved to ogs/{label}/exp_group_comment[folder_add]/exp_comment[comment_add]')
parser.add_argument('--folder_add', type=str, default="", metavar='TEXT', required=False, help='logs and models are saved to ogs/{label}/exp_group_comment[folder_add]/exp_comment[comment_add]')
parser.add_argument('--add_args', type=str, default="", metavar='MODEL', required=False, help='if you wish to add any args to all commands')

args = parser.parse_args()
args.dataset = args.dataset.lower()
assert args.dataset in {"py", "js"}

if args.test:
    action = print
else:
    action = os.system

data_spec_full = " --N_filename pickle_data/JS_non_terminal.pickle --T_filename pickle_data/JS_terminal.pickle" if args.dataset=="js" else "" # PY is default in train.py
data_spec_anonym = " --N_filename pickle_data/JS_non_terminal.pickle --T_filename pickle_data/JS_terminal_anon_vocab500.pickle" if args.dataset=="js" else " --T_filename pickle_data/PY_terminal_anon_vocab500.pickle"

def get_run_command(command):
    ### add everything you need to run training, e. g. set CUDA_VISIBLE_DEVICES or use sbatch
    return "python3 " + command

vocab_full = 50000
vocab_anonym = 500

if args.model == "fulldata_static":
    for model_type_spec, comment in [("", "pointer"), (" --no_pointer", "attn"), \
                                     (" --no_attn --no_pointer", "simple")]:
        if not args.test:
            time.sleep(5)
            num_runs = args.num_runs
        else:
            num_runs = 1
        comment = comment + args.comment_add
        for i in range(num_runs):
            action(get_run_command('train.py --save_verbose --dir logs/%s/%s_fulldata_static%s/ --comment fulldata_static_vocab%d_%s --vocab_size %d%s%s%s'%(args.label, args.dataset, args.folder_add, vocab_full, comment, vocab_full, data_spec_full, model_type_spec, args.add_args)))
            
elif args.model == "anonym_static":
    for model_type_spec, comment in [("", "pointer"), (" --no_pointer", "attn"), \
                                     (" --no_attn --no_pointer", "simple")]:
        if not args.test:
            time.sleep(5)
            num_runs = args.num_runs
        else:
            num_runs = 1
        loss_type_spec = " --loss_type=3" if comment == "pointer" else ""
        comment = comment + args.comment_add
        for i in range(num_runs):
            action(get_run_command('train.py --save_verbose --dir logs/%s/%s_static%s/ --comment static_vocab%d_%s --vocab_size %d%s%s%s%s'%(args.label, args.dataset, args.folder_add, vocab_anonym, comment, vocab_anonym, data_spec_anonym, model_type_spec, loss_type_spec, args.add_args))) 

elif args.model == "anonym_dynamic":
    for model_type_spec, comment in [("", "pointer"), (" --no_pointer", "attn"), \
                                     (" --no_attn --no_pointer", "simple")]:
        if not args.test:
            time.sleep(5)
            num_runs = args.num_runs
        else:
            num_runs = 1
        comment = comment + args.comment_add
        for i in range(num_runs):
            action(get_run_command('train.py --save_verbose --dir logs/%s/%s_dynamic%s/ --comment dynamic_vocab%d_%s --vocab_size %d%s%s --anonym=pure_uni --embedding_sizeT 500%s'%(args.label, args.dataset, args.folder_add, vocab_full, comment, vocab_anonym, data_spec_anonym, model_type_spec, args.add_args)))

elif args.model == "vocab1":
    for model_type_spec, comment in [("", "pointer"), (" --no_pointer", "attn"), \
                                     (" --no_attn --no_pointer", "simple")]:
        if not args.test:
            time.sleep(5)
            num_runs = args.num_runs
        else:
            num_runs = 1
        comment = comment + args.comment_add
        for i in range(num_runs):
            action(get_run_command('train.py --save_verbose --dir logs/%s/%s_vocab1%s/ --comment vocab1_%s --vocab_size 1%s%s%s'%(args.label, args.dataset, args.folder_add, comment, data_spec_full, model_type_spec, args.add_args)))
            
elif args.model == "fulldata_dynamic":
    for model_type_spec, comment in [("", "pointer"), 
                                     (" --no_pointer", "attn"), \
                                     (" --no_attn --no_pointer", "simple")
                                     ]:
        if not args.test:
            time.sleep(5)
            num_runs = args.num_runs
        else:
            num_runs = 1
        comment = comment + args.comment_add
        action(get_run_command('train.py --save_verbose --dir logs/%s/%s_mixed_uni_fulldata%s/ --comment fulldata_dynamic_vocab%d_%s --vocab_size %d%s%s --anonym=mixed_uni --embedding_sizeT 500%s'%(args.label, args.dataset, args.folder_add, vocab_full, comment, vocab_full, data_spec_full, model_type_spec, args.add_args)))