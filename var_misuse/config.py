""" Implementation of all available options """
from __future__ import print_function

"""Model architecture/optimization options for Seq2seq architecture."""

import argparse

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type',
    'emsize',
    'emsize_type',
    'emsize_dyn',
    'rnn_type',
    'nhid',
    'bidirection',
}

DATA_OPTIONS = {
    'max_src_len',
    'max_tgt_len',
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'optimizer',
    'learning_rate',
    'momentum',
    'weight_decay',
    'cuda',
    'grad_clipping',
    'lr_decay',
    'warmup_steps',
    'num_epochs',
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Data options
    data = parser.add_argument_group('Data parameters')
    data.add_argument('--max_src_len', type=int, default=400,
                      help='Maximum allowed length for the source sequence')
    data.add_argument('--max_tgt_len', type=int, default=50,
                      help='Maximum allowed length for the target sequence')

    # Model architecture
    model = parser.add_argument_group('Model')
    model.add_argument('--model_type', type=str, default='rnn_dynemb_mixed',
                       choices=["rnn", "rnn_dynemb", "rnn_dynemb_mixed", "rnn_dynembOOV"],
                       help='Model architecture type')
    model.add_argument('--emsize', type=int, default=1200,
                       help='Embedding size for values')
    model.add_argument('--emsize_type', type=int, default=300,
                       help='Embedding size for types')
    model.add_argument('--emsize_dyn', type=int, default=None,
                       help='Embedding size for values (dynamic): if emsize_dyn is not None, additional linear transform will be used on top of dynamic embeddings')
    model.add_argument('--rnn_type', type=str, default='LSTM',
                       help='RNN type: LSTM, GRU')
    model.add_argument('--nhid', type=int, default=1500,
                       help='Hidden size of RNN units')
    model.add_argument('--bidirection', type='bool', default=True,
                       help='use bidirectional recurrent unit')

    # Optimization details
    optim = parser.add_argument_group('Neural QA Reader Optimization')
    optim.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'adamW'],
                       help='Name of the optimizer')
    optim.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate for the optimizer')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='Decay ratio for learning rate')
    optim.add_argument('--grad_clipping', type=float, default=5.0,
                       help='Gradient clipping')
    parser.add_argument('--early_stop', type=int, default=200,
                        help='Stop training if performance doesn\'t improve')
    optim.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--warmup_steps', type=int, default=0,
                       help='Number of of warmup steps')
    optim.add_argument('--warmup_epochs', type=int, default=0,
                       help='Number of of warmup steps')


def get_model_args(args):
    """Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER, DATA_OPTIONS
    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER | DATA_OPTIONS

    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)


def override_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))

    return argparse.Namespace(**old_args)


def add_new_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global ADVANCED_OPTIONS
    old_args, new_args = vars(old_args), vars(new_args)
    for k in new_args.keys():
        if k not in old_args:
            if k in ADVANCED_OPTIONS:
                print('Adding arg %s: %s' % (k, new_args[k]))
                old_args[k] = new_args[k]

    return argparse.Namespace(**old_args)
