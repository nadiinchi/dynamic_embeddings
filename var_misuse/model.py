import copy
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from config import override_model_args
from inputters import constants
from modules import EncoderRNNSimple, DynamicRNNEncoder, DynamicRNNEncoderBidirectional
from prettytable import PrettyTable

EMPTY_VAL = "<emptyvalue>" # from https://github.com/bayesgroup/code_transformers/blob/main/vm_fn/preprocess/select_functions_and_save_seq_data.py


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    :param lengths: 1d tensor [batch_size]
    :param max_len: int
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)  # (0 for pad positions)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

class RNN(nn.Module):
    """Module that predicts two pointers given input code passage"""

    def __init__(self, args):
        """"Constructor of the class."""
        super(RNN, self).__init__()

        self.name = 'RNN'

        self.dynamic = (args.model_type != "rnn")
        if args.model_type == "rnn":
            self.encoder = EncoderRNNSimple(\
                                hidden_size=args.nhid,
                                vocab_sizeT=args.src_vocab_size,
                                vocab_sizeN=args.type_vocab_size,
                                embedding_sizeT=args.emsize,
                                embedding_sizeN=args.emsize_type,
                                dropout=0,
                                num_layers=1,
                                bidirectional=args.bidirection
                            )

        else:
            encoder_class = DynamicRNNEncoder \
                            if not args.bidirection else \
                            DynamicRNNEncoderBidirectional
            self.encoder = encoder_class(anonym_type=\
                                "pure_uni" if args.model_type=="rnn_dynemb" \
                                           else "mixed_uni",
                                hidden_size=args.nhid,
                                vocab_sizeT=args.src_vocab_size,
                                vocab_sizeN=args.type_vocab_size,
                                embedding_sizeN=args.emsize_type,
                                embedding_sizeT=args.emsize,
                                dropout=0,
                                num_layers=1,
                                static_x_ids=args.static_x_ids,\
                                embedding_sizeT_dyn=args.emsize_dyn)
        
        self.decoder_loc = nn.Linear(args.nhid, 1)
        self.decoder_fix = nn.Linear(args.nhid, 1)
        self.nobug_embedding = nn.Parameter(torch.zeros(1))
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def get_logits(self, ex):
        batch_size = ex["code_len"].size(0)
        # embed and encode the source sequence
        inp = (ex["code_type_rep"], ex["code_word_rep"])
        if not self.dynamic:
            memory_bank = self.encoder(inp)
        else:
            memory_bank = self.encoder(inp)
        memory_bank_loc = memory_bank
        memory_bank_fix = memory_bank
            
        # B x (1+seq_len) x h
        logits_loc = self.decoder_loc(memory_bank_loc).squeeze(-1) # B x (seq_len)
        logits_loc = torch.cat([self.nobug_embedding[None, :].repeat(batch_size, 1),\
                                logits_loc], dim=1)# B x (1+seq_len)
        logits_fix = self.decoder_fix(memory_bank_fix).squeeze(-1) # B x seq_len
        return logits_loc, logits_fix
    
    def get_loss(self, logits, ex):
        loss = dict()
        loc_predictions, pointer_logits = logits
        seq_mask = sequence_mask(ex["code_len"], logits[0].shape[1]-1)
        # batch_size, 1+seq_len
        seq_mask = torch.cat([seq_mask.new_ones((logits[0].shape[0], 1)), seq_mask],\
                              dim=1).bool()
        # batch_size, 1+seq_len
        loc_predictions = loc_predictions.masked_fill(~seq_mask, -1e18)
        loc_loss = self.criterion(loc_predictions, ex["target_pos"]+1).mean()
        # -1 -> 0, >=                                             0 -> >=0 + 1
        scope_mask = ex["scope_t"] # batch x seq_len
        pointer_logits = pointer_logits.masked_fill(~scope_mask, -1e18)
        pointer_probs = F.softmax(pointer_logits, dim=1) # batch x seq_len
        target_mask = ex["fixes_t"] # batch x seq_len
        target_probs = (target_mask * pointer_probs).sum(dim=-1) # batch
        if ex["mask_incorrect"].sum() > 0:
            target_loss = (ex["mask_incorrect"].float() * (-torch.log(target_probs + 1e-9))).sum()  / (1e-9 + ex["mask_incorrect"].sum())
        else:
            target_loss = logits[0].new_zeros(1).sum()
                
        loss["loc_loss"] = loc_loss
        loss["target_loss"] = target_loss
        loss["total_loss"] = loc_loss + target_loss
        return loss

    def forward(self, ex):
        """
        Input:
            - code_word_rep: ``(batch_size, max_doc_len)``
            - code_char_rep: ``(batch_size, max_doc_len, max_word_len)``
            - code_len: ``(batch_size)``
            - summ_word_rep: ``(batch_size, max_que_len)``
            - summ_char_rep: ``(batch_size, max_que_len, max_word_len)``
            - summ_len: ``(batch_size)``
            - tgt_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        logits = self.get_logits(ex)
        if self.training:
            return self.get_loss(logits, ex)
        else:
            return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_encoder_parameters(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def count_decoder_parameters(self):
        s = 0
        for m in [self.decoder_loc, self.decoder_fix]:
            s += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return s

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table
    
class VarmisuseModel:
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------
    def __init__(self, args, src_dict, type_dict, state_dict=None):
        self.args = args
        self.src_dict = src_dict
        self.args.src_vocab_size = len(src_dict)
        self.type_dict = type_dict
        self.args.type_vocab_size = len(type_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        
        empty_idx = src_dict.tok2ind[EMPTY_VAL]
        if args.model_type != "rnn_dynembOOV":
            args.static_x_ids = [constants.PAD, constants.UNK, empty_idx]
        else:
            args.static_x_ids = [i for i in src_dict.ind2tok \
                                 if not "<var" in src_dict.ind2tok[i]]
        
        self.network = RNN(self.args)
        
        if state_dict:
            self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """

        if self.args.optimizer == 'sgd':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(parameters,
                                       self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)

        elif self.args.optimizer == 'adam':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(parameters,
                                        self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamW':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.AdamW(parameters,
                                        self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)

        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        if self.use_cuda:
            for key in ex:
                try:
                    ex[key] = ex[key].cuda(non_blocking=True)
                except:
                    pass

        # Run forward
        net_loss = self.network(ex)

        loss = net_loss["total_loss"]

        loss.backward()

        clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.updates += 1
        return {
            'loss': loss,
            "loc_loss": net_loss["loc_loss"],
            "fix_loss": net_loss["target_loss"],
        }

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch examples
        Output:
            predictions: #batch predicted sequences
        """
        # Eval mode
        self.network.eval()

        if self.use_cuda:
            for key in ex:
                try:
                    ex[key] = ex[key].cuda(non_blocking=True)
                except:
                    pass
            
        # Run forward
        logits_loc, logits_bug = self.network(ex)

        return logits_loc, logits_bug

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'src_dict': self.src_dict,
            "type_dict": self.type_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            print('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'src_dict': self.src_dict,
            "type_dict": self.type_dict,
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            print('WARN: Saving failed... continuing anyway.')
            
    @staticmethod
    def load(filename, new_args=None):
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        type_dict = saved_params["type_dict"]
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return VarmisuseModel(args, src_dict, type_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        type_dict = saved_params["type_dict"]
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = VarmisuseModel(args, src_dict, type_dict, state_dict)
        model.updates = updates
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch
    
    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)