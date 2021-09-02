import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class VocPointerLoss(nn.Module):
    def __init__(self, type, vocab_sizeT, unk_idx):
        """
        types:
        1: original loss as in the paper (predict with pointer only for OOV)
        2: predict with vocabulary only if cannot point
        3: min(voc loss, pointer loss)
        4: average: if can predict with both vocabulary and pointer, use (voc loss + pointer loss) / 2, otherwise use available loss
        5: random: if can predict with both vocabulary and pointer, use random(voc loss, pointer loss), otherwise use available loss
        """
        self.type = type
        super(VocPointerLoss, self).__init__()
        self.vocab_sizeT = vocab_sizeT
        self.unk_idx = unk_idx
        self.lossfun = nn.NLLLoss(reduction='none')

    def forward(self, output, target_voc, target_attn):
        """
        output (FloatTensor): batch_size x (vocab_sizeT+attn_size)
        target_voc, target_attn (LongTensor): batch_size
        target_voc: <vocab_sizeT, =unk_idx => ignore
        target_attn: <=attn_size, <=0 => ignore
        """
        mask_voc = (target_voc != self.unk_idx)
        mask_attn = (target_attn > 0)
        mask_11 = mask_voc & mask_attn
        mask_10 = mask_voc & (~mask_attn)
        mask_01 = (~mask_voc) & mask_attn
        l = target_voc.new_zeros(target_voc.shape, dtype=output.dtype)
        if self.type == 1:
            if mask_voc.sum() > 0:
                l[mask_voc] = self.lossfun(output[mask_voc], \
                                       target_voc[mask_voc])
            if output[mask_01, self.vocab_sizeT:].numel() > 0:
                l[mask_01] = self.lossfun(output[mask_01, self.vocab_sizeT:],\
                                      target_attn[mask_01]-1)
            summa = (mask_voc+mask_01).sum()
            return (l.sum() / summa) if summa > 0 else 0
        if self.type == 2:
            if mask_attn.sum() > 0:
                l[mask_attn] = self.lossfun(output[mask_attn, \
                                                   self.vocab_sizeT:],\
                                      target_attn[mask_attn]-1)
            if mask_10.sum() > 0:
                l[mask_10] = self.lossfun(output[mask_10],\
                                         target_voc[mask_10])
            summa = (mask_attn+mask_10).sum()
            return (l.sum()/summa) if summa > 0 else 0
        if self.type >= 3:
            if mask_10.sum() > 0:
                l[mask_10] = self.lossfun(output[mask_10], target_voc[mask_10])
            if mask_01.sum() > 0:
                l[mask_01] = self.lossfun(output[mask_01, self.vocab_sizeT:],\
                                      target_attn[mask_01]-1)
            if mask_11.sum() > 0:
                if self.type == 3:
                    l[mask_11] = torch.min(self.lossfun(output[mask_11],\
                                                target_voc[mask_11]),\
                                  self.lossfun(output[mask_11, self.vocab_sizeT:],\
                                               target_attn[mask_11]-1))
                elif self.type == 4:
                    l[mask_11] = self.lossfun(output[mask_11],\
                                                target_voc[mask_11]) * 0.5 + \
                                 self.lossfun(output[mask_11, self.vocab_sizeT:],\
                                               target_attn[mask_11]-1) * 0.5
                else: # self.type = 5
                    randmask = mask_11.new_zeros(mask_11.shape,\
                                                 dtype=float).bernoulli_().bool()
                    if (mask_11&randmask).sum() > 0:
                        l[mask_11&randmask] = self.lossfun(output[mask_11&randmask],\
                                                target_voc[mask_11&randmask])
                    if (mask_11&(~randmask)).sum() > 0:
                        l[mask_11&(~randmask)] = self.lossfun(\
                                             output[mask_11&(~randmask), \
                                                    self.vocab_sizeT:],\
                                               target_attn[mask_11&(~randmask)]-1)
            summa = (mask_01+mask_10+mask_11).sum()
            return (l.sum() / summa) if summa > 0 else 0
        
def compute_acc(ans, batch_T_raw, batch_T_raw_prev, vocab_sizeT,
                vocab_sizeT_cl, unk_idx, eof_idx):
    ans_comb = ans * (ans < vocab_sizeT_cl).long() + (-1)*(ans>=vocab_sizeT_cl).long()
    ans_comb[ans==eof_idx] = -55 # same as in batch_T_raw (see dataloader)
    ans_idx = torch.max(ans - vocab_sizeT, \
                        ans.new_zeros(ans.shape).long()-1) + 1
               # if nonzero, then equals to shift to the left
    curlen = batch_T_raw.shape[1]
    m = (ans_idx>0) & (ans_idx<=torch.arange(curlen, device=ans.device)[None, :])
    idxs_1 = (torch.arange(curlen, device=ans.device)[None, :]-ans_idx)[m] 
    # shift to the left
    idxs_0 = torch.arange(ans.shape[0], device=ans.device)[:, None].\
           repeat([1, curlen])[m]
    ans_comb[m] = batch_T_raw[idxs_0, idxs_1]
    prevlen = batch_T_raw_prev.shape[1]
    m = (ans_idx>0) & (ans_idx>torch.arange(curlen, device=ans.device)[None, :]) & \
          (ans_idx<=torch.arange(prevlen, curlen+prevlen, device=ans.device)\
           [None, :])
    idxs_1 = (torch.arange(curlen, device=ans.device)[None, :]-ans_idx)[m]
    idxs_0 = torch.arange(ans.shape[0], device=ans.device)[:, None].\
    repeat([1, curlen])[m]
    ans_comb[m] = batch_T_raw_prev[idxs_0, idxs_1]
    ans_comb[ans==unk_idx] = -23 # ignore in computing acc
    acc = (ans_comb==batch_T_raw).float().mean()
    return acc

def load(cpt, use_cuda_if_av=True):
    m = torch.load(cpt, map_location="cuda" if torch.cuda.is_available() and use_cuda_if_av else 'cpu')
    if "state_dict" in m:
        m = m["state_dict"]
    return m

def sd2tensor(sd):
    res = None
    for key in sd:
        if res is None:
            res = sd[key].new_zeros(0)
        res = torch.cat((res, sd[key].reshape(-1).float()))
    return res
    
class VocabManager:
    def __init__(self, vocab_size_in, vocab_size_out=None, anonym=None, static_vocab_size=1):
        self.vocab_size_in_cl = vocab_size_in
        self.vocab_size_out_cl = vocab_size_out if vocab_size_out is not None else vocab_size_in
        self.different = (vocab_size_out is not None)
        self.vocab_size_in = self.vocab_size_in_cl + 2
        self.vocab_size_out = self.vocab_size_out_cl + 2
        self.eof_idxT_in = vocab_size_in
        self.unk_idx_in = vocab_size_in + 1
        self.eof_idxT_out = self.vocab_size_out_cl
        self.unk_idx_out = self.vocab_size_out_cl + 1
        if anonym is not None:
            self.static_x_ids_in = list(range(static_vocab_size))+[self.eof_idxT_in, self.unk_idx_in]
            self.static_x_ids_out = list(range(static_vocab_size))+[self.eof_idxT_out, self.unk_idx_out]
    
    def reset(self):
        pass
    
    def get_batch_T_voc(self, batch_T_raw):
        batch_T_voc_in = batch_T_raw*((batch_T_raw<self.vocab_size_in_cl)&\
                               (batch_T_raw!=-55)).long() +\
                  self.eof_idxT_in * (batch_T_raw==-55).long() + \
                  self.unk_idx_in * (batch_T_raw>=self.vocab_size_in_cl).long()
        if self.different:
            batch_T_voc_out = batch_T_raw*((batch_T_raw<self.vocab_size_out_cl)&\
                               (batch_T_raw!=-55)).long() +\
                  self.eof_idxT_out * (batch_T_raw==-55).long() + \
                  self.unk_idx_out * (batch_T_raw>=self.vocab_size_out_cl).long()
        else:
            batch_T_voc_out = batch_T_voc_in + 0
        
        return batch_T_voc_in, batch_T_voc_out