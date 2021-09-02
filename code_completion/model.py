import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import utils

def masked_log_softmax(x, mask=None, dim=1):
    """
    from https://gist.github.com/kaniblu/94f3ede72d1651b087a561cf80b306ca
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
    return torch.log(x_exp/x_exp.sum(dim).unsqueeze(-1)+1e-10)

class DecoderSimple(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_sizeT_in,
        vocab_sizeT_out,
        vocab_sizeN,
        embedding_sizeT,
        embedding_sizeN,
        dropout,
        tie_weights=False
    ):
        super(DecoderSimple, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.embeddingN = nn.Embedding(vocab_sizeN, embedding_sizeN)
        self.embeddingT = nn.Embedding(vocab_sizeT_in, embedding_sizeT)

        self.lstm = nn.LSTM(
            embedding_sizeN + embedding_sizeT,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.w_global = nn.Linear(hidden_size * 2, hidden_size) # map into T
        if not tie_weights:
            self.w_out = nn.Linear(hidden_size, vocab_sizeT_out) # map into T
        self.vocab_sizeT_in = vocab_sizeT_in
        self.vocab_sizeT_out = vocab_sizeT_out
        
        self.tie_weights = tie_weights
        
    def embedded_dropout(self, embed, words, scale=None):
        dropout = self.dropout
        if dropout > 0:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = F.embedding(words, masked_embed_weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )
        return X

    def forward(
        self,
        input,
        hc,
        enc_out,
        mask,
        h_parent,
    ):
        n_input, t_input = input
        batch_size = n_input.size(0)
        st = n_input
        
        n_input = self.embedded_dropout(self.embeddingN, n_input)
        t_input = self.embedded_dropout(self.embeddingT, t_input)
        input = torch.cat([n_input, t_input], 1)

        out, (h, c) = self.lstm(input.unsqueeze(1), hc)

        hidden = h[-1] # use only last layer hidden in attention
        out = out.squeeze(1)
        
        w_t = torch.tanh(self.w_global(torch.cat([out, h_parent], dim=1)))
        if not self.tie_weights:
            w_t = self.w_out(w_t)
        else:
            w_t = torch.mm(w_t, self.embeddingT.weight.T)
        
        w_t = F.log_softmax(w_t, dim=1)
        
        return w_t, (h, c)

class DecoderAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_sizeT_in,
        vocab_sizeT_out,
        vocab_sizeN,
        embedding_sizeT,
        embedding_sizeN,
        dropout,
        attn_size=50,
        pointer=True,
        tie_weights=False
    ):
        super(DecoderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.pointer = pointer
        self.dropout = dropout
        
        self.embeddingN = nn.Embedding(vocab_sizeN, embedding_sizeN)
        self.embeddingT = nn.Embedding(vocab_sizeT_in, embedding_sizeT)

        self.W_hidden = nn.Linear(hidden_size, hidden_size)
        self.W_mem2hidden = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        self.W_context = nn.Linear(
            embedding_sizeN + embedding_sizeT + \
            hidden_size,
            hidden_size
        )
        self.lstm = nn.LSTM(
            embedding_sizeN + embedding_sizeT,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.w_global = nn.Linear(hidden_size * 3, hidden_size) # map into T
        if not tie_weights:
            self.w_out = nn.Linear(hidden_size, vocab_sizeT_out) 
        if self.pointer:
            self.w_switcher = nn.Linear(hidden_size * 2, 1)
            self.logsigmoid = torch.nn.LogSigmoid()
            
        self.vocab_sizeT_in = vocab_sizeT_in
        self.tie_weights = tie_weights

    def embedded_dropout(self, embed, words, scale=None):
        dropout = self.dropout
        if dropout > 0:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = F.embedding(words, masked_embed_weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )
        return X

    def forward(
        self,
        input,
        hc,
        enc_out,
        mask,
        h_parent,
    ):
        n_input, t_input = input
        batch_size = n_input.size(0)
        
        n_input = self.embedded_dropout(self.embeddingN, n_input)
        t_input = self.embedded_dropout(self.embeddingT, t_input)
        input = torch.cat([n_input, t_input], 1)

        out, (h, c) = self.lstm(input.unsqueeze(1), hc)

        hidden = h[-1] # use only last layer hidden in attention
        out = out.squeeze(1)

        scores = self.W_hidden(hidden).unsqueeze(1) # (batch_size, max_length, hidden_size)
        if enc_out.shape[1] > 0:
            scores_mem = self.W_mem2hidden(enc_out)
            scores = scores.repeat(1, scores_mem.shape[1], 1) + scores_mem
        scores = torch.tanh(scores)
        scores = self.v(scores).squeeze(2) # (batch_size, max_length)
        scores = scores.masked_fill(mask, -1e20) # (batch_size, max_length)
        attn_weights = F.softmax(scores, dim=1) # (batch_size, max_length)
        attn_weights = attn_weights.unsqueeze(1) # (batch_size, 1,  max_length)
        context = torch.matmul(attn_weights, enc_out).squeeze(1) # (batch_size, hidden_size)
 
        logits = torch.tanh(self.w_global(torch.cat([context, out, h_parent], dim=1)))
        if not self.tie_weights:
            logits = self.w_out(logits)
        else:
            logits = torch.mm(logits, self.embeddingT.weight.T)
        if self.pointer:
            w_t = F.log_softmax(logits, dim=1)
            attn_weights = F.log_softmax(scores, dim=1)
            w_s = self.w_switcher(torch.cat([context, out], dim=1))
            return torch.cat([self.logsigmoid(w_s) + w_t, self.logsigmoid(-w_s) + attn_weights], dim=1), (h, c)
        else:
            w_t = F.log_softmax(logits, dim=1)
            return w_t, (h, c)
        
class DecoderSimpleAn(nn.Module):
    def __init__(
        self,
        anonym_type,
        hidden_size,
        vocab_sizeT_in,
        vocab_sizeT_out,
        vocab_sizeN,
        embedding_sizeN,
        embedding_sizeT,
        dropout,
        static_x_ids=[],
        attn_size=50,
        attn=False,
        pointer=False,
        embedding_sizeT_dyn=None
    ):
        super(DecoderSimpleAn, self).__init__()
        self.num_layers = 1
        self.attn = attn
        self.pointer = pointer
        self.dropout = dropout
        
        if embedding_sizeT_dyn is None:
            self.embedding_sizeT_dyn = embedding_sizeT
            self.use_dyn_emb_transform = False
        else:
            self.embedding_sizeT_dyn = embedding_sizeT_dyn
            self.use_dyn_emb_transform = True
        
        self.embeddingN = nn.Embedding(vocab_sizeN, embedding_sizeN)
        
        self.embedding_sizeT = embedding_sizeT
        self.static_x_ids = torch.from_numpy(np.array(\
                                    sorted(static_x_ids)))
        self.dynamic_x_ids = torch.from_numpy(np.array(\
                        sorted(list(set(range(vocab_sizeT_in)).\
                                 difference(set(static_x_ids))))))
        self.static2dense = torch.zeros(max(self.static_x_ids)+1).long()
        self.static2dense[self.static_x_ids] =\
                       torch.arange(len(self.static_x_ids))
        self.dynamic2dense = torch.zeros(max(self.dynamic_x_ids)+1).long()
        self.dynamic2dense[self.dynamic_x_ids] =\
                       torch.arange(len(self.dynamic_x_ids))
        self.vocab_size_static = len(static_x_ids)
        self.vocab_size_dyn = vocab_sizeT_in - len(self.static_x_ids)
        self.vocab_sizeT_in = vocab_sizeT_in
        self.vocab_sizeT_out = vocab_sizeT_out
        
        self.hidden_size = hidden_size
        if self.vocab_size_static > 0:
            self.embeddingT = nn.Embedding(self.vocab_size_static, \
                                           embedding_sizeT)
        self.lstm = nn.LSTM(
            embedding_sizeN + embedding_sizeT,
            self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_dyn = nn.LSTM(
            embedding_sizeN + self.hidden_size,
            self.embedding_sizeT_dyn,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.w_out = nn.Linear(self.hidden_size * (2+int(self.attn)), embedding_sizeT)
        
        if len(self.static_x_ids) > 0:
            self.w_global = nn.Linear(self.hidden_size, \
                                  len(self.static_x_ids))
            
        self.anonym_type = anonym_type
        self.h_init_type = "zeros" if anonym_type in {"mixed_zeros",\
                                                      "pure_zeros"}\
                                   else "uni"
        if anonym_type in {"mixed_uni", "mixed_zeros"}:
            self.h_type = "mixed"
            self.dyn_inits_h = nn.Parameter(torch.zeros((\
                                        len(self.dynamic_x_ids),\
                                                   1, self.embedding_sizeT_dyn), \
                                        dtype=torch.float))
            self.dyn_inits_c = nn.Parameter(torch.zeros((\
                                        len(self.dynamic_x_ids),\
                                                   1, self.embedding_sizeT_dyn), \
                                        dtype=torch.float))
            if anonym_type == "mixed_uni":
                self.dyn_inits_h.data.uniform_(-0.05, 0.05)
                self.dyn_inits_c.data.uniform_(-0.05, 0.05)
        else: # pure_uni, pure_zeros
            self.h_type = "pure"
            self.dyn_inits_h = nn.Parameter(torch.zeros((1, self.embedding_sizeT_dyn), \
                                        dtype=torch.float))
            self.dyn_inits_c = nn.Parameter(torch.zeros((1, self.embedding_sizeT_dyn), \
                                        dtype=torch.float))
            if anonym_type == "pure_uni":
                self.dyn_inits_h.data.uniform_(-0.05, 0.05)
                self.dyn_inits_c.data.uniform_(-0.05, 0.05)
        
        if self.attn:
            self.W_hidden = nn.Linear(hidden_size, hidden_size)
            self.W_mem2hidden = nn.Linear(hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, 1)

            self.W_context = nn.Linear(
                embedding_sizeN + (embedding_sizeT) + \
                hidden_size,
                hidden_size
            )
        if self.pointer:
            assert self.attn
            self.w_switcher = nn.Linear(embedding_sizeT, 1)
            self.logsigmoid = torch.nn.LogSigmoid()
        
        if self.use_dyn_emb_transform:
            self.emb_transform = nn.Linear(self.embedding_sizeT_dyn, embedding_sizeT)
        else:
            self.emb_transform = lambda x: x
        
    def embedded_dropout(self, embed, words, scale=None):
        dropout = self.dropout
        if dropout > 0:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = F.embedding(words, masked_embed_weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )
        return X

    def forward(
        self,
        input, # (batch_size,)
        hc, # 2 x (batch_size, hidden_size)
        enc_out, 
        mask, 
        h_parent, # batch_size, hidden_size
        dyn_embs, # list of dics
    ):
        n_input, t_input = input
        batch_size = n_input.size(0)
        st = n_input
        
        lookup_table, de_h_tensor, de_c_tensor = dyn_embs
        # lookup_table: batch_size, vocab_size_dyn,
        #               stores index in de_tensor or -1
        # de_tensors: (num of indeces in lookup_table), 1, hidden_size
        
        # hidden_size = (vocab_size+1) mult hidden_size_basic
        (h, c) = hc
        
        is_static = ((t_input[:, None]==self.static_x_ids[None, :]).\
                                           sum(dim=1) > 0).detach()
        is_dynamic = ~ is_static
        any_dynamic = is_dynamic.sum() > 0
        loktab_sel = lookup_table[torch.arange(batch_size).\
                                      to(st.device)[is_dynamic],\
                           self.dynamic2dense[t_input[is_dynamic]]]
        is_dynamic_new = st.new_zeros(is_dynamic.shape).bool()
        is_dynamic_new[is_dynamic] = loktab_sel == -1
        is_dynamic_update = st.new_zeros(is_dynamic.shape).bool()
        is_dynamic_update[is_dynamic] = loktab_sel != -1
        inds_update = loktab_sel[loktab_sel != -1]
        n_input = self.embedded_dropout(self.embeddingN, n_input)
        
        # update global hidden state
        h_tensor = st.new_zeros((1, batch_size, \
                                       self.embedding_sizeT)).float() # hidden_size
        c_dynamic = st.new_zeros((1, int(is_dynamic.sum().detach()), \
                                       self.embedding_sizeT_dyn)).float() # hidden_size
        h_dynamic = st.new_zeros((1, int(is_dynamic.sum().detach()), \
                                       self.embedding_sizeT_dyn)).float() # hidden_size
        if is_static.sum() > 0:
            h_tensor[:, is_static] = self.embedded_dropout(\
                                             self.embeddingT, \
                         self.static2dense[t_input[is_static]])
        if is_dynamic_update.sum() > 0:
            h_tensor[:, is_dynamic_update] = self.emb_transform(\
                                                     de_h_tensor[inds_update]).\
                                                      transpose(0, 1)
            c_dynamic[:, is_dynamic_update[is_dynamic]] = de_c_tensor[inds_update].\
                                                      transpose(0, 1)
            h_dynamic[:, is_dynamic_update[is_dynamic]] = de_h_tensor[inds_update].\
                                                      transpose(0, 1)
        if is_dynamic_new.sum() > 0:
            if self.h_type == "pure":
                h_tensor[:, is_dynamic_new] = self.emb_transform(\
                                              self.dyn_inits_h[:, None, :])
                c_dynamic[:, is_dynamic_new[is_dynamic]] = \
                                              self.dyn_inits_c[:, None, :]
                h_dynamic[:, is_dynamic_new[is_dynamic]] = \
                                              self.dyn_inits_h[:, None, :]
            else: # "mixed"
                h_tensor[:, is_dynamic_new] = self.emb_transform(\
                                                 self.dyn_inits_h[\
                        self.dynamic2dense[t_input[is_dynamic_new]]]).\
                                                      transpose(0, 1)
                c_dynamic[:, is_dynamic_new[is_dynamic]] = \
                                                 self.dyn_inits_c[\
                        self.dynamic2dense[t_input[is_dynamic_new]]].\
                                                      transpose(0, 1)
                h_dynamic[:, is_dynamic_new[is_dynamic]] = \
                                                 self.dyn_inits_h[\
                        self.dynamic2dense[t_input[is_dynamic_new]]].\
                                                      transpose(0, 1)
        t_input_tensor = h_tensor[0]
        
        input = torch.cat([n_input, t_input_tensor], 1)
        
        out, (h_hid, c_hid) = self.lstm(input.unsqueeze(1), hc)
        out = out.squeeze(1)
        
        if self.attn:
            hidden = h_hid[-1] # use only last layer hidden in attention
            scores = self.W_hidden(hidden).unsqueeze(1) # (batch_size, max_length, hidden_size)
            if enc_out.shape[1] > 0:
                scores_mem = self.W_mem2hidden(enc_out)
                scores = scores.repeat(1, scores_mem.shape[1], 1) + scores_mem
            scores = torch.tanh(scores)
            scores = self.v(scores).squeeze(2) # (batch_size, max_length)
            scores = scores.masked_fill(mask, -1e20) # (batch_size, max_length)
            attn_weights = F.softmax(scores, dim=1) # (batch_size, max_length)
            attn_weights = attn_weights.unsqueeze(1) # (batch_size, 1,  max_length)
            context = torch.matmul(attn_weights, enc_out).squeeze(1) # (batch_size, hidden_size)
            out = torch.tanh(self.w_out(torch.cat([context, out, h_parent], dim=1)))
        else:
            out = torch.tanh(self.w_out(torch.cat([out, h_parent], dim=1)))
        
        # update dynamic embeddings
        if any_dynamic:
            dyn_input = torch.cat([n_input[is_dynamic],\
                                   h[-1, is_dynamic]], dim=1)
            hc_dyn = [h_dynamic, c_dynamic]
            _, (h_dyn, c_dyn) = self.lstm_dyn(dyn_input.unsqueeze(1), \
                                              hc_dyn)
            de_h_tensor = de_h_tensor + 0
            de_c_tensor = de_c_tensor + 0
            de_h_tensor[inds_update] = h_dyn\
                 [:, is_dynamic_update[is_dynamic]].transpose(0, 1)
            de_c_tensor[inds_update] = c_dyn\
                 [:, is_dynamic_update[is_dynamic]].transpose(0, 1)
            lookup_table[torch.arange(batch_size).\
                                          to(st.device)[is_dynamic_new],\
                         self.dynamic2dense[t_input[is_dynamic_new]]] = \
                             torch.arange(de_h_tensor.shape[0],\
                                          de_h_tensor.shape[0]+\
                                          int(is_dynamic_new.sum().\
                                          detach())).to(st.device)
            
            de_h_tensor = torch.cat([de_h_tensor,\
              h_dyn[:, is_dynamic_new[is_dynamic]].transpose(0, 1)],\
                                    dim=0)
            de_c_tensor = torch.cat([de_c_tensor,\
              c_dyn[:, is_dynamic_new[is_dynamic]].transpose(0, 1)],\
                                    dim=0)
        
        # output with "attention" over dynamic embeddings
        if self.vocab_sizeT_out > self.vocab_size_static:
            w_t = st.new_zeros((batch_size, self.vocab_sizeT_out)).float()
            w_t[:, self.static_x_ids] = torch.mm(out, self.embeddingT.weight.T)
            if self.h_init_type != "zeros":
                if self.h_type == "mixed":
                    w_t_dynamic = torch.mm(out, self.emb_transform(self.dyn_inits_h[:, -1]).T)
                    # batch, voc_dynamic
                else:
                    w_t_dynamic = torch.mv(out, \
                               self.emb_transform(self.dyn_inits_h[-1]))[:, None].\
                                                   repeat(1, self.vocab_size_dyn)
            else:
                w_t_dynamic = st.new_zeros((batch_size, self.vocab_size_dyn)).float()
            all_inds = lookup_table[lookup_table!=-1]
            objs = torch.arange(batch_size).reshape(-1, 1).\
                   repeat(1, lookup_table.shape[1]).to(st.device)
            objs_inds = objs[lookup_table!=-1]
            w_t_dynamic[lookup_table!=-1] = \
                (out[objs_inds]*self.emb_transform(de_h_tensor[all_inds, -1])).sum(dim=1)
            w_t[:, self.dynamic_x_ids] = w_t_dynamic
        else:
            w_t = torch.mm(out, self.embeddingT.weight.T)
        
        w_t = torch.clamp(w_t, -15, 15)
        
        w_t = F.log_softmax(w_t, dim=1)
            
        if self.pointer:
            attn_weights = F.log_softmax(scores, dim=1)
            w_s = self.w_switcher(out)
            w_t = torch.cat([self.logsigmoid(w_s) + w_t, self.logsigmoid(-w_s) + attn_weights], dim=1)
        
        dyn_embs = [lookup_table, de_h_tensor, de_c_tensor]
        return w_t, (h_hid, c_hid), dyn_embs

class MixtureAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_sizeT_in,
        vocab_sizeT_out,
        vocab_sizeN,
        embedding_sizeT,
        embedding_sizeN,
        dropout,
        attn=True,
        pointer=True,
        train_init = True,
        attn_size=50,
        anonymized=None,
        static_x_ids=[],\
        eof_N_id=0,\
        unk_id_out=0,\
        loss_type=1,
        tie_weights=False,\
        embedding_sizeT_dyn=None,
    ):
        super(MixtureAttention, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.eof_N_id = eof_N_id
        self.unk_id_out = unk_id_out
        self.attn_size = attn_size
        self.vocab_sizeT_in = vocab_sizeT_in
        self.vocab_sizeT_out = vocab_sizeT_out
        self.vocab_sizeN = vocab_sizeN
        
        if anonymized is None:
            if attn:
                self.decoder = DecoderAttention(
                    hidden_size=hidden_size,
                    vocab_sizeT_in=vocab_sizeT_in,
                    vocab_sizeT_out=vocab_sizeT_out,
                    vocab_sizeN=vocab_sizeN,
                    embedding_sizeT=embedding_sizeT,
                    embedding_sizeN=embedding_sizeN,
                    attn_size=attn_size,
                    dropout=dropout,
                    pointer=pointer,
                    tie_weights=tie_weights,
                )
            else:
                self.decoder = DecoderSimple(
                    hidden_size=hidden_size,
                    vocab_sizeT_in=vocab_sizeT_in,
                    vocab_sizeT_out=vocab_sizeT_out,
                    vocab_sizeN=vocab_sizeN,
                    embedding_sizeT=embedding_sizeT,
                    embedding_sizeN=embedding_sizeN,
                    dropout=dropout,
                    tie_weights=tie_weights,
                )
        else: # anonymized
            self.decoder = DecoderSimpleAn(
                    anonymized,
                    hidden_size=hidden_size,
                    vocab_sizeT_in=vocab_sizeT_in,
                    vocab_sizeT_out=vocab_sizeT_out,
                    vocab_sizeN=vocab_sizeN,
                    embedding_sizeT=embedding_sizeT,
                    embedding_sizeN=embedding_sizeN,
                    static_x_ids=static_x_ids,
                    dropout=dropout,
                    attn=attn,
                    pointer=pointer,
                    attn_size=attn_size,\
                    embedding_sizeT_dyn=embedding_sizeT_dyn
                ) 
        self.anonymized = (anonymized is not None)
        self.criterion_universal = utils.VocPointerLoss(loss_type, \
                                                        vocab_sizeT_out, unk_id_out)
        self.train_init = train_init
        if train_init:
            self.hid_init = nn.Parameter(torch.zeros((1, hidden_size), \
                                                     dtype=torch.float))
            self.cell_init = nn.Parameter(torch.zeros((1, hidden_size), \
                                                     dtype=torch.float))
        else:
            self.hid_init = torch.zeros((1, hidden_size), dtype=torch.float,\
                                       requires_grad=False).to("cuda")
            self.cell_init = torch.zeros((1, hidden_size), dtype=torch.float,\
                                       requires_grad=False).to("cuda")
                
        self.pointer = pointer
        self.some_tensor = self.decoder.embeddingN.weight
        
    def forward(
        self,
        n_tensor,
        t_voc_tensor_in,
        t_voc_tensor_out,
        t_attn_tensor,
        p_tensor,
        n_tensor_prev,
        t_voc_tensor_prev,
        p_tensor_prev,
        hs_prev,
        hc_prev,
        dyn_embs=None
    ):
        batch_size = n_tensor.size(0)
        max_length = n_tensor.size(1) 
        prev_max_length = n_tensor_prev.size(1)
        # suppose max_length == self.attn_size

        full_mask_prev = (n_tensor_prev == self.eof_N_id)
        inv_idx = torch.arange(full_mask_prev.size(1)-1, -1, -1).long()
        full_mask_prev = (torch.cumsum(full_mask_prev[:, inv_idx], dim=1)\
                          [:, inv_idx] > 0)
        full_mask = self.some_tensor.new_zeros(n_tensor.shape, dtype=bool)
        
        hs = self.some_tensor.new_zeros(
            batch_size,
            max_length,
            self.hidden_size,
            requires_grad=False
        )

        loss = self.some_tensor.new_tensor(0.0)

        ans = []
        
        parent_hs = self.some_tensor.new_zeros((batch_size, \
                                                self.hidden_size))
        
        hc = [hc_prev[0].clone().detach(), \
              hc_prev[1].clone().detach()]
        # 2 x 1, batch_size, hidden_size
        if self.anonymized:
            dyn_embs = [tensor.clone().detach() for tensor in dyn_embs]

        # filter hs for last beginning
        for iter in range(max_length): # what we now predict
            # reset h0 and c0 for starting seqs
            if iter > 0:
                input = (n_tensor[:, iter-1].clone().detach(), \
                         t_voc_tensor_in[:, iter-1].clone().detach())
                parent = p_tensor[:, iter-1]
                mask_start_ = (n_tensor[:, iter-1]==self.eof_N_id)
                mask_start = mask_start_[None, :, None].float() 
            else:
                input = (n_tensor_prev[:, -1].clone().detach(), \
                         t_voc_tensor_prev[:, -1].clone().detach())
                parent = p_tensor_prev[:, -1]
                mask_start_ = (n_tensor_prev[:, -1]==self.eof_N_id)
                mask_start = mask_start_[None, :, None].float() 
            newhc = []
            hid_init, cell_init = self.hid_init, self.cell_init
            newhc.append(hc[0] * (1-mask_start) + \
                    hid_init[:, None, :] * mask_start)
            # hid_init 1, 1, 1500, mask 1, 128, 1
            newhc.append(hc[1] * (1-mask_start) + \
                    cell_init[:, None, :] * mask_start)
            hc = newhc
            if self.anonymized:
                if mask_start_.any():
                    lookup_table, de_h_tensor, de_c_tensor = dyn_embs
                    remaining_lookup_table = lookup_table[~mask_start_]
                    remain_inds = remaining_lookup_table[remaining_lookup_table!=-1]
                    de_h_tensor_new = de_h_tensor[remain_inds]
                    de_c_tensor_new = de_c_tensor[remain_inds]
                    remaining_lookup_table[remaining_lookup_table!=-1] = \
                                      torch.arange(remain_inds.shape[0]).\
                                      to(lookup_table.device)
                    lookup_table_new = lookup_table.new_zeros(lookup_table.shape).long() - 1
                    lookup_table_new[~mask_start_] = remaining_lookup_table
                    dyn_embs = [lookup_table_new, de_h_tensor_new, de_c_tensor_new]
            prev_len = self.attn_size - iter
            cur_len = iter
            # hs are parallel with prediction
            memory = torch.cat([hs_prev[:, -prev_len:],\
                               hs[:, :cur_len]], dim=1)
            mask = torch.cat([full_mask_prev[:, -prev_len:],\
                              full_mask[:, :cur_len]], dim=1)
            inv_idx = torch.arange(memory.shape[1]-1, -1, -1).long()
            memory = memory[:, inv_idx]
            mask = mask[:, inv_idx]
            mp = (parent < 0) & (parent >= -(iter-1))
            parent_hs[mp] = \
                   hs[torch.arange(batch_size)[mp],\
                      parent[mp]+iter-1].clone().detach()
            mp = (parent<-(iter-1)) & (parent>=-(iter-1)-prev_max_length)
            parent_hs[mp] = \
                   hs_prev[torch.arange(batch_size)[mp],\
                           parent[mp]+iter-1].clone().detach()
            if not self.anonymized:
                output, hc = self.decoder(
                    input,
                    hc, # 2 x batch_size, hidden_size
                    memory.clone().detach(),
                    mask.clone().detach(),
                    parent_hs.clone().detach()
                )
            else:
                output, hc, dyn_embs = self.decoder(
                    input,
                    hc, # 2 x batch_size, hidden_size
                    memory.clone().detach(),
                    mask.clone().detach(),
                    parent_hs.clone().detach(),
                    dyn_embs,
                )
            hs[:, iter] = hc[0][-1] # store last layer hidden state only
            # start new seqs for the next timestep
            full_mask[n_tensor[:, iter]==self.eof_N_id, :iter+1] = 1
            full_mask_prev[n_tensor[:, iter]==self.eof_N_id, :] = 1
            topv, topi = output.topk(1)

            ans.append(topi.detach())
            loss = loss + \
                   self.criterion_universal(output, \
                                          t_voc_tensor_out[:, iter].clone().detach(), \
                                          t_attn_tensor[:, iter].clone().detach())
        if not self.anonymized:
            return loss, torch.cat(ans, dim=1), hs, hc
        else:
            return loss, torch.cat(ans, dim=1), hs, hc, dyn_embs