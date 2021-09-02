import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from inputters.constants import PAD
    
class EncoderRNNSimple(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_sizeT,
        vocab_sizeN,
        embedding_sizeT,
        embedding_sizeN,
        dropout,
        num_layers,
        bidirectional=False
    ):
        super(EncoderRNNSimple, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.embeddingN = nn.Embedding(vocab_sizeN, embedding_sizeN)
        self.embeddingT = nn.Embedding(vocab_sizeT, embedding_sizeT)
        
        
        self.lstm = nn.LSTM(
             embedding_sizeN + embedding_sizeT,
             hidden_size,  
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.hid_init = nn.Parameter(torch.zeros((hidden_size,), \
                                                     dtype=torch.float))
        self.cell_init = nn.Parameter(torch.zeros((hidden_size,), \
                                                     dtype=torch.float))
        
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.transform = nn.Linear(2*hidden_size, hidden_size)
        
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
    ):
        n_input_all, t_input_all = input # 2 x batch, seq
        batch_size = n_input_all.size(0)
        if not self.bidirectional:
            (h, c) = (self.hid_init[None, :].repeat((batch_size, 1)),\
                  self.cell_init[None, :].repeat((batch_size, 1)))
            h = h[None, :, :]
            c = c[None,:, :]
        else:
            (h, c) = (self.hid_init[None, None, :].repeat((2, batch_size, 1)),\
                  self.cell_init[None, None, :].repeat((2, batch_size, 1)))
        n_input_all = self.embeddingN(n_input_all)
        t_input_all =  self.embeddingT(t_input_all)
        input_all = torch.cat([n_input_all, t_input_all], 2)
        
        out, _ = self.lstm(input_all, (h, c))
        
        if self.bidirectional:
            out = self.transform(out)
        
        return out


class DynamicRNNEncoder(nn.Module):
    def __init__(
        self,
        anonym_type,
        hidden_size,
        vocab_sizeT,
        vocab_sizeN,
        embedding_sizeN,
        embedding_sizeT,
        dropout,
        num_layers,
        static_x_ids=[],
        reverse=False,
        embedding_sizeT_dyn=None
    ):
        super(DynamicRNNEncoder, self).__init__()
        self.num_layers = num_layers
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
                        sorted(list(set(range(vocab_sizeT)).\
                                 difference(set(static_x_ids))))))
        self.static2dense = torch.zeros(max(self.static_x_ids)+1).long()
        self.static2dense[self.static_x_ids] =\
                       torch.arange(len(self.static_x_ids))
        self.dynamic2dense = torch.zeros(max(self.dynamic_x_ids)+1).long()
        self.dynamic2dense[self.dynamic_x_ids] =\
                       torch.arange(len(self.dynamic_x_ids))
        self.vocab_size_static = len(static_x_ids)
        self.vocab_size_dyn = vocab_sizeT - len(self.static_x_ids)
        self.vocab_sizeT = vocab_sizeT
        
        self.hidden_size = hidden_size
        if self.vocab_size_static > 0:
            self.embeddingT = nn.Embedding(self.vocab_size_static, \
                                           embedding_sizeT) # self.hidden_size
        self.lstm = nn.LSTMCell(
            embedding_sizeN + embedding_sizeT,
            self.hidden_size,
        
        )
        self.lstm_dyn = nn.LSTMCell(
            embedding_sizeN + self.hidden_size,
            self.embedding_sizeT_dyn,
            
        )
        
        self.hid_init = nn.Parameter(torch.zeros((hidden_size,), \
                                                     dtype=torch.float))
        self.cell_init = nn.Parameter(torch.zeros((hidden_size,), \
                                                     dtype=torch.float))
        
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
            self.dyn_inits_h = nn.Parameter(torch.zeros((1, \
                                                         self.embedding_sizeT_dyn), \
                                        dtype=torch.float))
            self.dyn_inits_c = nn.Parameter(torch.zeros((1, \
                                                         self.embedding_sizeT_dyn), \
                                        dtype=torch.float))
            if anonym_type == "pure_uni":
                self.dyn_inits_h.data.uniform_(-0.05, 0.05)
                self.dyn_inits_c.data.uniform_(-0.05, 0.05)
                
        self.reverse = reverse
        
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
        input,
    ):
        n_input_all, t_input_all = input # 2 x batch, seq
        batch_size = n_input_all.size(0)
        (h, c) = (self.hid_init[None, :].repeat((batch_size, 1)),\
              self.cell_init[None, :].repeat((batch_size, 1)))
        st = n_input_all
        dyn_embs = [st.new_zeros((batch_size, \
                        self.vocab_size_dyn)).long()-1,\
                    st.new_zeros((0, self.embedding_sizeT_dyn)).float(),\
                    st.new_zeros((0, self.embedding_sizeT_dyn)).float()]
        
        out = []
        if self.reverse:
            mask_pad = (t_input_all[:, 0]!=PAD)[:, None]
        for ts in range(n_input_all.shape[1]):
            n_input = n_input_all[:, ts]
            t_input = t_input_all[:, ts]

            lookup_table, de_h_tensor, de_c_tensor = dyn_embs
            # lookup_table: batch_size, vocab_size_dyn,
            #               stores index in de_tensor or -1
            # de_tensors: (num of indeces in lookup_table), hidden_size

            # hidden_size = (vocab_size+1) mult hidden_size_basic

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
            h_tensor = st.new_zeros((batch_size, \
                                       self.embedding_sizeT)).float() # hidden_size
            c_dynamic = st.new_zeros((int(is_dynamic.sum().detach()), \
                                           self.embedding_sizeT_dyn)).float() # hidden_size
            h_dynamic = st.new_zeros((int(is_dynamic.sum().detach()), \
                                           self.embedding_sizeT_dyn)).float() # hidden_size
            if is_static.sum() > 0:
                h_tensor[is_static] = self.embedded_dropout(\
                                                 self.embeddingT, \
                             self.static2dense[t_input[is_static]])
            if is_dynamic_update.sum() > 0:
                h_tensor[is_dynamic_update] = self.emb_transform(\
                                                     de_h_tensor[inds_update])
                c_dynamic[is_dynamic_update[is_dynamic]] = \
                                               de_c_tensor[inds_update]
                h_dynamic[is_dynamic_update[is_dynamic]] = \
                                               de_h_tensor[inds_update]
            if is_dynamic_new.sum() > 0:
                if self.h_type == "pure":
                    h_tensor[is_dynamic_new] = self.dyn_inits_h[None, :]
                    c_dynamic[is_dynamic_new[is_dynamic]] = \
                                                         self.dyn_inits_c[None, :]
                    h_dynamic[is_dynamic_new[is_dynamic]] = \
                                                         self.dyn_inits_h[None, :]
                else: # "mixed"
                    h_tensor[is_dynamic_new] = self.dyn_inits_h[\
                            self.dynamic2dense[t_input[is_dynamic_new]]].\
                                                          transpose(0, 1)
                    c_dynamic[is_dynamic_new[is_dynamic]] = \
                                                     self.dyn_inits_c[\
                            self.dynamic2dense[t_input[is_dynamic_new]]].\
                                                          transpose(0, 1)
                    h_dynamic[is_dynamic_new[is_dynamic]] = \
                                                     self.dyn_inits_h[\
                            self.dynamic2dense[t_input[is_dynamic_new]]].\
                                                          transpose(0, 1)
            
            t_input_tensor = h_tensor

            input = torch.cat([n_input, t_input_tensor], 1)

            (h_new, c_new) = self.lstm(input, (h, c))

            # update dynamic embeddings
            if any_dynamic:
                dyn_input = torch.cat([n_input[is_dynamic],\
                                       h[is_dynamic]], dim=1)
                hc_dyn = [h_dynamic, c_dynamic]
                (h_dyn, c_dyn) = self.lstm_dyn(dyn_input, \
                                                  hc_dyn)
                de_h_tensor = de_h_tensor + 0
                de_c_tensor = de_c_tensor + 0
                de_h_tensor[inds_update] = h_dyn\
                     [is_dynamic_update[is_dynamic]]
                de_c_tensor[inds_update] = c_dyn\
                     [is_dynamic_update[is_dynamic]]
                lookup_table[torch.arange(batch_size).\
                                              to(st.device)[is_dynamic_new],\
                             self.dynamic2dense[t_input[is_dynamic_new]]] = \
                                 torch.arange(de_h_tensor.shape[0],\
                                              de_h_tensor.shape[0]+\
                                              int(is_dynamic_new.sum().\
                                              detach())).to(st.device)

                de_h_tensor = torch.cat([de_h_tensor,\
                  h_dyn[is_dynamic_new[is_dynamic]]],\
                                        dim=0)
                de_c_tensor = torch.cat([de_c_tensor,\
                  c_dyn[is_dynamic_new[is_dynamic]]],\
                                        dim=0)
            if self.reverse:
                # in reverse mode, models observes [PAD, PAD, PAD, t_1, ... t_N] and 
                # we hold hid_init while current symbol is PAD
                (h_in, c_in) = (self.hid_init[None, :].repeat((batch_size, 1)),\
                  self.cell_init[None, :].repeat((batch_size, 1)))
                mask_pad = mask_pad | (t_input_all[:, ts]!=PAD)[:, None]
                h_new = h_new*mask_pad.float() + h_in*(1-mask_pad.float())
                c_new = c_new*mask_pad.float() + c_in*(1-mask_pad.float())
            out.append(h_new[:, None, :])
            dyn_embs = [lookup_table, de_h_tensor, de_c_tensor]
            h, c = h_new, c_new
        out = torch.cat(out, dim=1)
        return out
    
    
class DynamicRNNEncoderBidirectional(nn.Module):
    def __init__(
        self,
        anonym_type,
        hidden_size,
        vocab_sizeT,
        vocab_sizeN,
        embedding_sizeN,
        embedding_sizeT,
        dropout,
        num_layers,
        static_x_ids,
        embedding_sizeT_dyn=None
    ):
        super(DynamicRNNEncoderBidirectional, self).__init__()
        self.dyn_rnn_fw = DynamicRNNEncoder(anonym_type,
                                            hidden_size,
                                            vocab_sizeT,
                                            vocab_sizeN,
                                            embedding_sizeN,
                                            embedding_sizeT,
                                            dropout,
                                            num_layers,
                                            static_x_ids,
                                            embedding_sizeT_dyn)
        self.dyn_rnn_bw = DynamicRNNEncoder(anonym_type,
                                            hidden_size,
                                            vocab_sizeT,
                                            vocab_sizeN,
                                            embedding_sizeN,
                                            embedding_sizeT,
                                            dropout,
                                            num_layers,
                                            static_x_ids,
                                            embedding_sizeT_dyn)
        self.transform = nn.Linear(2*hidden_size, hidden_size)
        
    def forward(
        self,
        input,
    ):
        out_fw = self.dyn_rnn_fw(input)
        out_bw = torch.flip(self.dyn_rnn_bw([torch.flip(input[0], [1]),\
                                             torch.flip(input[1], [1])]),\
                            [1])
        out = torch.cat([out_fw, out_bw], dim=2)
        out = self.transform(out)
        return out