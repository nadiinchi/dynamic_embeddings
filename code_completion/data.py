import numpy as np
import pickle
import torch

def input_data(N_filename, T_filename):
    with open(N_filename, 'rb') as f:
        print ("reading data from ", N_filename)
        save = pickle.load(f)
        train_dataN = save['trainData']
        test_dataN = save['testData']
        train_dataP = save['trainParent']
        test_dataP = save['testParent']
        vocab_sizeN = save['vocab_size']

    with open(T_filename, 'rb') as f:
        print ("reading data from ", T_filename)
        save = pickle.load(f)
        train_dataT = (save['trainData_token'], save['trainData_attn'])
        test_dataT = (save['testData_token'], save['testData_attn'])
        attn_size = save['attn_size']

    return train_dataN, test_dataN, vocab_sizeN, train_dataT, test_dataT, attn_size, train_dataP, test_dataP

class DataLoader(object):
    """
    DataLoader stores all the data (without vocabulary filtering)
    as a chain of elements separated with EOF
    shaped as [batch_size, len] where len = len(chain) // batch_size
    To generate a batch, DataLoader chooses tensors[:, i:i+truncate_size]
    You can optionally choose to reduce the amount of data by setting sieve_step (selects only sieve_step-th data examples)
    """
    def __init__(self,
            N_filename,
            T_filename,
            is_train,
            truncate_size,
            batch_size,
            sieve_step=0):
        
        train_dataN, test_dataN, vocab_sizeN, train_dataT, \
        test_dataT, attn_size, train_dataP, test_dataP = \
        input_data(
                        N_filename, T_filename
                    )
        if is_train:
            dataN, dataT, dataP = train_dataN, train_dataT, train_dataP
        else:
            dataN, dataT, dataP = test_dataN, test_dataT, test_dataP
        if sieve_step > 0:
            dataN = dataN[::sieve_step]
            dataT_ = []
            dataT_.append(dataT[0][::sieve_step])
            dataT_.append(dataT[1][::sieve_step])
            dataT = dataT_
            dataP = dataP[::sieve_step]
        self.is_train = is_train
        self.eof_idxN = vocab_sizeN
        self.vocab_sizeN = vocab_sizeN + 1
        
        total_len = sum([len(seq)+1 for seq in dataN])
        split_len = total_len // batch_size
        cropped_len = split_len * batch_size
        def split_seqs(seqs, eof_idx):
            long_seq = [symb for seq in seqs for symb in seq+[eof_idx]]\
                       [:cropped_len]
            res = []
            for i in range(batch_size):
                res.append(long_seq[i*split_len:(i+1)*split_len])
            return res
        self.split_N = np.array(split_seqs(dataN, self.eof_idxN))
        dataP = [[-p if i>0 else -10**5 for i, p in enumerate(seqP)] for seqP in dataP]
        self.split_P = np.array(split_seqs(dataP, -10**5))
        
        self.split_T_raw = np.array(split_seqs(dataT[0], -55))
        self.split_T_attn = np.array(split_seqs(dataT[1], -55))
        self.fix_seqsT()
        
        self.num_examples = len(self.split_T_raw[0])
        self.pointer = np.arange(self.num_examples)
        self.batch_size = batch_size
        self.truncate_size = truncate_size
        self.attn_size = attn_size
        
        self.reset()
         
    def fix_seqsT(self):
        for idx in range(self.split_T_attn.shape[0]):
            first_eof = np.where(self.split_T_attn[idx]==-55)[0][0]
            l = self.split_T_attn[idx, :first_eof]
            l[l>=np.arange(1, first_eof+1)] = -1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= self.num_examples:
            self.reset()
            raise StopIteration()
        
        batch_N = torch.tensor(self.split_N[:, self.index: \
                                        self.index + self.truncate_size],\
                               dtype=torch.long)
        batch_T_raw = torch.tensor(self.split_T_raw[:, self.index: \
                                        self.index + self.truncate_size],\
                               dtype=torch.long)
        batch_T_attn = torch.tensor(self.split_T_attn[:, self.index: \
                                        self.index + self.truncate_size],\
                               dtype=torch.long)
        batch_P = torch.tensor(self.split_P[:, self.index: \
                                        self.index + self.truncate_size],\
                               dtype=torch.long)
        
        # update index
        self.index += self.truncate_size

        return batch_N, batch_T_raw, batch_T_attn, batch_P
    
    def reset(self):
        self.index = 0
        
    def __len__(self):
        return (self.num_examples // self.truncate_size) + \
                ((self.num_examples % self.truncate_size) > 0)