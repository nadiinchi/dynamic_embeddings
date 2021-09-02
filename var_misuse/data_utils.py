import numpy as np
from torch.utils.data import Dataset
from inputters.vector import vectorize
from inputters.vector import batchify
from inputters.utils import process_examples
from collections import Counter
from tqdm import tqdm
import multiprocessing as mp

import torch
import json

def read_lengths(tgt_filename, offset=0):
    num_exs = []
    with open(tgt_filename) as f:
        for line in f:
            num_exs.append(len(line.strip().split())+offset)
    return num_exs
    
class FileReader:
    def __init__(self, filename):
        self.fin = open(filename, "r")
        self.line_map = list() # Map from line index -> file position.
        self.line_map.append(0)
        while self.fin.readline():
            self.line_map.append(self.fin.tell())

    def get_line(self, index):
        self.fin.seek(self.line_map[index])
        return self.fin.readline()    
    
class VarmisuseDataset(Dataset):
    def __init__(self, model, args, filenames):
        self.model = model
        self.filenames = filenames
        self.args = args
        self.num_exs = read_lengths(filenames["tgt"], offset=-1)
        if self.args.short_dataset:
            self.num_exs = [1 for _ in self.num_exs]
        self._lengths = read_lengths(filenames["src"], offset=0)
        self.code_idxs, self.target_idxs = [], []
        for i, num_ex in enumerate(self.num_exs):
            self.target_idxs += list(range(2*num_ex))
            self.code_idxs += [i] * (2*num_ex)
            # 2 means that for each buggy example read from file, a non-buggy example is added
        self.fds = []
        self.locks = []
        self.num_fds = 0
        self.global_lock = mp.Lock()

    def __len__(self):
        return len(self.target_idxs)
    
    def get_fd(self):
        """
        file descriptors are needed to provide parallel access to data files
        when reading data lines on the fly (=during batch contruction)
        """
        res = -1
        with self.global_lock:
            for i in range(self.num_fds):
                if not self.locks[i]:
                    res = i
                    break
            if res == -1:
                self.locks.append(False)
                self.fds.append({key:(FileReader(self.filenames[key]) \
                               if self.filenames[key] is not None \
                               else None) for key in self.filenames})
                res = self.num_fds
                self.num_fds += 1
            self.locks[res] = True
        return res

    def __getitem__(self, index):
        num_fd = self.get_fd()
        fd = self.fds[num_fd]
        ci = self.code_idxs[index]
        ex = {}
        # read ci-th line from all input data files
        for key in fd:
            if fd[key] is not None:
                f = fd[key]
                line = f.get_line(ci).strip()
                if key == "rel_matrix":
                    line = json.loads(line)
                ex[key] = line
            else:
                ex[key] = None
        with self.global_lock:
            self.locks[num_fd] = False
        # preprocess function data
        ex_obj = process_examples(ex["src"],
                                   ex["src_tag"],
                                   self.args.max_src_len,
                                   self.args.max_tgt_len,
                                   uncase=False)
        # preprocess target data
        ti = self.target_idxs[index]
        elems = ex["tgt"].split()
        if not self.args.short_dataset:
            target_criteria = ti < len(elems) - 1
        else:
            target_criteria = ti < 1
        if target_criteria:
            # buggy example
            pos_bug_fixes = elems[ti+1].split("_")
            target_pos = int(pos_bug_fixes[0])
            target_bug = int(pos_bug_fixes[1])
            target_fixes = [int(pos) for pos in pos_bug_fixes[2].split("|")]
        else:
            # non-buggy example
            target_pos = -1
            target_bug = -1
            target_fixes = []
        if target_pos != -1: # inject bug
            ex_obj["code"].tokens[target_pos] = ex_obj["code"].tokens[target_bug]
            if len(ex_obj["code"].subtokens):
                ex_obj["code"].subtokens[target_pos] = \
                              ex_obj["code"].subtokens[target_bug]
        
        vector = vectorize(ex_obj, self.model)
        vector["scope"] = torch.LongTensor([int(pos) for pos in elems[0].split("_")])
        vector["target_pos"] = torch.LongTensor([target_pos])
        vector["target_bug"] = torch.LongTensor([target_bug])
        vector["target_fixes"] = torch.LongTensor(target_fixes)
        return vector
        
    def lengths(self):
        return [(self._lengths[self.code_idxs[index]], 1)
                for index in range(len(self.code_idxs))]

def batchify_varmisuse(list_of_vectors):
    batch = batchify(list_of_vectors)
    batch_size = len(list_of_vectors)
    pos_t = torch.cat([v["target_pos"] for v in list_of_vectors])
    bug_t = torch.cat([v["target_bug"] for v in list_of_vectors])
    fixes_t = torch.zeros(batch["code_word_rep"].shape[:2]).float()
    scope_t = torch.zeros(batch["code_word_rep"].shape[:2]).bool()
    for i, v in enumerate(list_of_vectors):
        fixes_t[i][v["target_fixes"]] = 1
        scope_t[i][v["scope"]] = 1
    arange = torch.arange(batch_size)
    mask_incorrect = pos_t != -1
    target_pos = pos_t*mask_incorrect.long() + (-1)*(1-mask_incorrect.long())
    batch["mask_incorrect"] = mask_incorrect
    batch["target_pos"] = target_pos
    batch["target_fix"] = fixes_t
    batch["fixes_t"] = fixes_t
    batch["scope_t"] = scope_t
    return batch

