import torch
from inputters.constants import PAD

def pad_subtokens(seq, pad_symb):
    """
    batch is a list of lists (seq - subtokens)
    """
    max_len = max([len(elem) for elem in seq])
    seq = [elem+[pad_symb]*(max_len-len(elem)) for elem in seq]
    return seq

def vectorize(ex, model):
    """Vectorize a single example."""
    src_dict = model.src_dict
    code = ex['code']
    type_dict = model.type_dict
    
    vectorized_ex = dict()
    vectorized_ex['id'] = code.id

    vectorized_ex['code'] = code.text
    vectorized_ex['code_tokens'] = code.tokens
    vectorized_ex['code_type_rep'] = None
    
    code_vectorized = code.vectorize(word_dict=src_dict,\
                                     attrname="tokens")
    vectorized_ex['code_word_rep'] = torch.LongTensor(code_vectorized)
    vectorized_ex['code_type_rep'] = torch.LongTensor(code.vectorize(word_dict=type_dict, attrname="type"))

    vectorized_ex['src_vocab'] = code.src_vocab

    return vectorized_ex


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    
    ids = [ex['id'] for ex in batch]

    # --------- Prepare Code tensors ---------
    code_words = [ex['code_word_rep'] for ex in batch]
    code_type = [ex['code_type_rep'] for ex in batch]
    max_code_len = max([d.size(0) for d in code_words])

    # Batch Code Representations
    code_len_rep = torch.zeros(batch_size, dtype=torch.long)
    code_word_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long)
    code_type_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long)
    
    for i in range(batch_size):
        code_len_rep[i] = code_words[i].size(0)
        code_word_rep[i, :code_words[i].size(0)].copy_(code_words[i])
        code_type_rep[i, :code_type[i].size(0)].copy_(code_type[i])
    
    return {
        'ids': ids,
        'batch_size': batch_size,
        'code_word_rep': code_word_rep,
        'code_type_rep': code_type_rep,
        'code_len': code_len_rep,
        'code_text': [ex['code'] for ex in batch],
        'code_tokens': [ex['code_tokens'] for ex in batch],
    }
