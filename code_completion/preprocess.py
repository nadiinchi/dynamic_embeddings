# src: https://github.com/oleges1/code-completion/tree/master/preprocess_utils

import sys
import numpy as np
from six.moves import cPickle as pickle
import json
from collections import Counter, defaultdict, deque
import os
import operator

EMPTY_TOKEN = '<empty>'

### GATHERING DICTIONARY ###

def process_dict(filename, freq_dict, is_js, increase_counts=True):
  with open(filename, encoding='latin-1') as lines:
    print ('Gathering vocab, start processing %s !!!'%(filename))
    line_index = 0
    for line in lines:
      line_index += 1
      if line_index % 10000 == 0:
        print ('Gathering vocab, processing line:', line_index)
      data = json.loads(line)
      if len(data) < 3e4:
        for i, dic in enumerate(data[:-1] if is_js else data):
          if "value" in dic.keys():
            if not dic["value"] in freq_dict:
                freq_dict[dic["value"]] = 0
            if increase_counts:
                freq_dict[dic["value"]] += 1
                # for test set, do not increase counts
          else:
            if increase_counts:
                freq_dict[EMPTY_TOKEN] += 1
            
def get_terminal_dict(freq_dict):
  terminal_dict = dict()
  sorted_freq_dict = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
  for i, (terminal, frequent) in enumerate(sorted_freq_dict):
    terminal_dict[terminal] = i
  return terminal_dict
    
def save_terminal_dict(filename, terminal_dict):
  with open(filename, 'wb') as f:
    save = {'terminal_dict': terminal_dict, 'vocab_size': len(terminal_dict)}
    pickle.dump(save, f, protocol=2)
    
def restore_terminal_dict(filename):
  with open(filename, 'rb') as f:
    save = pickle.load(f)
    terminal_dict = save['terminal_dict']
    vocab_size = save['vocab_size']
    return terminal_dict, vocab_size

### PROCESSING NON-TERMINALS ###

def process_nonterminals(filename, typeDict, is_js):
  with open(filename, encoding='latin-1') as lines:
    print ('Gathering non-terminals, start processing %s'%(filename))
    line_index = 0
    corpus_N = list()
    corpus_parent = list()

    for line in lines:
        line_index += 1
        if line_index % 10000 == 0:
            print ('Gathering non-terminals, processing line: ', line_index)
        data = json.loads(line)
        line_N = list()
        has_sibling = Counter()
        parent_counter = defaultdict(lambda: 1) #default parent is previous one
        parent_list = list()

        if len(data) >= 3e4:
            continue

        for i, dic in enumerate(data[:-1] if is_js else data):
            typeName = dic['type']
            if not typeName in typeDict:
                typeDict[typeName] = len(typeDict)
            base_ID = typeDict[typeName]

            #expand the ID into the range of 4*base_ID, according to whether it has sibling or children. Sibling information is got by the ancestor's children information
            if 'children' in dic.keys():
                    if has_sibling[i]:
                        ID = base_ID * 4 + 3
                    else:
                        ID = base_ID * 4 + 2

                    childs = dic['children']
                    for j in childs:
                        parent_counter[j] = j-i

                    if len(childs) > 1:
                        for j in childs:
                            has_sibling[j] = 1
            else:
                if has_sibling[i]:
                    ID = base_ID * 4 + 1
                else:
                    ID = base_ID * 4

            line_N.append(ID)
            parent_list.append(parent_counter[i])

        corpus_N.append(line_N)
        corpus_parent.append(parent_list)
    return corpus_N, corpus_parent

def map_dense_id(data, dicID):
    result = list()
    for line_id in data:
        line_new_id = list()
        for i in line_id:
            if i in dicID.keys():
                line_new_id.append(dicID[i])
            else:
                dicID[i] = len(dicID)
                line_new_id.append(dicID[i])
        result.append(line_new_id)
    return result


def make_type_dict(typeDict, dicID):
    type_dict = {} # type: sparse id
    reverse_typeDict = {}
    for typeName, baseID in typeDict.items():
        reverse_typeDict[baseID] = typeName
    for ID, dense_ID in dicID.items():
        baseID = ID // 4
        type_dict[dense_ID] = reverse_typeDict[baseID]
    return type_dict


def save_nonterminals(filename, type_dict, trainData, testData, trainParent, testParent):
  with open(filename, 'wb') as f:
    save = {
        'type_dict': type_dict, # id:TypeName
        'vocab_size':len(type_dict),
        'trainData': trainData, # list of list of type ids
        'testData': testData, 
        'trainParent': trainParent, # list of list of shifts to parent
        'testParent': testParent,
    }
    pickle.dump(save, f, protocol=2)
    

### PROCESSING TERMINALS ###

def process_terminals(filename, terminal_dict, attn_size, is_js):
  with open(filename, encoding='latin-1') as lines:
    print ('Processing terminals, start procesing %s'%(filename))
    terminal_corpus = []
    attn_corpus = []
    attn_que = deque(maxlen=attn_size)
    attn_success_total = 0
    attn_fail_total = 0
    length_total = 0
    line_index = 0
    for line in lines:
      line_index += 1
      if line_index % 10000 == 0:
        print ('Processing terminals, processing line:', line_index)
      data = json.loads(line)
      if len(data) < 3e4:
        terminal_line = []
        attn_line = []
        attn_que.clear() # have a new queue for each file
        attn_success_cnt  = 0
        attn_fail_cnt  = 0
        for i, dic in enumerate(data[:-1] if is_js else data):
          if 'value' in dic.keys():
            dic_value = dic['value']
            if dic_value in attn_que:
                location_index = [len(attn_que)-ind for ind, x in enumerate(attn_que) if x==dic_value][-1] # take last occurence
                attn_line.append(location_index)
                attn_success_cnt += 1
            else:
                attn_fail_cnt += 1
                attn_line.append(-1) # not found
            attn_que.append(dic_value) 
            terminal_line.append(terminal_dict[dic_value])
            # now all words should appear in voc
          else:
            terminal_line.append(terminal_dict[EMPTY_TOKEN])
            attn_line.append(-2) # empty value
            attn_que.append(EMPTY_TOKEN)
        terminal_corpus.append(terminal_line)
        attn_corpus.append(attn_line)
        attn_success_total += attn_success_cnt
        attn_fail_total += attn_fail_cnt
        attn_total = attn_success_total + attn_fail_total
        length_total += len(data)
        if line_index % 10000 == 0:
          print("Until line %d: attn_success_total: %d, attn_fail_total: %d, length_total: %d"%(line_index, attn_success_total, attn_fail_total, length_total))
    
    return terminal_corpus, attn_corpus

                
def save_terminals(filename, terminal_dict, attn_size, trainData_token, trainData_attn, testData_token, testData_attn):
  with open(filename, 'wb') as f:
    save = {'terminal_dict': terminal_dict,
            'attn_size': attn_size,
            'trainData_token': trainData_token,
            'trainData_attn': trainData_attn,
            'testData_token': testData_token,
            'testData_attn': testData_attn
            }
    pickle.dump(save, f, protocol=2)
    
### ANONYMIZE TERMINALS ###
def anonymize(train_dataT, vocab_size=500):
    empty_num = 0 #'<empty>'
    new_seqsT = []
    mappings = []
    inds = []
    for ind, seqT in enumerate(train_dataT[0]):
        seq_set = set()
        for w in seqT:
            if w != empty_num:
                seq_set.add(w)
        mapping = {w:newid for newid, w in zip(
                  np.random.permutation(np.arange(1, vocab_size))\
                                                 [:len(seq_set)],\
                  list(seq_set))}
        mapping[empty_num] = 0
        
        new_seqsT.append([mapping[w]\
                          if w in mapping else np.random.randint(10000, 20000)
                          for w in seqT])
        mappings.append(mapping)
        inds.append(ind)
    return new_seqsT, mappings, inds

if __name__ == "__main__":
    assert len(sys.argv) >= 2 and len(sys.argv) <= 3, "Usage: python preprocess.py {PY|JS} {optional: src_dir}"
    marker = sys.argv[1].upper()
    srcdir = sys.argv[2] if len(sys.argv) == 3 else "../data/"
    assert marker in {"PY", "JS"}
    is_js = marker == "JS"
    train_filename = "%s/train_%s.dedup.json"%(srcdir, marker.lower())
    test_filename = "%s/test_%s.dedup.json"%(srcdir, marker.lower())
    savedir = "pickle_data"
    os.makedirs(savedir, exist_ok=True)
    attn_size = 50
    
    # gather dictionary
    target_filename = "%s/terminal_dict_full_%s.pickle" % (savedir, marker)
    freq_dict = {EMPTY_TOKEN:0}
    process_dict(train_filename, freq_dict, is_js)
    process_dict(test_filename, freq_dict, is_js, increase_counts=False)
    terminal_dict = get_terminal_dict(freq_dict)
    save_terminal_dict(target_filename, terminal_dict)
    print("Finishing gathering terminal dictionary")
    
    # process non-terminals
    target_filename = '%s/%s_non_terminal.pickle' % (savedir, marker)
    typeDict = dict() #map N's name into its original ID(before expanding into 4*base_ID)
    dicID = dict() #map sparse id to dense id (remove empty id inside 4*base_ID)
    # we use all non-terminals, without frequency-based filtering
    trainData, trainParent = process_nonterminals(train_filename, typeDict, is_js)
    testData, testParent = process_nonterminals(test_filename, typeDict, is_js)
    trainData = map_dense_id(trainData, dicID)
    testData = map_dense_id(testData, dicID)
    type_dict = make_type_dict(typeDict, dicID)
    print('The number of non-terminals:', len(type_dict))
    save_nonterminals(target_filename, type_dict, trainData, testData, trainParent, testParent)
    print('Finishing processing non-terminals')

    # process terminals
    target_filename = "%s/%s_terminal.pickle" % (savedir, marker)
    trainData_token, trainData_attn = process_terminals(train_filename, terminal_dict, attn_size=attn_size, is_js=is_js)
    testData_token, testData_attn = process_terminals(test_filename, terminal_dict, attn_size=attn_size, is_js=is_js)
    save_terminals(target_filename, terminal_dict, attn_size, trainData_token, trainData_attn, testData_token, testData_attn)
    print('Finishing processing terminals')
    
    # anonymize terminals
    print('Start anonymization')
    ano_vocab_size = 500
    with open("pickle_data/%s_terminal.pickle"%marker, 'rb') as f:
        save = pickle.load(f)
    with open("pickle_data/%s_terminal_anon_vocab%d.pickle"%(marker, ano_vocab_size), 'wb') as f:
        new_train_dataT, mappings_train, _ = anonymize([save["trainData_token"]], \
                                              ano_vocab_size)
        new_test_dataT, mappings_test, _ = anonymize([save["testData_token"]], \
                                              ano_vocab_size)
        newsave = {"terminal_dict": None,
                "terminal_num": None,
                "vocab_size": 0,
                "attn_size": save["attn_size"],
                "trainData_token": new_train_dataT,
                "mappings_train": mappings_train,
                "trainData_attn": save["trainData_attn"],
                "testData_token": new_test_dataT,
                "mappings_test": mappings_test,
                "testData_attn": save["testData_attn"]
                }
        pickle.dump(newsave, f, protocol=2)
    print('Saved anonymized terminals')

