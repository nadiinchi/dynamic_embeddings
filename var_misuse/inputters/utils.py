import random
import string
import json
from collections import Counter
from tqdm import tqdm
import subprocess

from inputters.objects import Code
from inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD

def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

def generate_random_string(N=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))

def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def process_examples(source,
                     source_tag,
                     max_src_len,
                     max_tgt_len,
                     uncase=False):
    code_tokens = source.split()
    code_type = []
    code_type = source_tag.split()
    if len(code_tokens) != len(code_type):
        print(code_tokens, code_type)
        raise ValueError("len(code_tokens) != len(code_type): %d %d" % \
                         (len(code_tokens), len(code_type)))

    code_tokens = code_tokens[:max_src_len]
    code_type = code_type[:max_src_len]
    
    if len(code_tokens) == 0:
        raise ValueError("empty code_tokens:", code_tokens)

    code = Code()
    code.text = source
    code.tokens = code_tokens
    code.type = code_type

    example = dict()
    example['code'] = code
    return example

def load_word_and_char_dict(args, dict_filename, dict_size=None,
                             special_tokens="pad_unk"):
    """Return a dictionary from question and document words in
    provided examples.
    """
    with open(dict_filename) as fin:
        words = set(fin.read().split("\n")[:dict_size])
    dictionary = UnicodeCharsVocabulary(words,
                                        100,
                                        special_tokens)
    return dictionary

def build_word_and_char_dict_from_file(filenames, dict_size=None,
                             special_tokens="pad_unk", sum_over_subtokens=False, 
                             split_elem="_", max_characters_per_token=100):
    """Return a dictionary from tokens in provided files.
       max_characters_per_token would be needed if words were encoded by chars
    """
    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    if type(filenames) == str:
        filenames = [filenames]
    for fn in filenames:
        with open(fn) as f:
            for line in tqdm(f, total=count_file_lines(fn)):
                tokens = line.strip().split()
                if not sum_over_subtokens:
                    _insert(tokens)
                else:
                    for elem in tokens:
                        _insert(elem.split(split_elem))

    num_spec_tokens = len(special_tokens.split("_"))
    dict_size = dict_size - num_spec_tokens if dict_size and dict_size > num_spec_tokens else dict_size
    most_common = word_count.most_common(dict_size)
    words = [word for word, _ in most_common]
    dictionary = UnicodeCharsVocabulary(words,
                                        max_characters_per_token,
                                        special_tokens)
    return dictionary
