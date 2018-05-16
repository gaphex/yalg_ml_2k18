import numpy as np
import itertools

from tqdm import tqdm
from collections import Counter


def read_fasttext(fpath):
    embeddings_index = {}
    with open(fpath,encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def build_vocab(tokenlists, max_size=200000, emb_model=None):
    """
    Builds a vocabulary of at most max_size words from the supplied list of lists of tokens.
    If a word embedding model is provided, adds only the words present in the model vocabulary.
    """

    all_words = list(itertools.chain.from_iterable(tokenlists))
    counter = Counter(all_words)
    if emb_model:
        counter = Counter({x: counter[x] for x in counter if x in emb_model})
        
    reserved_symbols = ["NULL", "UNKN", "<S>", "</S>"]
    for reserved in reserved_symbols:
        try:
            counter.pop(reserved)
        except KeyError:
            pass
            
    commons = counter.most_common(max_size-len(reserved_symbols))

    voc = {}
    for i, reserved in enumerate(reserved_symbols):
        voc[reserved] = i

    for i, k in enumerate(commons):
        voc[k[0]] = i+len(reserved_symbols)

    rvoc = {v: k for k, v in voc.items()}

    return voc, rvoc

def vectorize_tokens(tokens, token_to_id, max_len):
    """
    Converts a list of tokens to a list of token ids using the supplied dictionary.
    Pads resulting list with NULL identifiers up to max_len length.
    """
    ids = [token_to_id["<S>"]]
    
    for token in tokens:
        ids.append(token_to_id.get(token, token_to_id["UNKN"]))

    if len(ids) < max_len:
        ids.append(token_to_id["</S>"])
        ids += (max_len-len(ids))*[token_to_id["NULL"]]
    else:
        ids = ids[:max_len]
        ids[-1] = token_to_id["</S>"]

    return ids

def vectorize(tok_lists, token_to_id, max_len=150):
    """
    Converts a list of lists of tokens to a numpy array of token identifiers
    """
    
    token_matrix = []
        
    for tok_list in tqdm(tok_lists):
        token_ids = vectorize_tokens(tok_list, token_to_id, max_len)
        token_matrix.append(token_ids)
    
    token_matrix = np.array(token_matrix)
        
    return token_matrix

def get_embeddings(model, rev_voc, dim=300):
    """
    Prepares a matrix of embeddings corresponding to the provided id-to-word mapping
    """

    myembeddings = []
    for ik, key in enumerate(sorted(rev_voc.keys())):
        val = rev_voc[key]
        if val == 'NULL':
            myembeddings.append(np.zeros((dim,)))
        elif val == 'UNKN':
            myembeddings.append(np.random.normal(size=(dim,)))
        elif val in {'<S>','</S>'}:
            myembeddings.append(np.ones((dim,)))
        else:
            try:
                myembeddings.append(model[val])
            except KeyError:
                print("OOV: {}, {}".format(ik, val))
                myembeddings.append(np.random.normal(size=(dim,)))

    myembeddings = np.array(myembeddings)
    return myembeddings