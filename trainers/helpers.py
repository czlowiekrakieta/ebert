import numpy as np
import re
from collections import defaultdict
from random import sample

clearer = re.compile(r'[^a-z ]')
spaces = re.compile(r'\s{2,')

def sample_cap(arr, size):
    if size < len(arr):
        return sample(arr, size)
    return arr

def presplit(sentence):
    if isinstance(sentence, list):
        return list(map(presplit, sentence))
    sentence = clearer.sub(' ', sentence)
    sentence = spaces.sub(' ', sentence)
    return sentence.split()


def concat_with_zeros(arr, bs):
    sh = arr.shape
    return np.concatenate((arr, np.zeros(tuple([bs-sh[0]] + list(sh[1:])), dtype=arr.dtype)), axis=0)


def tokenize_seqs(sequences_train,
                  sequences_val=None,
                  melted=False,
                  count_threshold=5):
    idx = {}
    counts = defaultdict(int)
    tokenized = []
    for i, seq in enumerate(sequences_train):
        stok = []
        for j, ent in enumerate(seq):
            if ent not in idx:
                idx[ent] = len(idx)+1
            counts[idx[ent]] += 1
            stok.append(idx[ent] if not melted else (i, j))
        if not melted:
            tokenized.append(stok)
        else:
            tokenized.extend(stok)

    def cut(seq):
        return counts[seq] > count_threshold

    tokenized = [list(filter(cut, seq)) for seq in tokenized]
    if sequences_val is not None:
        tokenized_val = [[idx[x] for x in seq if x in idx] for seq in sequences_val]
        tokenized_val = [list(filter(cut, seq)) for seq in tokenized_val]

        return tokenized, tokenized_val, idx

    return tokenized, idx


def _melt(seqs_of_ints):
    melted = []
    for i, seq in enumerate(seqs_of_ints):
        melted.extend((i, j-1) for j in seq)
    return list(zip(*melted))


def _pad(sequences, PAD=0):
    L = max(map(len, sequences))
    return np.asarray(
        [x + [PAD]*(L-len(x)) for x in sequences]
    )


def _build_binary_mask(lengths):
    return np.asarray(
        [[1]*x+[0]*(max(lengths)-x) for x in lengths]
    )