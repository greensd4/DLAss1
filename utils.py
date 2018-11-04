# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
import xor_data as xd
def read_data(fname):
    data = []
    for line in file(fname):
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

def dim_to_couples(dims):
    return [(n1,n2) for n1,n2 in zip(dims,dims[1:])]

def params_to_couples(params):
    p = list(params)
    coupled_list = []
    for index in range(0, len(p), 2):
        coupled_list.append((p[index],p[index+1]))
    return coupled_list


def reverse_params(params):
    rev_list = []
    for w, b in zip(params[0::2], params[1::2]):
        rev_list.append(b)
        rev_list.append(w)
    return list(reversed(rev_list))

TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data("train")]
DEV   = [(l,text_to_bigrams(t)) for l,t in read_data("dev")]
XOR = [(l,data) for l,data in xd.data]

from collections import Counter
fc = Counter()
for l,feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x,c in fc.most_common(600)])

# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}
# XOR labels to IDs
#X2I = {l:i for i,l in enumerate(list(sorted(set(XOR))))}

