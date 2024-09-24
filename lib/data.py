# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
import datasets
import json

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets

    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_generated(nsamples, seed, seqlen, tokenizer):

    f = open("self_data.jsonl", "r", encoding="utf-8")

    data = []
    i = 0
    while True:
        line = f.readline()
        if not line:
            break
        try:
            data.append(json.loads(line))
        except:
            pass
        i += 1
    f.close()
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            trainenc = tokenizer(data[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, trainloader

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, self_nsamples=0):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "gen_add" in name:
        # the first "nsamples" are from c4 and the remaining "self_nsamples" are from self-generated data
        c4_trainloader, c4_testloader = get_c4(nsamples, seed, seqlen, tokenizer)
        gen_trainloader, gen_testloader = get_generated(self_nsamples, seed, seqlen, tokenizer)
        return c4_trainloader + gen_trainloader, c4_testloader