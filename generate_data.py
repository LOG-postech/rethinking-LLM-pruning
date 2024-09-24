# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code taken from https://github.com/facebookresearch/LLM-QAT

from transformers import AutoTokenizer
import torch
import json
import sys
import os

from absl import flags, app
from ml_collections import config_flags

from llama import get_llm

import numpy as np

from langdetect import detect

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)

def main(argv):
    config = FLAGS.config
    np.random.seed(config.seed)
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=False)
    print("Tokenizer loaded!")
    print("Loading model")
    model = get_llm(config.model, config.cache_dir, config.seqlen)
    model = model.cuda()
    print("Model loaded!")
    
    for i in range(config.nsamples):
        while True:
            # generate 5 initial tokens
            start_vocab = np.random.randint(tokenizer.vocab_size)
            input_ids = torch.tensor([[start_vocab]]).cuda()
            outputs1 = model.generate(input_ids, do_sample=False, max_length=5)
            text = tokenizer.batch_decode(outputs1, skip_special_tokens=True)[0]
            # only accept the English strings. otherwise, re-generate the data
            try:
                if detect(text) == 'en':
                    break
            except:
                pass
                
        # generate the remaining tokens
        outputs = model.generate(outputs1, do_sample=True, max_length=2048)
        
        # decode the outputs and save the data
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        text_dict = {"text" : gen_text[0]}
        with open( "self_data.jsonl", "a") as f:
            f.write(json.dumps(text_dict))
            f.write('\n')


if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  app.run(main)
