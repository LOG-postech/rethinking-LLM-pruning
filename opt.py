import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from absl import logging

from lib.opt.prune import prune_model, check_sparsity
from lib.opt.eval import eval_ppl, eval_zero_shot

def get_llm(model_name, cache_dir="llm_weights", seqlen=2048):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = min(model.config.max_position_embeddings, seqlen)
    return model

def main(config):
    # Setting seeds for reproducibility
    logging.info(config)
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if config.sparsity_type != "unstructured":
        assert config.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, config.sparsity_type.split(":"))

    logging.info(f"loading llm model {config.model}")
    model = get_llm(config.model, config.cache_dir, config.seqlen)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in config.model or "66b" in config.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    logging.info(f"use device {device}")

    logging.info("pruning starts")
    if config.sparsity_ratio != 0.0:
        prune_model(config, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    else:
        model = get_llm(config.model, config.cache_dir, config.seqlen)
    ################################################################
    logging.info("*"*30)
    sparsity_ratio, block_sparsities = check_sparsity(model)
    logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
    logging.info("*"*30)
    ################################################################
    model = model.to(torch.float16)
    model.seqlen = model.config.max_position_embeddings 
    model = model.to(device)
    eval_ppl(config, model, tokenizer, device)

    if config.eval_zero_shot:
        accelerate=False
        if "30b" in config.model or "66b" in config.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(config.model, model, tokenizer, task_list, num_shot, accelerate)
        logging.info("********************************")
        logging.info("zero_shot evaluation results")
        logging.info(results)