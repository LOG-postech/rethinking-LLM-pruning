import time 
import torch 
import torch.nn as nn 
from ..sparsegpt import SparseGPT 
from ..layerwrapper import WrappedGPT
from ..data import get_loaders 
from .finetune import train, val, obtain_output
from ..linear_type import LinearMasked, LinearMaskedOPT
from torch.cuda.amp import autocast

from absl import logging

def find_layers(module, layers=[nn.Linear], masked_layers=[LinearMasked, LinearMaskedOPT], name=''):
    """
    Recursively find the layers of a certain type in a module.

    config:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers or type(module) in masked_layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, masked_layers=masked_layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    sparsities = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        logging.info(f"block {i} sparsity {float(sub_count)/sub_params:.6f}")

        sparsities.append(float(sub_count)/sub_params)

    model.config.use_cache = use_cache 
    return float(count)/total_params, sparsities

def prepare_calibration_input(model, dataloader, device, config):
    """
    Prepare calibration samples to be used for pruning
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    recon_nsamples = config.nsamples + config.self_nsamples
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((recon_nsamples, config.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    attention_mask = cache['attention_mask']

    if config.use_fp32:
        inps = inps.float()
        dtype = torch.float32

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache

    return inps, outs, attention_mask

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    W = layer.weight.data 
    M = layer.mask.data
    W_metric = torch.abs(W)
    if prune_n != 0:
        W_mask = (torch.zeros_like(W)==1)
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=True)[1], True)
    else:
        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
        W_mask = (W_metric>thresh)

    M[W_mask] = 1

def prune_wanda(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    W_metric = torch.abs(layer.weight.data) * torch.sqrt(wrapped_layer.scaler_row.reshape((1,-1)))

    if prune_n != 0:
        # structured n:m sparsity
        final_indices = None
        with torch.no_grad():
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    indices = ii+torch.topk(tmp, prune_n,dim=1, largest=True)[1].cuda()
                    layer.mask.scatter_(1, indices, 1)
                    if final_indices is None:
                        final_indices = indices.cpu()
                    else:
                        final_indices = torch.concat((final_indices,indices.cpu()),1)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True, descending=True)

        #unstructured pruning
        indices = sort_res[1][:,:int(W_metric.shape[1]*(1 - sparsity_ratio))]
        with torch.no_grad():
            layer.mask.scatter_(1, indices.cuda(), 1)

def prune_sparsegpt(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    wrapped_layer.fasterprune(sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
    wrapped_layer.free()

def prune_model(config, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    Prune the given model
    """
    layers = model.model.decoder.layers 
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    prune_nsamples = config.nsamples
    recon_nsamples = config.nsamples + config.self_nsamples

    # load the calibration data
    dataloader, _ = get_loaders("gen_add",nsamples=recon_nsamples,seed=config.seed,seqlen=config.seqlen,tokenizer=tokenizer,self_nsamples=config.self_nsamples)
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device, config)
    layers = model.model.decoder.layers

    dense_inps = inps.clone()
    dense_outs = torch.zeros_like(inps)

    if config.prune_method == 'magnitude':
        prune_fn = prune_magnitude
    elif config.prune_method == 'wanda':
        prune_fn = prune_wanda
    elif config.prune_method == 'sparsegpt':
        prune_fn = prune_sparsegpt
    
    dense_layers = []

    if config.use_cr:
        update_round = len(layers) + 1
        update_start = -1
    else:
        update_round = len(layers)
        update_start = 0

    for i in range(update_start, update_start + update_round):
        if config.use_cr:
            if i == -1:
                logging.info(f'pruning layer {i + 1}')
            elif i == (len(layers) - 1):
                logging.info(f'pruning layer {len(layers) - 1}')
            else:
                logging.info(f'pruning layer {i} & {i+1}')
        else:
            logging.info(f'pruning layer {i}')

        if config.use_cr:
            # For CR, save the current dense layer
            start_idx = max(0, i)
            end_idx = min(i + 1, len(layers) - 1)
            layer = layers[start_idx:end_idx + 1]
            if not (i ==  (len(layers) - 1)):
                dense_layer = type(layer[-1])(model.config).to(torch.float16).eval()
                dense_layer.load_state_dict(layer[-1].state_dict())
                dense_layers.append(dense_layer)
            if config.use_fp32:
                dense_layer = dense_layer.float()
        else:
            layer = layers[i:i+1]

        if config.use_fp32:
            layer = layer.float()
            
        for layer_ in layer:
            layer_.self_attn.q_proj = LinearMaskedOPT(layer_.self_attn.q_proj)
            layer_.self_attn.k_proj = LinearMaskedOPT(layer_.self_attn.k_proj)
            layer_.self_attn.v_proj = LinearMaskedOPT(layer_.self_attn.v_proj)
            layer_.self_attn.out_proj = LinearMaskedOPT(layer_.self_attn.out_proj)
            layer_.fc1 = LinearMaskedOPT(layer_.fc1)
            layer_.fc2 = LinearMaskedOPT(layer_.fc2)

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            device = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask = inps.to(device), outs.to(device), attention_mask.to(device)
        
        wrapped_layers = {}

        for name in subset:
            subset[name].prune_rate = 0
            if config.prune_method == 'magnitude':
                wrapped_layers[name] = None
            if config.prune_method == 'wanda':
                wrapped_layers[name] = WrappedGPT(subset[name],device)
            elif config.prune_method == 'sparsegpt':
                wrapped_layers[name] = SparseGPT(subset[name], device)
        
        # to calculate intermediate statistics to be used for pruning
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        layer = layer.to(device)
        handles = []
        
        if config.prune_method in ['wanda', 'sparsegpt']:
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # obtain outputs for x_sparse
        with torch.no_grad():
            for j in range(prune_nsamples):
                outs[j] = obtain_output(layer, inps[j].unsqueeze(0).cuda(), attention_mask)
        
        for h in handles:
            h.remove()
        
        # obtain outputs for self-generated samples
        with torch.no_grad():
            for j in range(prune_nsamples, recon_nsamples):
                outs[j] = obtain_output(layer, inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)

        # obtain outputs for x_dense
        if config.use_cr:
            with torch.no_grad():
                for j in range(start_idx, end_idx + 1):
                    dense_layers[j] = dense_layers[j].to(device)
                for j in range(0, recon_nsamples, config.infer_batch_size):
                    dense_outs[j:j+config.infer_batch_size] = obtain_output(dense_layers[start_idx:end_idx + 1], dense_inps[j:j+config.infer_batch_size].to(device), attention_mask=attention_mask.expand(config.infer_batch_size, -1, -1, -1)).to("cpu")
                for j in range(start_idx, end_idx + 1):
                    dense_layers[j] = dense_layers[j].to('cpu')
        else:
            with torch.no_grad():
                for j in range(0, recon_nsamples, config.infer_batch_size):
                    dense_outs[j:j+config.infer_batch_size] = layer[0](dense_inps[j:j+config.infer_batch_size].to(device), attention_mask=attention_mask.expand(config.infer_batch_size, -1, -1, -1))[0].to("cpu")

        # prune the weights
        for name in subset:
            subset[name].mask = nn.Parameter(torch.zeros(subset[name].weight.shape, device=device))
            prune_fn(subset[name], wrapped_layers[name], config.sparsity_ratio, prune_n, prune_m)
            subset[name].prune_rate = config.sparsity_ratio
    
        # perform reconstruction
        if config.use_gp:
            train(layer, inps, dense_outs, dataloader, config, device, attention_mask=attention_mask)
        else:
            train(layer, inps, outs, dataloader, config, device, attention_mask=attention_mask)
        
        layer = layer.to(device)

        # calculate recon error
        if not config.use_cr:
            recon_error = val(layer, inps, dense_outs, config, device, attention_mask)
            logging.info(f"recon error {recon_error}")

        with torch.no_grad():
            for name in subset:
                mask_copy1 = subset[name].mask.clone()
                subset[name].weight.data = mask_copy1 * subset[name].weight.data
                subset[name].prune_rate = 0
        torch.cuda.empty_cache()

        # calculate outputs for x_sparse after pruning
        with torch.no_grad():
            with autocast():
                for j in range(recon_nsamples):
                    outs[j] = layer[0](inps[j].unsqueeze(0).to(device), attention_mask=attention_mask)[0].to("cpu")
        
        if config.use_cr:
            if i < 0:
                dense_outs = dense_inps.clone()
                outs = inps.clone()
            else:
                with torch.no_grad():
                    with autocast():
                        dense_layers[i] = dense_layers[i].to(device)
                        for j in range(0, recon_nsamples, config.infer_batch_size):
                            dense_outs[j:j+config.infer_batch_size] = obtain_output(dense_layers[i:i+1], dense_inps[j:j+config.infer_batch_size].to(device), attention_mask.expand(config.infer_batch_size, -1, -1, -1)).to('cpu')
                        dense_layers[i] = dense_layers[i].to('cpu')
                
                recon_error = val(layer[0:1], inps, dense_outs, config, device, attention_mask)
                logging.info(f"recon error {recon_error}")
                dense_layers[i] = None

        inps, outs = outs, inps
        dense_inps, dense_outs = dense_outs, dense_inps
        layer = layer.to('cpu')
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()