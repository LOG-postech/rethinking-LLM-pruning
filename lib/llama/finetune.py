# from transformers.utils import logging
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from ..tensor_dataloader import TensorData, TensorDataLoader
from absl import logging

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def loss_fn(inp, tar):
    losses = (inp - tar)**2
    loss = losses.mean()
    return loss

def obtain_output(sub_layers, inp, attention_mask, position_ids, device=torch.device("cuda:0"), offload=False):
    """
    Obtain outputs for inp when passing through sub_layers
    """
    for sub_layer_idx in range(len(sub_layers)):
        if offload:
            sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].to(device)
        if sub_layer_idx == 0:
            out = sub_layers[sub_layer_idx](inp, attention_mask=attention_mask, position_ids=position_ids)[0]
        else:
            out = sub_layers[sub_layer_idx](out, attention_mask=attention_mask, position_ids=position_ids)[0]
        
        if offload:
            sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].to('cpu')
    return out

def val(layer, inps, outs, config, device, attention_mask, position_ids):
    """
    Calculate recon error
    """
    ret_loss = 0
    tensordata = TensorData(inps, outs, device)
    tensordata_loader = TensorDataLoader(tensordata, config.infer_batch_size, shuffle=False, num_workers=0).get_loader()
    criterion = loss_fn
    with torch.no_grad():
        layer.eval()
        for inputs, outs in tensordata_loader:
            with autocast():
                outputs = obtain_output(layer, inputs, attention_mask=attention_mask.expand(len(inputs),-1,-1,-1), position_ids=position_ids)
                loss = criterion(outputs, outs)

            ret_loss += (loss.detach().cpu().item()) * len(inputs)
    return ret_loss / len(inps)

def train(layer, inps, outs, dataloader, config, device, attention_mask, position_ids):
    """
    Update the remaining weights after pruning
    """
    criterion = loss_fn
    init_loss = val(layer, inps, outs, config, device, attention_mask, position_ids)
    len_dataloader = len(dataloader)
    num_update_steps_per_epoch = len_dataloader // config.batch_size
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    max_steps = math.ceil(config.epochs * num_update_steps_per_epoch)

    mark_only_weights_as_trainable(layer)
    scaler = GradScaler()
    optimizer,lr_scheduler = prepare_optimizer_and_scheduler(layer, config, max_steps)
    losses = []
    tensordata = TensorData(inps, outs, device)
    tensordata_loader = TensorDataLoader(tensordata, config.batch_size // config.accumulation_steps, shuffle=True, num_workers=0).get_loader()

    n_iter = 0

    losses.append(init_loss)

    for epoch in range(0, config.epochs):
        ret_loss = 0
        for inputs, outps in tensordata_loader:
            with autocast():
                with torch.enable_grad():
                    outputs = obtain_output(layer, inputs, attention_mask=attention_mask.expand(len(inputs),-1,-1,-1), position_ids=position_ids)
                    loss = criterion(outputs, outps)
                    loss = loss / config.accumulation_steps
                    if config.use_fp32:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
            torch.nn.utils.clip_grad_norm_(
                        layer.parameters(), config.max_grad_norm)
            if (n_iter + 1) % config.accumulation_steps == 0:
                if config.use_fp32:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                layer.zero_grad()
            ret_loss += (loss.detach().cpu().item()) * len(inputs)
            n_iter += 1
        losses.append(ret_loss / len(inps))
    torch.cuda.empty_cache()
    final_loss = val(layer, inps, outs, config, device,attention_mask, position_ids)
    logging.info(losses)
    return losses, final_loss

def prepare_optimizer_and_scheduler(layer, config, max_steps):
    def log_params(param_groups, des):
        for i, grouped_parameters in enumerate(param_groups):
            logging.info(
                f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

    update_params = [p for n, p in layer.named_parameters() if 'mask' not in n]

    main_model_params = [
        {
            "params": update_params,
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate
        },
    ]
    log_params(main_model_params, "weight params")
    optimizer = AdamW(
        main_model_params,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=max_steps
    )
    return optimizer, lr_scheduler

def mark_only_weights_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'mask' in n:
            p.requires_grad = False