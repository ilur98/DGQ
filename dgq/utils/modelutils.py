import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, BloomAttention
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTAttention
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention
from dgq.quant.quant_linear import QuantLinear
from dgq.quant.kvquanter import BLOOMAttention_QKVQuant, OPTAttention_QKVQuant, LlamaAttention_QKVQuant
DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def gen_conditions(_wbits, _groupsize):
    wbits = _wbits
    groupsize = _groupsize
    conditions = []
    while True:
        if wbits >= 8:
            if groupsize == -1 or groupsize == 32:
                break

        if groupsize > 32:
            groupsize /= 2
        else:
            wbits *= 2
            groupsize = _groupsize

        conditions.append((int(wbits), int(groupsize)))
    return conditions


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))
    return layers

def move_norm_head(model, device):
    mod_list = torch.nn.ModuleList()
    mod_list.append(nn.Identity().to(device))
    if isinstance(model, LlamaForCausalLM):
        model.lm_head = model.lm_head.to(device)
        if model.model.norm is not None:
            mod_list.append(model.model.norm.to(device))
    elif isinstance(model, OPTForCausalLM):
        model.lm_head = model.lm_head.to(device)
        if model.model.decoder.final_layer_norm is not None:
            mod_list.append(model.model.decoder.final_layer_norm.to(device))
        if model.model.decoder.project_out is not None:
            mod_list.append(model.model.decoder.project_out.to(device))
    elif isinstance(model, BloomForCausalLM):
        model.lm_head = model.lm_head.to(device)
        mod_list.append(model.transformer.ln_f.to(device))
    # elif "mpt" in str(model.__class__).lower():
    #     model.transformer.wte = model.transformer.wte.to(device)
    #     model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    # elif "falcon" in str(model.__class__).lower():
    #     model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    else:
        raise NotImplementedError(type(model))
    return mod_list
def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    else:
        raise NotImplementedError(type(model))

def convert_model(module, qconfig):
    if isinstance(module, QuantLinear):
        return
    for name, mod in module.named_children():
        if isinstance(mod, nn.Linear) and not name.endswith("head"):
            newlayer = QuantLinear(mod.in_features, mod.out_features, hasattr(mod, "bias"), qconfig)
            newlayer.weight = mod.weight
            if hasattr(mod, "bias"):
                newlayer.bias = mod.bias
            setattr(module, name, newlayer)
        elif isinstance(mod, OPTAttention):
            OPTAttention_QKVQuant(mod, qconfig)
        elif isinstance(mod, BloomAttention):
            BLOOMAttention_QKVQuant(mod, qconfig)
        elif isinstance(mod, LlamaAttention):
            LlamaAttention_QKVQuant(mod, qconfig)
        convert_model(mod, qconfig)

# copy from https://github.com/openppl-public/ppq/blob/master/ppq/quantization/measure/norm.py
def torch_snr_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)
    
    SNR can be calcualted as following equation:
    
        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2
    
    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.
    
        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)
    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.
    Raises:
        ValueError: _description_
        ValueError: _description_
    Returns:
        torch.Tensor: _description_
    """
    y_pred = y_pred.type(torch.float32)
    y_real = y_real.type(torch.float32)

    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute snr loss for tensors with different shape. '
                         f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=1)
    y_real = y_real.flatten(start_dim=1)

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == 'mean':
        return torch.mean(snr)
    elif reduction == 'sum':
        return torch.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')
