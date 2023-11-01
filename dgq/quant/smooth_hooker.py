import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

def sta_batch_qkv(self, inp, out):
    # Hessian H = 2 X XT + λ I
    hidden_dim = out.shape[-1]
    comming_max = torch.max(out.reshape(-1, hidden_dim).abs().detach(), dim=0)[0].float().cpu()
    if not hasattr(self,'qkv_absmax'):
        self.qkv_absmax = comming_max
    else:
        self.qkv_absmax = torch.min(self.qkv_absmax, comming_max)

def sta_batch_minmax(self, inp, out):
    # Hessian H = 2 X XT + λ I
    hidden_dim = out.shape[-1]
    comming_max = torch.max(out.view(-1, hidden_dim).detach(), dim=0)[0]
    comming_min = torch.min(out.view(-1, hidden_dim).detach(), dim=0)[0]
    if not hasattr(self,'out_max'):
        self.out_max = comming_max
    else:
        self.out_max = torch.max(self.out_max,comming_max)
    if not hasattr(self,'out_min'):
        self.out_min = comming_min
    else:
        self.out_min = torch.min(self.out_min,comming_min)

def sta_batch0(self, inp, out):
    # Hessian H = 2 X XT + λ I
    hidden_dim = out.shape[-1]
    comming_max = torch.max(out.view(-1, hidden_dim).abs().detach(), dim=0)[0].float().cpu()
    comming_mean = torch.mean(out.view(-1, hidden_dim).abs().detach(), dim=0).float().cpu()
    if not hasattr(self,'out_absmax'):
        self.out_absmax = comming_max
    else:
        self.out_absmax = torch.max(self.out_absmax,comming_max)
    if not hasattr(self, 'out_absmean'):
        self.out_absmean = comming_mean
        self.count = 1
    else:
        self.out_absmean = self.out_absmean * self.count + comming_mean
        self.count += 1
        self.out_absmean /= self.count

def sta_batch1(self, inps, out):
    # Hessian H = 2 X XT + λ I
    inp = inps[0]
    hidden_dim = inp.shape[-1]
    comming_max = torch.max(inp.view(-1, hidden_dim).abs().detach(), dim=0)[0].float().cpu()
    comming_mean = torch.mean(inp.view(-1, hidden_dim).abs().detach(), dim=0).float().cpu()
    if not hasattr(self,'inp_absmax'):
        self.inp_absmax = comming_max
    else:
        self.inp_absmax = torch.max(self.inp_absmax,comming_max)
    if not hasattr(self, 'inp_absmean'):
        self.inp_absmean = comming_mean
        self.count = 1
    else:
        self.inp_absmean = self.inp_absmean * self.count + comming_mean
        self.count += 1
        self.inp_absmean /= self.count

def prepare_hook(layer, inps, qconfig, inps_kwargs): 
    handles = []
    for mod in layer.modules():
        if isinstance(mod, nn.LayerNorm) or isinstance(mod, LlamaRMSNorm):
            if qconfig["meanact"]:
                handles.append(mod.register_forward_hook(sta_batch_minmax))
            if qconfig["smoothquant"]:
                handles.append(mod.register_forward_hook(sta_batch0))
    if isinstance(layer, LlamaDecoderLayer):
        handles.append(layer.mlp.down_proj.register_forward_hook(sta_batch1))
        handles.append(layer.self_attn.o_proj.register_forward_hook(sta_batch1))
        if qconfig['kvquant']:
            handles.append(layer.self_attn.k_quant.register_forward_hook(sta_batch_qkv))
            handles.append(layer.self_attn.v_quant.register_forward_hook(sta_batch_qkv))
            handles.append(layer.self_attn.q_quant.register_forward_hook(sta_batch_qkv))
    elif isinstance(layer, OPTDecoderLayer):
        handles.append(layer.mlp.down_proj.register_forward_hook(sta_batch1))
        handles.append(layer.self_attn.o_proj.register_forward_hook(sta_batch1))
        if qconfig['kvquant']:
            handles.append(layer.self_attn.k_quant.register_forward_hook(sta_batch_qkv))
            handles.append(layer.self_attn.v_quant.register_forward_hook(sta_batch_qkv))
            handles.append(layer.self_attn.q_quant.register_forward_hook(sta_batch_qkv))
    elif isinstance(layer, BloomBlock):
        if qconfig['kvquant']:
            handles.append(layer.self_attn.k_quant.register_forward_hook(sta_batch_qkv))
            handles.append(layer.self_attn.v_quant.register_forward_hook(sta_batch_qkv))
            handles.append(layer.self_attn.q_quant.register_forward_hook(sta_batch_qkv))
    else:
        raise NotImplemented

    for inp in inps:
        # print(inp.unsqueeze(0).shape)
        layer(inp.unsqueeze(0), **inps_kwargs)
    for h in handles:
        h.remove()
    return 