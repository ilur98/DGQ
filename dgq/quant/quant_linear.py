import math
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.cuda.amp import custom_bwd, custom_fwd

@torch.no_grad()
def python_compress(fdata,bit=4):
    assert bit==4
    int_data = fdata.view(-1,8//bit).to(torch.int8)
    int_data[:,0] =  (int_data[:,0] << 4) + int_data[:,1]
    return int_data[:,0].contiguous()

@torch.no_grad()
def python_decompress(int_data,bit=4):
    assert bit==4
    numel_h = int_data.shape[0]
    fdata = torch.empty((numel_h,2),device=int_data.device)
    fdata[:,0] = (int_data >> 4) % 16
    fdata[:,1] = (int_data) % 16
    return fdata

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.squeeze().view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().clamp_(min=-q_max-1,max=q_max).mul_(scales)
    return t.view(t_shape)


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.squeeze().view(-1, t_shape[-1])
    if t.shape[1] > 10:
        maxs = t.abs().max(dim=0)[0]
        maxs = maxs.sort()[0]
        scales = min(maxs[-10]*2, maxs[-1])
    else:
        scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().clamp_(min=-q_max-1,max=q_max).mul_(scales)
    return t.view(t_shape)

@torch.no_grad()
def quantize_activation_per_tensor_asym(t, n_bits=8):
    t_shape = t.shape
    t = t.squeeze().view(-1, t_shape[-1])
    if len(t) > 10:
        minv = t[3:].min()
        maxv = t[3:].max()
    else:
        minv = t.min()
        maxv = t.max()
    q_max = 2**(n_bits)-1
    scales = (maxv-minv).clamp_(min=1e-5).div_(q_max)
    t -= minv
    t.div_(scales).round_().clamp_(min=0,max=q_max).mul_(scales).add_(minv)
    return t.view(t_shape)

@torch.no_grad()
def quantize_activation_static(t, absmax, n_bits=8):
    q_max = 2**(n_bits-1)-1
    scale = absmax.to(t.device)/q_max
    t.div_(scale).round_().clamp_(-q_max,q_max).mul_(scale)
    return t

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, qconfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.actq = qconfig["act_quant"] is not None
        self.wtq = qconfig["wt_quant"] is not None
        self.qconfig = qconfig
        if self.actq:
            self.abits = self.qconfig["act_quant"]["bits"]
            self.register_buffer("amax", torch.zeros(1, dtype=torch.bfloat16))
        if self.wtq:
            self.groupsize = self.qconfig["wt_quant"]["groupsize"] if self.qconfig["wt_quant"]["groupsize"] != -1 else self.in_features
            self.wbits = self.qconfig["wt_quant"]["bits"]
            self.register_buffer('qweight', torch.zeros((in_features // 32 * self.wbits, out_features), dtype=torch.int32))
            self.register_buffer('wscales', torch.zeros((math.ceil(in_features / self.groupsize), out_features), dtype=torch.bfloat16))
            self.register_buffer('wzeros', torch.zeros((math.ceil(in_features / self.groupsize), out_features // 32 * self.wbits), dtype=torch.int32))
            if qconfig["wt_quant"]["w4w8"]:
                self.register_buffer('wscales8', torch.zeros((out_features, ), dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16))
        else:
            self.bias = None

    def unpack(self, tensor):
        if self.wbits < 8:
            fintweight = python_decompress(tensor).view(-1, self.groupsize)
        else:
            fintweight = tensor.view(-1, self.groupsize)
        if hasattr(self, "wscales8"):
            qscales = (self.wscales.view(self.out_features, -1) * self.wscales8).view(-1, 1).to(tensor.device)
        else:
            qscales = self.wscales.to(tensor.device)
        fweight = (fintweight - self.wzeros.to(tensor.device)) * qscales

        return fweight.view(self.out_features, self.in_features).bfloat16()

    def pack(self, scales, zeros):
        scales = scales.contiguous().bfloat16().reshape(-1, 1)
        self.wscales = scales
        zeros = zeros.contiguous().bfloat16().reshape(-1, 1)
        self.wzeros = zeros
        scale_zeros = zeros.reshape(-1,1) * scales.reshape(-1,1)
        intweight = torch.round((self.weight.data.view(-1, self.groupsize)) / self.wscales + self.wzeros).to(torch.int)
        delattr(self, "weight")
        if self.wbits < 8:
            self.qweight = python_compress(intweight)
        else:
            self.qweight = intweight
    def prepare_actfun(self):
        if self.qconfig["act_quant"] is None:
            return
        if self.qconfig["act_quant"]["method"] == "static":
            self.act_quant = partial(quantize_activation_static,absmax=self.amax)
            # self.act_quant = quantize_activation_static
        elif self.qconfig["act_quant"]["method"] == "per_tensor":
            self.act_quant = quantize_activation_per_tensor_absmax
        elif self.qconfig["act_quant"]["method"] == "per_token":
            self.act_quant = quantize_activation_per_token_absmax
        else:
            raise NotImplemented
    def packW4W8(self, scales, zeros, scales8):
        scales = scales.contiguous().char().reshape(-1, 1)
        self.wscales = scales
        zeros = zeros.contiguous().char().reshape(-1, 1)
        self.wzeros = zeros
        scales8 = scales8.contiguous().bfloat16().reshape(-1, 1)
        self.wscales8 = scales8.reshape(-1, 1)
        qscales = (self.wscales.view(self.out_features, -1) * self.wscales8).view(-1, 1)
        intweight = torch.round((self.weight.data.view(-1, self.groupsize).float()) / qscales.reshape(-1, 1) + self.wzeros).to(torch.int)
        self.qweight = python_compress(intweight)
        delattr(self, "weight")

    def setquant(self, actq, wtq):
        self.actq = actq
        self.wtq = wtq

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        if self.actq:
            x = self.act_quant(x)
        if self.wtq:
            weight = self.unpack(self.qweight)
        else:
            weight = self.weight
        out = x.reshape(-1, x.shape[-1]) @ weight.t()
        out = out + self.bias if self.bias is not None else out 
        return out.reshape(out_shape).to(x.dtype)

