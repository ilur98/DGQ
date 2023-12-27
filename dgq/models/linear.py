import torch
from dgq.quant.quant_linear import QuantLinear
from dgq._CUDA import (linear_a8_w4_b8_o8,
                     linear_a8_w4_bfp32_ofp32
                     )

class W4A8B8O8Linear(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features, groupsize=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.groupsize = groupsize
        self.register_buffer('weight', torch.zeros((self.out_features,
                                                    self.in_features // 2), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('a', torch.zeros(1, self.out_features))
        self.register_buffer('b', torch.ones(1))
        self.register_buffer('scales8', torch.zeros((self.out_features, self.in_features//self.groupsize),dtype=torch.int8))
        self.register_buffer('zeros', torch.zeros((self.out_features, self.in_features//self.groupsize),dtype=torch.int8))
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w4_b8_o8(x, self.weight, self.bias,
                               self.a, self.b,
                               self.scales8, self.zeros, self.in_features, self.out_features,
                               self.groupsize // 8)
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: QuantLinear, input_scale, output_scale):
        a8w4_module = W4A8B8O8Linear(
            module.in_features, module.out_features)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * module.wscales8.float() / output_scale
        beta = bias_scale / output_scale
        a8w4_module.weight = module.qweight
        a8w4_module.bias = int8_bias
        a8w4_module.a = alpha.reshape(-1, 8, 2, 8).transpose(1, 2).flatten()
        a8w4_module.b = beta
        a8w4_module.scales8 = module.wscales
        a8w4_module.zeros = module.wzeros
        return a8w4_module

class W4A8BF32OF32Linear(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features, groupsize=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.register_buffer('weight', torch.zeros((self.out_features,
                                                    self.in_features // 2), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.float, requires_grad=False))
        self.register_buffer('a', torch.zeros(1, self.out_features))
        self.register_buffer('b', torch.zeros(1, self.out_features))
        self.register_buffer('scales8', torch.zeros((self.out_features, self.in_features//self.groupsize),dtype=torch.int8))
        self.register_buffer('zeros', torch.zeros((self.out_features, self.in_features//self.groupsize),dtype=torch.int8))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w4_bfp32_ofp32(x, self.weight, self.bias,
                               self.a, self.b,
                               self.scales8, self.zeros, self.in_features, self.out_features,
                               self.groupsize // 8)
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: QuantLinear, input_scale):
        a8w4_module = W4A8BF32OF32Linear(
            module.in_features, module.out_features, module.groupsize)
        alpha = module.wscales8.float() * input_scale
        a8w4_module.weight = module.qweight
        if module.bias is not None:
            a8w4_module.bias = module.bias.float()
        a8w4_module.a = alpha
        a8w4_module.scales8 = module.wscales
        a8w4_module.zeros = module.wzeros
        return a8w4_module

@torch.no_grad()
def quantize_per_tensor_absmax(t):
    scale = t.abs().max() / 127
    if not t.is_cuda:
        # half rounding is not supported on CPU
        t = t.float()
    # use inplace operation to save memory
    t.div_(scale).round_()
    t_q = t.to(torch.int8)
    return t_q, scale