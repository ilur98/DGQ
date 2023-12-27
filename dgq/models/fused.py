import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm
class LayerNormQ(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.input_scale = 1.0
        self.eps = eps
        self.register_buffer('weight', torch.ones(dim, dtype=torch.float32))
        self.register_buffer('bias', torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        x = x.to(self.weight.dtype)
        ln_output_fp = torch.nn.functional.layer_norm(
            x, x.shape[-1:], self.weight, self.bias, self.eps)
        ln_output_int8 = ln_output_fp.round().clamp(-128, 127).to(torch.int8)
        return ln_output_int8

    @staticmethod
    def from_float(module: torch.nn.LayerNorm, output_scale: float):
        assert module.normalized_shape[0] == module.weight.numel()
        assert module.normalized_shape[0] == module.bias.numel()
        q_module = LayerNormQ(module.normalized_shape[0], module.eps)
        q_module.weight = module.weight.float() / output_scale
        q_module.bias = module.bias.float() / output_scale
        return q_module

class RMSNormQ(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.variance_epsilon = eps
        self.register_buffer('weight', torch.ones(dim, dtype=torch.float32))
    old_forward = LlamaRMSNorm.forward

    def forward(self, x):
        ln_output_fp = self.old_forward(x)
        ln_output_int8 = ln_output_fp.round().clamp(-128, 127).to(torch.int8)
        return ln_output_int8

    @staticmethod
    def from_float(module: torch.nn.LayerNorm, output_scale: float):
        q_module = RMSNormQ(module.weight.numel(), module.variance_epsilon)
        q_module.weight = module.weight.float() / output_scale
        return q_module
