import torch
from dgq._CUDA import bmm_s8t_s8n_f32t


class BMM_S8T_S8N_F32T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer('a', torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int32
        return bmm_s8t_s8n_f32t(a, b, self.a.item())

    @staticmethod
    def from_scale(a_scale, b_scale):
        bmm_module = BMM_S8T_S8N_F32T(1.0)
        alpha = a_scale * b_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module