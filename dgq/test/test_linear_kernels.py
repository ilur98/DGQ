import sys
sys.path.append(".")
import torch
import os
from dgq._CUDA import linear_a8_w4_b8_o8, linear_a8_w4_bfp32_ofp32
from icecream import ic
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.setd
def decompress_python(weight, scales, qzeros, infeatures):
    numel = weight.shape[0]
    groupsize = qzeros.shape[0]
    # weight = weight.view(groupsize, -1)
    fdata = torch.empty((numel, 2), dtype = torch.int8)
    fdata[:, 0] = (weight >> 4)%16
    fdata[:, 1] = weight % 16
    fdata = (fdata.view(groupsize, -1) - qzeros) * scales

    return fdata.view(-1, infeatures)


@torch.no_grad()
def test_quant_linear_a8_w8_bfp32_ofp32():
    B, M, N = 2048, 5120, 5120
    weight = torch.randint(-128, 127, (N*M // 2,), dtype=torch.int8)
    bias = torch.rand(N, dtype=torch.float)
    x = torch.randint(-127, 127, (B, M), dtype=torch.int8)
    # x = torch.ones( (B, M), dtype=torch.int8)
    alpha = torch.rand((N, 1), dtype=torch.float)
    beta = torch.rand(1, dtype=torch.float)
    scales8 = torch.randint(0, 8, (N * M // 128, 1), dtype=torch.int8)
    zeros = torch.randint(0, 15, (N * M // 128, 1), dtype=torch.int8)
    linear = torch.nn.Linear(M, N, bias=True)
    fweight = decompress_python(weight, scales8, zeros, M)
    linear.weight.data = fweight.float() * alpha.float()
    linear.bias.data = bias.float()
    y_gt = linear(x.float())
    y = linear_a8_w4_bfp32_ofp32(
        x.cuda(), weight.cuda(), bias.cuda(), 
        alpha.cuda(), beta.cuda(),
        scales8.cuda(), zeros.cuda(), M, N, 128 // 8).float()
    ic(torch.allclose(y_gt.float(), y.float().cpu(), atol=0.5))


@torch.no_grad()
def test_quant_linear_a8_w8_b8_o8():
    B, M, N = 128, 512, 1024
    weight = torch.randint(-128, 127, (N*M // 2,), dtype=torch.int8)
    bias = torch.randint(-128, 127, (N,), dtype=torch.int8)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    alpha = torch.rand((N, 1), dtype=torch.float) * 0.001
    beta = torch.rand(1, dtype=torch.float)
    scales8 = torch.randint(0, 8, (N * M //128 , 1), dtype=torch.int8)
    zeros = torch.randint(0, 15, (N * M // 128, 1), dtype=torch.int8)
    linear = torch.nn.Linear(M, N, bias=True)
    fweight = decompress_python(weight, scales8, zeros, M)
    linear.weight.data = fweight.float() * alpha
    linear.bias.data = bias.float() * beta
    y_gt = linear(x.float()).clamp(-128, 127).round().long()
    alpha_t = alpha.reshape(-1, 8, 2, 8).transpose(1, 2).flatten()
    y = linear_a8_w4_b8_o8(x.cuda(), weight.cuda(), bias.cuda(), 
        alpha_t.cuda(), beta.cuda(),
        scales8.cuda(), zeros.cuda(), M, N, 128 // 8).cpu().long()
    ic(torch.allclose(y_gt.float(), y.float().cpu(), atol=1))

if __name__ == '__main__':
    print('test_quant_linear_a8_w8_bfp32_ofp32')
    test_quant_linear_a8_w8_bfp32_ofp32()
    print('test_quant_linear_a8_w8_b8_o8')
    test_quant_linear_a8_w8_b8_o8()

