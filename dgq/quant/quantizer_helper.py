import math
import time

import torch
import torch.nn as nn
import transformers
from dgq.quant.quantizer import Quantizer
from dgq.quant.quant_linear import QuantLinear
from texttable import Texttable
from dgq.utils import torch_snr_error

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

Observer = None
class QuantizerHelper:

    def __init__(self, layer, observe=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()
        self.observe = observe
        self.inp_absmax = None

    def add_batch(self, inp, out):
        # Hessian H = 2 X XT + λ I
        hidden_dim = inp.shape[-1]
        comming_max = torch.max(inp.view(-1, hidden_dim).abs().detach(), dim=0)[0].float().cpu()
        # print(comming_max)
        if self.inp_absmax is None:
            self.inp_absmax = comming_max
            self.inp_absmax2 = comming_max
            self.cnt = 1
        else:
            self.inp_absmax = self.inp_absmax.min( comming_max)
            self.inp_absmax2 = (self.inp_absmax+comming_max*self.cnt)/(self.cnt+1)
            self.cnt += 1
        self.layer.inp_absmax = self.inp_absmax #+ (self.inp_absmax2-self.inp_absmax)*0.2

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        self.inp1 = inp.squeeze()
        self.out1 = None
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D) or isinstance(self.layer, QuantLinear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def print_loss(self, name, q_weight, weight_error, timecost):
        table = Texttable()
        name += ' ' * (16 - len(name))

        table.header(['name', 'weight_error', 'fp_inp_SNR', 'q_inp_SNR', 'time'])

        # assign weight
        self.layer.weight.data = q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if self.out1 is not None:
            # quantize input to int8
            quantizer = Quantizer()
            quantizer.configure(8, perchannel=False, sym=True, mse=False)
            quantizer.find_params(self.inp1)
            q_in = quantizer.quantize(self.inp1).type(torch.float16)
            q_out = self.layer(q_in)

            # get kinds of SNR
            q_SNR = torch_snr_error(q_out, self.out1).item()
            fp_SNR = torch_snr_error(self.layer(self.inp1), self.out1).item()
        else:
            q_SNR = '-'
            fp_SNR = '-'

        table.add_row([name, weight_error, fp_SNR, q_SNR, timecost])
        print(table.draw().split('\n')[-2])


    def naivequant(self, groupsize=-1):
        self.method = 'naive'
        self.layer.to(self.dev)

        W = self.layer.weight.data.clone()
        org_shape = W.shape
        W = W.float()
        if groupsize >0:
            org_shape = W.shape
            tmp_W = W.view(-1, groupsize)
            self.quantizer.find_params(tmp_W, True)
            self.layer.weight.data = self.quantizer.quantize(tmp_W).to(self.layer.weight.data.dtype).view(org_shape)
        else:
            self.quantizer.find_params(W, weight=True)
            self.layer.weight.data = self.quantizer.quantize(W).to(self.layer.weight.data.dtype)

        scale = self.quantizer.scale.view(org_shape[0], -1)
        zero = self.quantizer.zero.view(org_shape[0], -1)
        return scale, zero

    def searchquant(self, groupsize=-1, W4W8=False):
        self.method = 'search'
        W = self.layer.weight.data.clone()
        org_shape = W.shape

        device, dtype = W.device, W.dtype
        if groupsize > 0:
            g_idx = [i // groupsize for i in range(org_shape[-1])]
            g_idx = torch.tensor(g_idx, dtype=torch.int32, device=device)
        else:
            g_idx = torch.tensor([])
        
        groupsize = groupsize if groupsize > 0 else org_shape[-1]

        grid = 20
        best_scale = torch.ones([W.shape[1] // groupsize, W.shape[0]],dtype=torch.bfloat16, device=device)
        best_zero = torch.ones([W.shape[1] // groupsize, W.shape[0]],dtype=torch.bfloat16, device=device)
        assert org_shape[1] % groupsize == 0
        assert self.quantizer.sym == False
        for nn in range(org_shape[1] // groupsize):
            W_t = W[:,nn*groupsize:(nn+1)*groupsize]
            inp_t = self.inp1[:,nn*groupsize:(nn+1)*groupsize]
            org_out = inp_t@(W_t.t())
            W_max = W_t.amax(dim=-1, keepdim=True)
            W_min = W_t.amin(dim=-1, keepdim=True)
            best = torch.full([W.shape[0]], float('inf'), device=device, dtype=dtype)
            for i in range(grid):
                ratio = 1.02 - (i+1) / grid*0.22
                W_t = W_t.clamp(W_min*ratio, W_max*ratio)
                qscale = (W_max*ratio - W_min*ratio) / self.quantizer.maxq
                qzero = torch.round(- W_min*ratio / qscale)
                qtensor = torch.clamp(torch.round(W_t/qscale)+qzero,0,self.quantizer.maxq)
                W_qt = qscale*(qtensor-qzero)
                out = inp_t@(W_qt.t())
                mse = (org_out - out).abs().pow(2).mean(dim=0).view(-1)
                best_idx = (best > mse).view(-1)
                best[best_idx] = mse[best_idx]
                best_scale[nn][best_idx] = qscale[best_idx].view(-1)
                best_zero[nn][best_idx] = qzero[best_idx].view(-1)          

        best_scale = best_scale.t()
        best_zero = best_zero.t()
        self.quantizer.scale = best_scale.reshape(-1, 1)
        self.quantizer.zero = best_zero.reshape(-1, 1)
        self.layer.weight.data = self.quantizer.quantize(W.view(-1, groupsize)).to(self.layer.weight.data.dtype).view(org_shape)
        best_scale8 = torch.zeros((W.shape[0],), dtype=torch.bfloat16, device=device)
        if W4W8:
            grid = 80
            # best_scale = torch.ones([W.shape[0], 1], dtype=torch.float16, device=device)
            org_out = self.inp1@W.t()
            best = torch.full([W.shape[0]], float('inf'), device=device, dtype=dtype)
            for i in range(grid):
                ratio = 1.02 - (i+1) / grid*0.82
                # W_max = torch.abs(W_t).max() * ratio
                # W_t = W.clamp(-W_max, W_max)
                W_max = W.abs().amax(dim=-1, keepdim=True) * ratio
                qscale_8 = W_max / (2 ** (8-1) - 1)
                qscale = torch.round(best_scale / qscale_8).clamp(min=1.)
                # qtensor = torch.clamp(torch.round(W_t/qscale)+qzero,0,self.quantizer.maxq)
                int_max = torch.round(127 / qscale)
                # upper = torch.minimum(15, best_zero+int_max)
                # lower = torch.maximum(0, best_zero-int_max)
                inp_t = self.inp1
                upper = torch.clamp(best_zero+int_max, max=15.).reshape(-1, 1)
                lower = torch.clamp(best_zero-int_max, min=0.).reshape(-1, 1)
                qscale_q = (qscale * qscale_8).reshape(-1, 1)
                W_t = W.view(-1, groupsize)
                q_tensor = torch.clamp(torch.round(W_t/qscale_q) + best_zero.reshape(-1, 1), lower, upper) 
                W_qt = qscale_q*(q_tensor-best_zero.reshape(-1, 1))
                W_qt = W_qt.view(org_shape)
                out = inp_t@(W_qt.t())
                mse = (org_out - out).abs().pow(2).mean(dim=0).view(-1)
                best_idx = (best > mse).view(-1)
                best[best_idx] = mse[best_idx]
                best_scale8[best_idx] = qscale_8[best_idx].view(-1) 
            best_scale = torch.round(best_scale / best_scale8.view(-1, 1)).clamp(min=1.)
            int_max = torch.round(127 / best_scale)
            best_scale_q = (best_scale * best_scale8.view(-1, 1)).reshape(-1, 1)
            upper = torch.clamp(best_zero+int_max, max=15.).reshape(-1, 1)
            lower = torch.clamp(best_zero-int_max, min=0.).reshape(-1, 1)
            q_tensor = torch.clamp(torch.round(W.view(-1, groupsize)/ best_scale_q) + best_zero.reshape(-1, 1), lower, upper)
            self.layer.weight.data = best_scale_q*(q_tensor-best_zero.reshape(-1, 1))
        self.inp1 = None
        return best_scale, best_zero, best_scale8

    def gptqquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, name=''):
        self.layer.to(self.dev)

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        if not self.observe:
            del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        error = torch.sum(Losses).item()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.print_loss(name=name, q_weight=Q, weight_error=error, timecost=(time.time() - tick))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx, error

    def free(self):
        self.inp1 = None
        self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
