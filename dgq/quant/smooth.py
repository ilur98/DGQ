import os
import random
import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from dgq.quant.quant_linear import QuantLinear
import torch.nn.functional as F

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype

    maxsv,inds = act_scales.sort()
    basl = int(len(act_scales)*0.005+1.5) # hyperparameter
    # basl = 128 # hyperparameter
    baseline =  maxsv[-basl]
    if baseline < 1e-4:
        return
    scales = act_scales/baseline
    scales[act_scales<=baseline] = 1.

    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0).cpu().float()
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    # weight_m = weight_scales[inds[-basl:]]
    # weight_redu = 16*weight_scales.max()/weight_m # hyperparameter 16
    scales[inds[-basl:]] = scales[inds[-basl:]]  #.min(weight_redu)
    ##
    act_scales /= scales
    scales = scales.to(device=device, dtype=dtype)
    ln.weight.div_(scales)
    if hasattr(ln,'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def smooth_ln_fcs_weight(ln, fcs):
    if not isinstance(fcs, list):
        fcs = [fcs]


    weight_scales = torch.cat([fc.weight.abs().mean(
        dim=0, keepdim=True) for fc in fcs], dim=0)
    scales = weight_scales/weight_scales.mean(dim=1, keepdim=True)
    for i in range(len(fcs)-1):

        scales[0] *= scales[i+1]
    scales = scales[0].float().pow(1/len(fcs)).clamp_(min=0.2,max=5).to(ln.weight.dtype)

    ln.weight.mul_(scales)
    if hasattr(ln,'bias') and ln.bias is not None:
        ln.bias.mul_(scales)

    for fc in fcs:
        fc.weight.div_(scales.view(1, -1))


@torch.no_grad()
def mean_ln_fcs(ln, fcs, act_median):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm) or isinstance(ln, LlamaRMSNorm)
    for fc in fcs:
        assert isinstance(fc, QuantLinear)
        assert ln.weight.numel() == fc.in_features == act_median.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    ##
    if hasattr(ln,'bias') and ln.bias is not None:
        ln.bias.sub_(act_median)
    else:
        # if ln.bias is None:
        #     delattr(ln,'bias')
        ln.register_buffer('bias',-act_median)
    ##
    for fc in fcs:
        if act_median is not None:
            tmp_bias = F.linear(act_median, fc.weight)
            if hasattr(fc, 'bias') and fc.bias is not None:
                fc.bias += tmp_bias
            else:
                if fc.bias is None:
                    delattr(fc,'bias')
                fc.register_buffer('bias',tmp_bias)

@torch.no_grad()
def smooth_att_qk(qproj, kproj, q_out_scales, k_out_scales):
    smooth_scale = (q_out_scales / k_out_scales).pow(0.5)
    qproj.weight.data.div_(smooth_scale.view(-1, 1))
    kproj.weight.data.div_(smooth_scale.view(-1, 1))

@torch.no_grad()
def smooth_llama_mlp(gatep,upp, downp, act_scales):

    def addchannel(layer,ind_c,nums,dir = 'in',rescale=False):
        if isinstance(nums,torch.Tensor):
            num = int(nums.item())-1
        if dir=='in':
            weight_temp = layer.weight.data.T
            layer.in_features += num
        else:
            weight_temp = layer.weight.data
            if rescale:
                weight_temp[ind_c] /= num+1
            layer.out_features += num
        newweight = torch.vstack([weight_temp]+[weight_temp[ind_c:ind_c+1]]*num)
        if dir=='in':
            layer.weight.data = newweight.T
        else:
            layer.weight.data = newweight
            if hasattr(layer,'bias') and layer.bias is not None:
                bias_temp = layer.bias.data
                if rescale:
                    bias_temp[ind_c] /= num+1
                layer.bias.data=torch.hstack([bias_temp]+[bias_temp[ind_c:ind_c+1]]*num)

    device, dtype = downp.weight.device, downp.weight.dtype

    downp_scales = downp.weight.abs().max(dim=0)[0].cpu().float().clamp(min=1e-5)

    maxsv,inds = act_scales.sort()
    basl = int(len(act_scales)*0.005+1.5) # hyperparameter
    baseline =  maxsv[-basl]
    if baseline < 1e-4:
        return
    scales = act_scales/baseline
    scales[act_scales<=baseline] = 1.
    downp_m = downp_scales[inds[-basl:]]
    downp_redu = 50*downp_scales.max()/downp_m
    scales[inds[-basl:]] = scales[inds[-basl:]]
    # print(scales.max())

    act_scales /= scales
    scales = scales.to(device=device, dtype=dtype)
    # gatep.weight.div_(scales)
    upp.weight.data.div_(scales.view(-1,1))

    if hasattr(upp,'bias') and upp.bias is not None:
        upp.bias.div_(scales)
    downp.weight.data.mul_(scales.view(1, -1))



    # for i in range(1,basl):
    #     splnum = (act_scales[inds[-i]]/act_scales[inds[-basl]]/1.5).round()
    #     if splnum>=2:
    #         print('split ',splnum)
    #         addchannel(gatep,inds[-i],splnum,'out')
    #         addchannel(upp,inds[-i],splnum,'out',rescale=True)
    #         addchannel(downp,inds[-i],splnum)
    #         act_scales[inds[-i]] /= splnum
# @torch.no_grad()
# def smooth_fc_weight(o_proj, v_proj, group_size=-1):
#     tmp_weight = o_proj.weight.data.clone()
#     org_shape = tmp_weight.shape
#     device, dtype = o_proj.weight.device, o_proj.weight.dtype

#     if group_size > 0:
#         tmp_weight = tmp_weight.view(-1, group_size)
#     weight_scales = tmp_weight.abs() / tmp_weight.abs().amax(dim=1, keepdim=True)
#     # best_scales = weight_scales / weight_scales.mean()
#     weight_scales = weight_scales.view(org_shape)
#     best_scales = weight_scales.amax(dim=0) / weight_scales.amax()
#     best_scales.clamp_(min=1e-4)
#     v_proj.weight.data = v_proj.weight.data / best_scales.view(-1, 1)
#     o_proj.weight.data = tmp_weight * best_scales.view(1, -1)

@torch.no_grad()
def smooth_fc_weight(v_proj, o_proj, group_size=-1,qkv=False):
    best_scales = o_proj.weight.data.abs().mean(dim=0)
    scales = best_scales/best_scales.mean()
    scales = scales.float().clamp_(min=0.2,max=5).to(v_proj.weight.dtype)
    # scales.clamp_(min=1e-4)

    n = o_proj.weight.shape[1]
    # if qkv:
    #     scales = scales.float().clamp_(min=0.5,max=2).to(v_proj.weight.dtype)

    if qkv:
        scales = scales.float().clamp_(min=0.2,max=5).to(v_proj.weight.dtype)
        v_proj.weight.data[2::3].mul_( scales.view(-1, 1))
        if hasattr(v_proj,'bias') and v_proj.bias is not None:
            v_proj.bias.data[2::3].mul_(scales.view(-1))
        o_proj.weight.data = o_proj.weight.data / scales.view(1, -1)
    else:
        v_proj.weight.data.mul_( scales.view(-1, 1))
        if hasattr(v_proj,'bias') and v_proj.bias is not None:
            v_proj.bias.data.mul_(scales.view(-1))
        o_proj.weight.data = o_proj.weight.data / scales.view(1, -1)



@torch.no_grad()
def smooth_ov(v_proj, o_proj, act_scales, qkv=False, alpha=0.7):
    device, dtype = v_proj.weight.device, v_proj.weight.dtype
    Num = o_proj.in_features
    o_scales = o_proj.weight.abs().max(dim=0)[0].cpu().float().clamp(min=1e-5)

    maxsv,inds = act_scales.sort()
    basl = int(Num*0.5+1.5) # hyperparameter
    baseline =  maxsv[-basl]
    if baseline < 1e-4:
        return
    scales = act_scales/baseline
    scales[act_scales<=baseline] = 1.
    #op_m = o_scales[inds[-basl:]]
    #op_redu = 32*o_scales.max()/op_m # hyperparameter 32
    scales[inds[-basl:]] = scales[inds[-basl:]]  #.min(op_redu)
    ##
    o_proj.weight.data.mul_(scales.to(device=device, dtype=dtype).view(1,-1))
    if qkv:
        scales = torch.hstack([torch.ones_like(scales),torch.ones_like(scales),scales])
    # scales = act_scales.pow(alpha) / o_scales.pow(1-alpha)
    if hasattr(v_proj, "v_absmax") and (v_proj.v_absmax is not None):
        v_proj.v_absmax.div_(scales)
    scales = scales.to(device=device, dtype=dtype)
    v_proj.weight.data.div_(scales.view(-1, 1))
    if hasattr(v_proj, "bias") and (v_proj.bias is not None):
        v_proj.bias.div_(scales)


@torch.no_grad()
def smooth_module(module, alpha=0.5, group_size=-1, weight_smooth=False, attention_mask=None, position_ids=None):
    if weight_smooth:
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj]
            # smooth_ln_fcs_weight(attn_ln, qkv) ##opt66b very bad...
            smooth_fc_weight(module.self_attn.v_proj, module.self_attn.out_proj, group_size)
            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            smooth_ln_fcs_weight(ffn_ln, fc1)
            smooth_fc_weight(module.fc1, module.fc2, group_size)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            smooth_ln_fcs_weight(attn_ln, qkv)
            v_proj = module.self_attention.query_key_value
            o_proj = module.self_attention.dense
            # smooth_fc_weight(v_proj, o_proj,qkv=True) ##bloom3b bad
            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            smooth_ln_fcs_weight(ffn_ln, fc1)
            # smooth_fc_weight(module.mlp.dense_4h_to_h, module.mlp.dense_h_to_4h, group_size)
        elif isinstance(module, LlamaDecoderLayer):
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            smooth_ln_fcs_weight(attn_ln, qkv)
            smooth_fc_weight(module.self_attn.v_proj, module.self_attn.o_proj, group_size)
            ffn_ln = module.post_attention_layernorm
            gate_proj = [module.mlp.gate_proj,module.mlp.up_proj]
            smooth_ln_fcs_weight(ffn_ln, gate_proj)
            smooth_fc_weight(module.mlp.up_proj, module.mlp.down_proj, group_size)
    else:
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = attn_ln.out_absmax
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            v_proj = module.self_attn.v_proj
            o_proj = module.self_attn.out_proj
            # smooth_ov(v_proj, o_proj, o_proj.inp_absmax)
            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = ffn_ln.out_absmax
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            # fc2 = module.fc2
            # fc2.inp_bias = ((fc2.inp_absmax )/2  ).clamp(min=0.).to(torch.float16)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = attn_ln.out_absmax
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            v_proj = module.self_attention.query_key_value
            o_proj = module.self_attention.dense
            # smooth_ov(v_proj, o_proj, o_proj.inp_absmax,qkv=True)
            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = ffn_ln.out_absmax
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            fc2 = module.mlp.dense_4h_to_h
            fc2.inp_bias = ((fc2.inp_absmax + 0.2)/2 - 0.2 ).clamp(min=0.).to(torch.float16)
        elif isinstance(module, LlamaDecoderLayer):
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = attn_ln.out_absmax
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            v_proj = module.self_attn.v_proj
            o_proj = module.self_attn.o_proj
            # smooth_ov(v_proj, o_proj, o_proj.inp_absmax)
            ffn_ln = module.post_attention_layernorm
            gate_proj = [module.mlp.gate_proj,module.mlp.up_proj]
            gate_proj_scales = ffn_ln.out_absmax
            smooth_ln_fcs(ffn_ln, gate_proj, gate_proj_scales, alpha)
            smooth_llama_mlp(module.mlp.gate_proj,module.mlp.up_proj,module.mlp.down_proj,module.mlp.down_proj.inp_absmax)
    for mod in module.modules():
        if hasattr(mod, 'inp_absmax'):
            delattr(mod, 'inp_absmax')
        if hasattr(mod, 'out_absmax'):
            delattr(mod, 'out_absmax')
        if hasattr(mod, 'inp_absmean'):
            delattr(mod, 'inp_absmean')
        if hasattr(mod, 'out_absmean'):
            delattr(mod, 'out_absmean')
@torch.no_grad()
def mean_bias(module):
    if isinstance(module, OPTDecoderLayer):
        attn_ln = module.self_attn_layer_norm
        qkv = [module.self_attn.q_proj,
                module.self_attn.k_proj, module.self_attn.v_proj]
        qkv_input_scales = (attn_ln.out_max + attn_ln.out_min) / 2
        mean_ln_fcs(attn_ln, qkv, qkv_input_scales)

        ffn_ln = module.final_layer_norm
        fc1 = module.fc1
        fc1_input_scales = (ffn_ln.out_max + ffn_ln.out_min) / 2
        mean_ln_fcs(ffn_ln, fc1, fc1_input_scales)
    elif isinstance(module, BloomBlock):
        attn_ln = module.input_layernorm
        qkv = module.self_attention.query_key_value
        qkv_input_scales = (attn_ln.out_max + attn_ln.out_min) / 2
        mean_ln_fcs(attn_ln, qkv, qkv_input_scales)

        ffn_ln = module.post_attention_layernorm
        fc1 = module.mlp.dense_h_to_4h
        fc1_input_scales = (ffn_ln.out_max + ffn_ln.out_min) / 2
        mean_ln_fcs(ffn_ln, fc1, fc1_input_scales)
    elif isinstance(module, LlamaDecoderLayer):
        attn_ln = module.input_layernorm
        qkv = [module.self_attn.q_proj,
                module.self_attn.k_proj, module.self_attn.v_proj]
        qkv_input_scales = (attn_ln.out_max + attn_ln.out_min) / 2
        mean_ln_fcs(attn_ln, qkv, qkv_input_scales)
        ffn_ln = module.post_attention_layernorm
        gate_proj = [module.mlp.gate_proj,module.mlp.up_proj]
        gate_proj_scales = (ffn_ln.out_max + ffn_ln.out_min) / 2
        mean_ln_fcs(ffn_ln, gate_proj, gate_proj_scales)
    for mod in module.modules():
        if hasattr(mod, 'out_max'):
            delattr(mod, 'out_max')
        if hasattr(mod, 'out_min'):
            delattr(mod, 'out_min') 

@torch.no_grad()
def catch_the_scale(fname, data):
    if not os.path.exists(fname):
        pkl_data = {}
        pkl_data['1'] = data 
        torch.save(pkl_data, fname)
    else:
        pkl_data = torch.load(fname)
        count = int(list(pkl_data.keys())[-1])+1
        pkl_data[str(count)] = data 
        torch.save(pkl_data, fname)