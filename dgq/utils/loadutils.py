import torch
from dgq.quant.quant_linear import QuantLinear
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from dgq.models.opt_a8w4 import A8W4OPTForCausalLM
from dgq.models.llama_a8w4 import A8W4LlamaForCausalLM

def load_quant(model, checkpoint):
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        state_dict = model.state_dict()
        ckt = safe_load(checkpoint)
        for key in ckt.keys():
            try:
                state_dict[key].copy_(ckt[key])
            except Exception as e:
                print(key)
                print(e)
                pars = key.split('.')
                att = pars[-1]
                modname = '.'.join(pars[1:-1])
                for name,mod in model.named_modules():
                    if modname in name:
                        delattr(mod,att)
                        mod.register_buffer(att, ckt[key])
        # model.load_state_dict(ckt)
    else:
        model.load_state_dict(torch.load(checkpoint))

    for sublayer in model.modules():
        if isinstance(sublayer,QuantLinear):
            sublayer.prepare_actfun()  
            delattr(sublayer, "weight")


    model.seqlen = 2048
    print('Done.')
    return model



def inference_model(model):
    if isinstance(model, OPTForCausalLM):
        decoder_layer_scales = []
        for layer in model.model.decoder.layers:
            decoder_layer_scale = {"attn_input_scale": layer.self_attn.q_proj.amax.float() / (2 ** 7 - 1),
                                   "q_output_scale": layer.self_attn.q_quant.scale.float(),
                                   "k_output_scale": layer.self_attn.k_quant.scale.float(),
                                   "v_output_scale": layer.self_attn.v_quant.scale.float(),
                                   "out_input_scale": layer.self_attn.out_proj.amax.float() / (2 ** 7 - 1),
                                   "fc1_input_scale": layer.fc1.amax.float() / (2 ** 7 - 1),
                                   "fc2_input_scale": layer.fc2.amax.float() / (2 ** 7 - 1)}
            decoder_layer_scales.append(decoder_layer_scale)
        seqlen = model.seqlen
        model = A8W4OPTForCausalLM.from_float(model, decoder_layer_scales)
        model.seqlen = seqlen
    elif isinstance(model, LlamaForCausalLM):
        decoder_layer_scales = []
        for layer in model.model.layers:
            decoder_layer_scale = {"attn_input_scale": layer.self_attn.q_proj.amax.float() / (2 ** 7 - 1),
                                   "q_output_scale": layer.self_attn.q_quant.scale.float(),
                                   "k_output_scale": layer.self_attn.k_quant.scale.float(),
                                   "v_output_scale": layer.self_attn.v_quant.scale.float(),
                                   "out_input_scale": layer.self_attn.o_proj.amax.float() / (2 ** 7 - 1),
                                   "mlp_input_scale": layer.mlp.up_proj.amax.float() / (2 ** 7 - 1),
                                   "down_input_scale": layer.mlp.down_proj.amax.float() / (2 ** 7 - 1)}
            decoder_layer_scales.append(decoder_layer_scale)
        seqlen = model.seqlen
        model = A8W4LlamaForCausalLM.from_float(model, decoder_layer_scales)
        model.seqlen = seqlen
    else:
        raise NotImplementedError
    return model