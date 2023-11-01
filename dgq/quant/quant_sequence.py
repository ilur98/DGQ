import torch
import torch.nn as nn

from dgq.quant.smooth_hooker import prepare_hook
from dgq.quant.smooth import mean_bias, smooth_module
from dgq.quant.quant_linear import QuantLinear
from dgq.quant.quantizer_helper import QuantizerHelper
from dgq.quant.kvquanter import kvquant
from dgq.utils.modelutils import find_layers, move_embed, get_blocks

__all__ = ["quant_sequential"]

def set_quant_state(module, actq, wtq):
    for mod in module.modules():
        if isinstance(mod, QuantLinear):
            mod.setquant(actq, wtq)
@torch.no_grad()
def PTQ(model, enc, 
        qconfig, 
        nsamples=128, seqlen=2048):
    dev = "cuda:0"
    layers = get_blocks(model)
    layer_kwargs = {}
    cache={'i': 0}
    layers[0] = layers[0].cuda()
    move_embed(model, dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    outs = torch.zeros_like(inps)
    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            layer_kwargs.update(kwargs)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in enc:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    del enc
    layers[0] = layers[0].module  # restore
    # inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        full = find_layers(layer, [QuantLinear])
        sequential = [list(full.keys())]
        set_quant_state(layer, False, False)
        prepare_hook(layer, inps, qconfig, layer_kwargs)
        if qconfig["meanact"]:
            mean_bias(layer)
        if qconfig["smoothquant"]:
            smooth_module(layer)
        if qconfig["kvquant"]:
            kvquant(layer)
        for names in sequential:
            subset = {n: full[n] for n in names}
            helpers = {}
            for name in subset:
                helpers[name] = QuantizerHelper(subset[name])
                helpers[name].quantizer.configure(qconfig["wt_quant"]["bits"], perchannel=True, sym=False, mse=False)
                def add_batch(name):
                    def tmp(_, inp, out):
                        helpers[name].add_batch(inp[0].data, out.data)

                    return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()
            
            for name in subset:
                if qconfig["wt_quant"]["method"] == "gptq":
                    scale, zero = helpers[name].gptqquant(percdamp=qconfig["percdamp"], groupsize=qconfig["wt_quant"]["groupsize"], actorder=qconfig["act_order"], name=name)
                elif qconfig["wt_quant"]["method"] == "search":
                    scale, zero, scale8 = helpers[name].searchquant(groupsize=qconfig["wt_quant"]["groupsize"], W4W8=qconfig["wt_quant"]["w4w8"])
                elif qconfig["wt_quant"]["method"] == "naive":
                    scale, zero = helpers[name].naivequant(groupsize=qconfig["wt_quant"]["groupsize"])
                else:
                    raise NotImplemented
                if qconfig["wt_quant"]["w4w8"]:
                    subset[name].packW4W8(scale, zero, scale8)
                else:
                    subset[name].pack(scale, zero)
                if qconfig["act_quant"] is not None:
                    clamp = subset[name].inp_absmax.max()
                    subset[name].amax = clamp
                    delattr(subset[name], "inp_absmax")
                subset[name].prepare_actfun()
                helpers[name].free()
        set_quant_state(layer, qconfig['act_quant'] != None, qconfig['wt_quant'] != None) 
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        # del helpers
        torch.cuda.empty_cache()

        inps, outs = outs, inps