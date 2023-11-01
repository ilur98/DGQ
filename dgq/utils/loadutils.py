import torch
from dgq.quant.quant_linear import QuantLinear
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


    model.seqlen = 2048
    print('Done.')

    return model
