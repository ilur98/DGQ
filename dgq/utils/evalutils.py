import torch
import torch.nn as nn
from dgq.utils.modelutils import get_blocks, move_embed, move_norm_head

@torch.no_grad()
def model_eval(model, testenc, dev, local_args=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    # model = model.to(dev)
    model.eval()
    model.config.use_cache = False
    # testenc = testenc.to(dev)
    layers = get_blocks(model)
    layer_kwargs = {}
    cache={'i': 0}
    layers[0] = layers[0].to(dev)
    move_embed(model, dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    torch.cuda.memory_summary()
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
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module  # restore
    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    outs = torch.zeros_like(inps)
    torch.cuda.empty_cache()
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    mod_list = move_norm_head(model, dev)
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        for mod in mod_list:
            hidden_states = mod(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())