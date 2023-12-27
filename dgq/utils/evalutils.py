import torch
import torch.nn as nn
from dgq.utils.modelutils import get_blocks, move_embed, move_norm_head
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from dgq.utils.datautils import IGNORE_INDEX, DEFAULT_PAD_TOKEN
import numpy as np


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

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def mmlu_eval(model, mmlu_dataset, data_loader, abcd_idx, dev, local_args=None):
    abcd_idx = abcd_idx
    model.eval()
    preds, refs = [], []
    loss_mmlu = 0
    cnt = 0    
    for batch in tqdm(data_loader, total=len(data_loader)):
        cnt += 1 
                         
        batch = to_device(batch, model.device)
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        labels = batch['labels']
        # There are two tokens, the output, and eos token.
        for i, logit in enumerate(logits):
            label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
            logit_abcd = logit[label_non_zero_id-1][abcd_idx]
            preds.append(torch.argmax(logit_abcd).item())
        labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
        refs += [abcd_idx.index(label) for label in labels.tolist()]
        loss_mmlu += loss.item()
    # Extract results by subject.
    results = {'mmlu_loss':loss_mmlu/len(data_loader)}
    subject = mmlu_dataset['subject']
    subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
    for s,p,r in zip(subject, preds, refs):
        subjects[s]['preds'].append(p)
        subjects[s]['refs'].append(r)
    subject_scores = []
    for subject in subjects:
        nn = len(subjects[subject]['refs'])
        subject_score =  0 if nn==0 else sum([subjects[subject]['refs'][ii] == subjects[subject]['preds'][ii] for ii in range(nn)])/nn
        results[f'accuracy_{subject}'] = subject_score
        subject_scores.append(subject_score)
    results[f'accuracy'] = np.mean(subject_scores)
    return results

@torch.no_grad()
def total_model_eval(model, testenc, dev, local_args=None):
    # testenc = testenc.cpu()
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    model = model.to(dev)
    model.eval()
    model.config.use_cache = False
    torch.cuda.memory_summary()
    model = model.to(dev)
    nlls = []
    for i in range(nsamples):
        print(i)
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        out = model(batch)['logits']
        shift_logits = out[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:].cuda()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).cpu()
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
        torch.cuda.empty_cache()
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())