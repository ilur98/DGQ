import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import lm_eval
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from texttable import Texttable
from dgq.quant.quant_sequence import PTQ
from dgq.utils.datautils import get_loaders, prepare_mmlu
from dgq.utils.evalutils import model_eval, total_model_eval, mmlu_eval
from dgq.utils.loadutils import load_quant, inference_model
from dgq.utils.modelutils import convert_model

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='llama model to load')
parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
parser.add_argument('--nsamples', type=int, default=18, help='Number of calibration data samples.')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--wbits', type=int, default=4, choices=[2, 3, 4, 8, 16], help='#bits to use for weight quantization; use 16 for evaluating base model.')
parser.add_argument('--abits', type=int, default=8, choices=[8, 16], help='#bits to use for activation quantization; use 16 for evaluating base model.')
parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')

parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
parser.add_argument('--load', type=str, default='', help='Load quantized model.')

parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
parser.add_argument('--act_fun', type=str, default='static', help='activation quantization.')
parser.add_argument('--wt_fun', type=str, default='naive', help='weight quantization.')
parser.add_argument('--smoothquant', action='store_true', help='whether to   ')
parser.add_argument('--kvquant', action='store_true', help='whether to   ')
parser.add_argument('--meanact', action='store_true', help='whether to   ')    
parser.add_argument('--observe', action='store_true', help='whether to   ') 
parser.add_argument('--nearest', action='store_true', help='whether to   ')
parser.add_argument('--w4w8', action='store_true', help='wheter to open dual grained quantization')

parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
parser.add_argument('--mmlu_eval', type=str, default="no", help="mmlu evaluation.")
parser.add_argument('--csqa_eval', type=str, default="no", help="csqa evaluation.")
parser.add_argument('--inference_mod', action='store_true', help='whether to   ')
args = parser.parse_args()


def generate_qconfig(args):
    qconfig = {}
    if args.act_fun == "no":
        qconfig["act_quant"] = None
    else:
        act_qconfig = {}
        act_qconfig["bits"] = args.abits
        act_qconfig["method"] = args.act_fun
        qconfig["act_quant"] = act_qconfig
    
    if args.wt_fun == "no":
        qconfig["wt_quant"] = None
    else:
        wt_qconfig = {}
        wt_qconfig["bits"] = args.wbits
        wt_qconfig["method"] = args.wt_fun
        wt_qconfig["groupsize"] = args.groupsize
        wt_qconfig["w4w8"] = hasattr(args, "w4w8") and args.w4w8
        qconfig["wt_quant"] = wt_qconfig
    qconfig["smoothquant"] = hasattr(args, "smoothquant") and args.smoothquant
    qconfig["meanact"] = hasattr(args, "meanact") and args.meanact
    qconfig["kvquant"] = hasattr(args, "kvquant") and args.kvquant

    return qconfig

def prepare_model(model, seqlen=2048):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.seqlen = seqlen

    return model

def main():
    model = prepare_model(args.model)
    qconfig = generate_qconfig(args)
    convert_model(model, qconfig)
    print(args)
    enc, _ = get_loaders(args.dataset, args.nsamples, model=args.model)
    if args.load:
        load_quant(model, args.load)
        if hasattr(args, "inference_mod"):
            model = inference_model(model)
    else:
        tick = time.time()
        PTQ(model, enc, qconfig, args.nsamples)
        print(time.time() - tick)
        if args.save_safetensors:
            model = model.cpu()
            from safetensors.torch import save_file as safe_save
            state_dict = model.state_dict()
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            safe_save(state_dict, args.save_safetensors)
        if args.save:
            model = model.cpu()
            torch.save(model.state_dict(), args.save)
    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
        for dataset in datasets:
            _, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            # model_eval(model, testloader, torch.device('cuda:0'), local_args=args)
            total_model_eval(model, testloader, torch.device('cuda:0'), local_args=args)
    if args.mmlu_eval != 'no':
        model = model.to(torch.device('cuda:0'))
        dataset_test, testloader, abcd_idx = prepare_mmlu(args.model, args.mmlu_eval)
        result = mmlu_eval(model, dataset_test, testloader, abcd_idx, torch.device('cuda:0'), local_args=args)
        print(result)
if __name__ == '__main__':
    main()
