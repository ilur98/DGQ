# generate quantized model
python -m dgq.entry /data/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348/ wikitext2 --wt_fun search --groupsize 128 --wbits 4 --smoothquant --w4w8 --kvquant --save_safetensors model_w4w8.safetensors

#evaluate quantized model
python -m dgq.entry /data/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348/ wikitext2 --wt_fun search --groupsize 128 --wbits 4 --smoothquant --w4w8 --kvquant --load model_w4w8.safetensors --eval