export CUDA_VISIBLE_DEVICES=0
python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name /work/valex1377/llama/models_hf/7B --output_dir /work/valex1377/llama/PEFT/model