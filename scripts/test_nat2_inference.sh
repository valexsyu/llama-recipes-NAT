python  llama_inference.py \
         --dataset natsamsum_dataset \
         --enable_fsdp --model_name /work/valex1377/llama/models_hf/finetune-7B-ft-7epoch\
         --pure_bf16 --use_fast_kernels \
         --num_epochs 7