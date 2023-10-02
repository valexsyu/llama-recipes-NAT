torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py \
        --dataset natsamsum_dataset \
        --enable_fsdp --model_name /work/valex1377/llama/models_hf/7B \
        --use_peft --peft_method lora \
        --pure_bf16 --output_dir /work/valex1377/llama/PEFT/model-lora-epoch7-up1 \
        --num_epochs 7