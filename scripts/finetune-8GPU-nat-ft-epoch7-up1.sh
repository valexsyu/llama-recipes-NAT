torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py \
         --dataset natsamsum_dataset \
         --enable_fsdp --model_name /work/valex1377/llama/models_hf/7B \
         --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder /ft-epoch7-up1 \
         --pure_bf16 --use_fast_kernels \
         --save_optimizer \
         --num_epochs 7
