# python inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/llama-recipes/model_checkpoints/fine-tuned-/work/valex1377/llama/models_hf/7B \
#        /work/valex1377/llama/models_hf/finetune-7B-nonlora-1epoch \
#        /work/valex1377/llama/models_hf/7B

python inference/checkpoint_converter_fsdp_hf.py \
       /work/valex1377/llama-recipes/model_checkpointing/PATH/to/save/FSDP/model/fine-tuned-/work/valex1377/llama/models_hf/7B \
       /work/valex1377/llama/models_hf/finetune-7B-nonlora-7epoch \
       /work/valex1377/llama/models_hf/7B