# python inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/llama-recipes/model_checkpoints/fine-tuned-/work/valex1377/llama/models_hf/7B \
#        /work/valex1377/llama/models_hf/finetune-7B-nonlora-1epoch \
#        /work/valex1377/llama/models_hf/7B


#     fsdp_checkpoint_path="", # Path to FSDP Sharded model checkpoints
#     consolidated_model_path="", # Path to save the HF converted model checkpoints
#     HF_model_path_or_name="" # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
# python inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/llama-recipes/model_checkpoints/ft_e7_up1-/work/valex1377/llama/models_hf/7B \
#        /work/valex1377/llama/models_hf/finetune-7B-ft-1epoch \
#        /work/valex1377/llama/models_hf/7B
python inference/checkpoint_converter_fsdp_hf.py \
       /work/valex1377/llama-recipes/model_checkpoints/ft_e7_up1-last-/work/valex1377/llama/models_hf/7B \
       /work/valex1377/llama/models_hf/finetune-7B-ft-7epoch \
       /work/valex1377/llama/models_hf/7B