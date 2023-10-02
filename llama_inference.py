# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import policies
from configs import fsdp_config, train_config, inference_config
from policies import AnyPrecisionAdamW

from utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset

from utils.train_utils import (
    test,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)


def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config((inference_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(inference_config.seed)
    torch.manual_seed(inference_config.seed)

    if inference_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)


    # Load the pre-trained model and setup its configuration
    if inference_config.enable_fsdp and inference_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                inference_config.model_name,
                load_in_8bit=True if inference_config.quantization else None,
                device_map="auto" if inference_config.quantization else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(inference_config.model_name)
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            inference_config.model_name,
            load_in_8bit=True if inference_config.quantization else None,
            device_map="auto" if inference_config.quantization else None,
        )
    if inference_config.enable_fsdp and inference_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, inference_config, rank if inference_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if inference_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if inference_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(inference_config.model_name)
    tokenizer.add_special_tokens(
            {

                "pad_token": "<PAD>",
            }
        )
    if inference_config.use_peft:
        peft_config = generate_peft_config(inference_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    if inference_config.enable_fsdp:
        if not inference_config.use_peft and inference_config.freeze_layers:

            freeze_transformer_layers(inference_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if inference_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=inference_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if inference_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not inference_config.quantization and not inference_config.enable_fsdp:
        model.to("cuda")

    dataset_config = generate_dataset_config(inference_config, kwargs)

     # Load and preprocess the dataset for testing
    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )

    if not inference_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_test)}")


    test_sampler = None
    val_sampler = None
    if inference_config.enable_fsdp:
        test_sampler = DistributedSampler(
            dataset_test,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if inference_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    # Create DataLoaders for the training and validation dataset
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=inference_config.batch_size_training,
        num_workers=inference_config.num_workers_dataloader,
        pin_memory=True,
        sampler=test_sampler if test_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )


    # Start the inference process
    results = test(
        model,
        test_dataloader,
        tokenizer,
        inference_config,
        local_rank if inference_config.enable_fsdp else None,
        rank if inference_config.enable_fsdp else None,
    )
    if not inference_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
