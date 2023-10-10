# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys

import fire
from fire import parser
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

from utils.test_utils import (
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


    model = LlamaForCausalLM.from_pretrained(
        inference_config.model_name,
        load_in_8bit=True ,
        device_map="auto" ,
    )
            
    print_model_size(model, inference_config)

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



    dataset_config = generate_dataset_config(inference_config, kwargs)

     # Load and preprocess the dataset for testing
    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )


    print(f"--> Training Set Length = {len(dataset_test)}")


    test_sampler = None
    val_sampler = None

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
    )

    # [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
