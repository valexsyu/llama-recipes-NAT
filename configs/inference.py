# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class inference_config:
    model_name: str="PATH/to/LLAMA/7B"
    batch_size_training: int=1
    num_workers_dataloader: int=1
    seed: int=42
    use_fp16: bool=False
    dataset = "samsum_dataset"
    micro_batch_size: int=4
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    use_cahe: bool = False
    quantization: bool = False
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels

    
    
    
