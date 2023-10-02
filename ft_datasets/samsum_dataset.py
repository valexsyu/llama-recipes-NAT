# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
from .utils import Concatenator, Concatenator_NAT

def get_preprocessed_samsum(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("samsum", split=split)

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                dialog=sample["dialogue"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset

def get_preprocessed_natsamsum(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("samsum", split=split)
    upsampling_rate = 1
    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                dialog=sample["dialogue"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator_NAT(split, tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, upsampling_rate), batched=True, load_from_cache_file=False)
    return dataset

# def get_preprocessed_natsamsum(dataset_config, tokenizer, split):
#     upsampling_rate = 2 ###
#     dataset = datasets.load_dataset("samsum", split=split)

#     prompt = (
#         f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
#     )

#     def apply_prompt_template(sample):
#         return {
#             "text": prompt.format(
#                 dialog=sample["dialogue"],
#                 summary=sample["summary"],
#                 eos_token=tokenizer.eos_token,
#             )
#         }
#     def insert_unk(tokenizer, samples, upsampling_rate: int = 1,split='train'):
#         # Insert [UNK] token after each token
#         tokens_samples = tokenizer(samples)
#         if split != 'train':
#             return tokens_samples 
#         for i, (tokens ,attention_masks) in enumerate(zip(tokens_samples['input_ids'],tokens_samples['attention_mask'])):
#             modified_tokens = [tokenizer.bos_token_id]
#             for token in tokens[1:-1]:
#                 modified_tokens.extend([token] + [tokenizer.unk_token_id] * upsampling_rate)
#             modified_tokens.extend([tokenizer.eos_token_id])
#             tokens_samples['input_ids'][i] = modified_tokens
#             tokens_samples['attention_mask'][i] = attention_masks[1:] * upsampling_rate
#         return tokens_samples

#     dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
#     dataset = dataset.map(
#         lambda sample: tokenizer(sample["text"]),
#         # lambda sample: insert_unk(tokenizer,sample["text"],upsampling_rate),
#         batched=True,
#         remove_columns=list(dataset.features),
#     ).map(Concatenator_NAT(split=split, unk=tokenizer.unk_token_id, 
#                            eos=tokenizer.eos_token_id, upsampling_rate=upsampling_rate), batched=True)
#     return dataset

