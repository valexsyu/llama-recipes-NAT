# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset

class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result
    
class Concatenator_NAT(object):
    def __init__(self,split, unk, bos, eos, upsampling_rate, chunk_size=2048):
        self.chunk_size = chunk_size
        self.split = split
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.upsampling_rate = upsampling_rate      
        self.residual = {"input_ids": [], "attention_mask": []}
        
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }
        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])
        extend_input_ids = concatenated_samples['input_ids'] + [concatenated_samples['input_ids'][-1]] * self.upsampling_rate
        extend_attention_masks = concatenated_samples['attention_mask'] + [concatenated_samples['attention_mask'][-1]] * self.upsampling_rate
        
        if self.split == 'train' or self.split == 'validation' :
            insert_unk_samples = {"input_ids": [], "attention_mask": [], "labels":[]}
            for i, (token, attention) in enumerate(zip(extend_input_ids, extend_attention_masks)):
                if i == len(extend_input_ids)-self.upsampling_rate :
                    break
                insert_unk_tokens = [token] + [self.unk] * self.upsampling_rate
                insert_attention_mask = extend_attention_masks[i+1:i+2+self.upsampling_rate]
                insert_labels = extend_input_ids[i+1:i+2+self.upsampling_rate]
                index = insert_labels.index(self.eos) if self.eos in insert_labels else -1
                if index != -1:
                    insert_labels[index:] = [self.eos] * (len(insert_labels) - index)
                    insert_attention_mask[index:] = [insert_attention_mask[index]] * (len(insert_labels) - index)
                
                if token not in [self.unk, self.bos, self.eos] :
                    insert_unk_samples["input_ids"].extend(insert_unk_tokens)
                    insert_unk_samples["attention_mask"].extend(insert_attention_mask)
                    insert_unk_samples["labels"].extend(insert_labels)
                else:
                    insert_unk_samples["input_ids"].extend([token])
                    insert_unk_samples["attention_mask"].extend([attention]) 
                    insert_unk_samples["labels"].extend([extend_input_ids[i+1]]) 
        
            complement_unk_samples= {"input_ids": [], "attention_mask": [], "labels":[]}
            j = 1
            span_period = False
            
            for i, (token, attention, label) in enumerate(zip(insert_unk_samples['input_ids'],insert_unk_samples['attention_mask'],insert_unk_samples['labels'])):
                current_size = len(complement_unk_samples["input_ids"])
                if token not in [self.unk] : #, self.bos, self.eos] :
                    current_token_span = insert_unk_samples['input_ids'][i:i+self.upsampling_rate+1]
                    current_attention_span = insert_unk_samples['attention_mask'][i:i+self.upsampling_rate+1]
                    current_label_span = insert_unk_samples['labels'][i:i+self.upsampling_rate+1]
                    span_period = False
                if current_size % self.chunk_size == 0 and token == self.unk:
                    span_period = True
                    complement_unk_samples['input_ids'].extend(current_token_span)
                    complement_unk_samples['labels'].extend(current_label_span)
                    complement_unk_samples["attention_mask"].extend(current_attention_span) 
                elif not span_period:
                    complement_unk_samples["input_ids"].extend([token])
                    complement_unk_samples["labels"].extend([label])    
                    complement_unk_samples["attention_mask"].extend([attention])  

            concatenated_samples = complement_unk_samples   

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])
        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            for k, v in self.residual.items():
                if k != "labels":
                    concatenated_samples[k] = v + list(chain(*batch[k]))            
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys() if k != "labels"}

        if self.split == 'test':
            result["labels"] = result["input_ids"].copy()
        
        return result            
        
        
        # concatenated_samples = {
        #     k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        # }
        # total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])
        # breakpoint()
        # if self.split == 'train' :
        #     insert_unk_samples = {"input_ids": [], "attention_mask": [], "labels":[]}
        #     for i, (token, attention) in enumerate(zip(concatenated_samples['input_ids'],concatenated_samples['attention_mask'])):
        #         if token not in [self.unk, self.bos, self.eos] :
        #             insert_unk_samples["input_ids"].extend([token] + [self.unk] * self.upsampling_rate)
        #             insert_unk_samples["attention_mask"].extend([attention] * (self.upsampling_rate+1))
        #             insert_unk_samples["labels"].extend(concatenated_samples['input_ids'][i+1:i+2+self.upsampling_rate])
        #         else:
        #             insert_unk_samples["input_ids"].extend([token])
        #             insert_unk_samples["attention_mask"].extend([attention]) 
        #             insert_unk_samples["labels"].extend([token])    
        #     complement_unk_samples= {"input_ids": [], "attention_mask": [], "labels":[]}
        #     j = 1
        #     for i, (token, attention, label) in enumerate(zip(insert_unk_samples['input_ids'],insert_unk_samples['attention_mask'],insert_unk_samples['labels'])):
        #         current_size = len(complement_unk_samples["input_ids"])
        #         if current_size == j * self.chunk_size and token == self.unk :
        #             j = j+1
        #             complement_unk_samples['input_ids'].extend(insert_unk_samples['labels'][i-1:i] + [token])
        #             complement_unk_samples['labels'].extend(insert_unk_samples['labels'][i-1:i] + [label])
        #             complement_unk_samples["attention_mask"].extend(insert_unk_samples['attention_mask'][i-1:i] + [attention]) 
        #         else:
        #             complement_unk_samples["input_ids"].extend([token])
        #             complement_unk_samples["labels"].extend([label])    
        #             complement_unk_samples["attention_mask"].extend([attention]) 
        #     concatenated_samples = complement_unk_samples   

        # if total_length >= self.chunk_size:
        #     chunk_num = total_length // self.chunk_size
        #     result = {
        #         k: [
        #             v[i : i + self.chunk_size]
        #             for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
        #         ]
        #         for k, v in concatenated_samples.items()
        #     }
        #     for k, v in self.residual.items():
        #         if k != "labels":
        #             concatenated_samples[k] = v + list(chain(*batch[k]))            
        # else:
        #     result = concatenated_samples
        #     self.residual = {k: [] for k in concatenated_samples.keys() if k != "labels"}


        # if self.split != 'train' :
        #     result["labels"] = result["input_ids"].copy()
       
        # return result
    
# class Insert_unk(object):
#     def __init__(self, split, unk, bos, eos, upsampling_rate):
#         self.split = split
#         self.unk = unk
#         self.bos = bos
#         self.eos = eos
#         self.upsampling_rate = upsampling_rate      
#     def __call__(self, samples):
#         if self.split != 'train':
#             return  
#         for i, (tokens , attention_masks, extend_input_ids) in enumerate(zip(samples['input_ids'],samples['attention_mask'],samples['extend_input_ids'])):
#             modified_tokens = []
#             modified_attention = []
#             modified_labels = []
#             for j, token in enumerate(tokens):
#                 if token not in [self.unk, self.bos, self.eos] :
#                     modified_tokens.extend([token] + [self.unk] * self.upsampling_rate)
#                     modified_attention.extend([attention_masks[j]] * (self.upsampling_rate+1))
#                     # modified_labels.extend(extend_input_ids[j:j+self.upsampling_rate+1])
#                 else:
#                     modified_tokens.extend([token])
#                     modified_attention.extend([attention_masks[j]])
#                     modified_labels.extend([token])
#             samples['input_ids'][i] = modified_tokens
#             samples['attention_mask'][i] = modified_attention
        
        
#         return samples
    
# class Concatenator_NAT(object):
#     def __init__(self, split, unk, eos, upsampling_rate, chunk_size=2048):
#         self.chunk_size=chunk_size
#         self.residual = {"input_ids": [], "attention_mask": []}
#         self.split = split
#         self.unk = unk
#         self.eos = eos
#         self.upsampling_rate = upsampling_rate
        
#     def __call__(self, batch):
        
#         def concate(concatenated_samples, chunk_size, chunk_num):
#             result = {
#                 k: [
#                     v[i : i + chunk_size]
#                     for i in range(0, chunk_num * chunk_size, chunk_size)
#                 ]
#                 for k, v in concatenated_samples.items()
#             }
#             residual = {
#                 k: v[(chunk_num * chunk_size) :]
#                 for k, v in concatenated_samples.items()
#             }
#             return result, residual 
                       
                        
#         concatenated_samples = {
#             k: v + list(chain(*batch[k])) for k, v in self.residual.items()
#         }
#         total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])
        
#         if total_length >= self.chunk_size:
#             chunk_num = total_length // self.chunk_size
#             result, self.residual = concate(concatenated_samples, self.chunk_size, chunk_num)
#         else:
#             result = concatenated_samples
#             self.residual = {k: [] for k in concatenated_samples.keys()}
            
        
#         if self.split != 'train' :
#             result["labels"] = result["input_ids"].copy()
#         else:
#             concatenated_labels={'input_ids':[]}
#             next_copy_token = []
#             for i, token in enumerate(concatenated_samples['input_ids']):                  
#                 if token == self.unk :
#                     if next_copy_token == [] :
#                         import pdb;pdb.set_trace()
#                         print("Error")
#                     elif next_copy_token[0] == self.eos :
#                         concatenated_labels['input_ids'].extend([self.eos])
#                     else:
#                         concatenated_labels['input_ids'].extend([next_copy_token.pop(0)])
#                 else:
#                     concatenated_labels['input_ids'].extend([concatenated_samples['input_ids'][i]])    
#                     next_copy_token = [
#                         concatenated_samples['input_ids'][min((self.upsampling_rate + 1) * (j + 1) + i, (total_length - 1))]
#                         for j in range(self.upsampling_rate)
#                     ]

#             result_label, _ = concate(concatenated_labels, self.chunk_size, chunk_num)            
#             result["labels"] = result_label["input_ids"].copy()
            
#         return result

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        
        self.samples = []
        
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }
        
        for sample in tqdm(self.dataset, desc="Preprocessing dataset"):
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
                
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)