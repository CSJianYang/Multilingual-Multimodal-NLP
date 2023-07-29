# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

class MGSMDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=256):
        self.langs = ['bn', 'de', 'es', 'fr', 'jp', 'ru', 'sw', 'te', 'th', 'zh']
        self.ann = []

        # load en data 
        file_path = dataset_config.data_path.format_map({'lang':'zh'})
        temp = json.load(open(file_path,))
        for item in temp:
            self.ann.append({
                'question': item['question_en'],
                'answer'  : item['answer_en'],
                'lang':'en',
            })

        # load other 10 lang data.
        # /Users/chai/Desktop/XCoT/data/gsm_translate/scored_data/train_bn_scored.json
        for lang in self.langs:
            file_path = dataset_config.data_path.format_map({'lang':lang})
            for item in json.load(open(file_path)):
                item['lang'] = lang
                self.ann.append(item)
         
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]
        self.max_words = max_words
        self.tokenizer = tokenizer
  
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        prompt = ann['question']
        example = prompt + ann["answer"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )

        print(ann['lang'], example.shape[0])
        
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
