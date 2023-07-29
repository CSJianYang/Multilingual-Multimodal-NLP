# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
from typing import List, Union

import fire
import os.path as osp
from tqdm import tqdm

from transformers import (
    LlamaTokenizer,
)

from utils.dataset_utils import get_preprocessed_dataset

from utils.config_utils import (
    update_config,
    generate_dataset_config,
)
from configs import fsdp_config, train_config



def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)


    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
  

    dataset_config = generate_dataset_config(train_config, kwargs)
    
     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    
    for item in dataset_train:
        continue 
        # print(item["input_ids"].shape[0])


if __name__ == "__main__":
    fire.Fire(main)
