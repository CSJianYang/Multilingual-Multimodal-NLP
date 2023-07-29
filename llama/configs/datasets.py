# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import List 
    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass 
class mgsm_dataset:
    dataset: str = "mgsm_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/home/wangzixiang-b17/data/chai/XCoT/data/gsm_translate/scored_data/train_{lang}_scored.json"

@dataclass 
class mgsm_cw_dataset:
    dataset: str = "mgsm_cw_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/home/wangzixiang-b17/data/chai/XCoT/data/gsm_translate/scored_data/train_{lang}_scored.json"
    subword_align_path: str = "/home/wangzixiang-b17/data/chai/XCoT/align/align_result/train.align.en-{lang}.npy"
    phrase_align_path : str = "/home/wangzixiang-b17/data/chai/XCoT/align/phrase_align_result/train.phrase.align.en-{lang}.npy"
    langs: tuple = ('bn', 'de', 'es', 'fr', 'jp', 'ru', 'sw', 'te', 'th', 'zh')
    # langs: tuple = ('zh', )
    replace_percent_threshhold: float = 0.3 
    construct_phrase_level_align_dataset: bool = False
    examplar_cnt: int = 3
    switch_prob: float = 1
    subword_prob: float = 0
    max_span_length: int = 256
    max_switch_num: int = 10 
    
