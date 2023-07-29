import copy
import json
import os
import torch
import random
import numpy as np  
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List
from ft_datasets.align_dataset import AlignDataset
from transformers import AutoTokenizer
random.seed(2023)

def xlm_tok_sen(text, tokenizer):
    encoded_input = tokenizer(text)
    input_ids = encoded_input['input_ids']  
    toks = tokenizer.convert_ids_to_tokens(input_ids)[1:-1]
    return toks 



class MGSMCodeSwitchDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=256):
        self.dataset_config = dataset_config 
        self.langs = dataset_config.langs
        
        self.lang_template = {
            'en':['Question : ','Step-by-Step Answer : ', 'The answer is {number}.'],
            'zh':['问题 : ','逐步解答 : ', '答案是 {number}。'],
            'de':['Frage : ','Schrit-für-Schrit-Antwor : ', 'Die Antwor lautet {number}.'],
            'fr':['Question : ','Réponse étape par étape : ', 'La réponse est {number}.'],
            'bn':['প্রশ্ন : ', 'ধাপে ধাপেউত্তর : ', 'উত্তর হল {number}।'],
            'jp':['問題 : ', 'ステップごとの答え : ', '答えは {number} です。'],
            'ru':['Задача : ', 'Пошаговое решение : ', 'Ответ — {number}.'],
            'es':['Pregunta : ', 'Respuesta paso a paso : ', 'La respuesta es {number}.'],
            'sw':['Swali : ', 'Jibu la Hatua kwa Hatua : ', 'Jibu ni {number}.'],
            'te':['ప్రశ్న : ', 'దశలవారీగా సమాధానం : ', 'సమాధానం {number}.'],
            'th':['โจทย์ : ', 'คำตอบทีละขั้นตอน : ', 'คำตอบคือ {number}'],
        }

        self.xlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        self.ann = []
        self.align_dict = {}
        
        # load en data 
        file_path = dataset_config.data_path.format_map({'lang':'zh'})
        self.en_ann = []
        for idx, item in enumerate(json.load(open(file_path,))):
            en_item = {
                'idx': idx, 
                'question': item['question_en'],
                'answer'  : item['answer_en'],
                'question_toks': xlm_tok_sen(item['question_en'], self.xlm_tokenizer),
                'answer_toks': xlm_tok_sen(item['answer_en'], self.xlm_tokenizer),
                'lang':'en',
                'number_answer': int(item['answer_en'].split('####')[1].strip().replace(',', ''))
            }
            self.en_ann.append(en_item)
            self.ann.append(en_item)


        # load other 10 lang data.
    
        for lang in self.langs:
            file_path = dataset_config.data_path.format_map({'lang':lang})
            for idx, item in enumerate(json.load(open(file_path))):
                item['lang'] = lang
                item['idx'] = idx 
                item['question_toks'] = xlm_tok_sen(item['question'], self.xlm_tokenizer)
                item['answer_toks'] = xlm_tok_sen(item['answer'], self.xlm_tokenizer)
                item['number_answer'] = int(item['answer_en'].split('####')[1].strip().replace(',', ''))
                self.ann.append(item)
            subword_align_path = dataset_config.subword_align_path.format_map({'lang':lang})
            phrase_align_path  = dataset_config.phrase_align_path.format_map({'lang':lang})
            
            subword_aligns  = np.load(subword_align_path,  allow_pickle=True)  
            phrase_aligns   = np.load(phrase_align_path,   allow_pickle=True) 
            align_dataset = AlignDataset(subword_align_dataset=subword_aligns, phrase_align_dataset=phrase_aligns)
            self.align_dict[lang] = align_dataset
        
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def build_incontext(self, ann_index):
        # sample examplar which index != ann_index
        all_idxs = list(range(len(self.ann)))
        all_idxs.remove(ann_index)
        
        examplar_idxs = random.sample(all_idxs, k=self.dataset_config.examplar_cnt)
        in_contexts = []
        for idx in examplar_idxs:
            examplar = self.ann[idx]
            question_item = examplar['question_toks']
            answer_item   = examplar['answer_toks']
            idx  = examplar['idx']
            lang = examplar['lang']
            number_answer = examplar['number_answer']
            
            if lang == 'en':
                in_contexts.append([lang, examplar['question'], examplar['answer'], number_answer])
                continue 

            en_question_item  = self.en_ann[idx]['question_toks']
            en_answer_item    = self.en_ann[idx]['answer_toks']
            
            if self.align_dict is not None:
                if self.dataset_config.construct_phrase_level_align_dataset:
                    question_item = self.pattern_prompt[idx]
                    answer_item   = self.pattern_answer[idx]
                else:
                    subword_aligns, phrase_aligns = self.align_dict[lang][idx]
                    if subword_aligns is not None:
                        subword_question_aligns, subword_answer_aligns = subword_aligns
                        subword_question_aligns = subword_question_aligns.tolist()
                        subword_answer_aligns   = subword_answer_aligns.tolist()
                    if phrase_aligns is not None:
                        phrase_question_aligns, phrase_answer_aligns = phrase_aligns
                        phrase_question_aligns = phrase_question_aligns.tolist()
                        phrase_answer_aligns   = phrase_answer_aligns.tolist()                       
                    
                    codeswitched_question_item = self.prepare_phrase_level_pattern_based_sentence(subword_question_aligns, phrase_question_aligns, question_item, en_question_item)
                    
                    codeswitched_answer_item = self.prepare_phrase_level_pattern_based_sentence( subword_answer_aligns, phrase_answer_aligns, answer_item, en_answer_item)

                    cs_question = ''.join(codeswitched_question_item).replace('▁▁', '▁').replace('▁', ' ')
                    cs_answer   = ''.join(codeswitched_answer_item  ).replace('▁▁', '▁').replace('▁', ' ')

                    in_contexts.append([lang, cs_question, cs_answer, number_answer])
        


        # use template build context
        in_context_str = []
        for lang, question, answer, number_answer in in_contexts:
            template = self.lang_template[lang]
            # Question: ... \n Answer step-by-step: ... The Answer is 
            answer = answer.split('#')[0]  # remove number result at the end of the answer.
            temp = f"{template[0]}{question}\n{template[1]}{answer} {template[2].format_map({'number':number_answer})}"
            
            in_context_str.append(temp)
        in_context_str = "\n\n".join(in_context_str)
        return in_context_str

                    
    def __getitem__(self, index):
        ann = self.ann[index]

        question = ann['question']
        answer   = ann['answer']
        number_answer = ann['number_answer']
        
        lang = ann['lang']
        template = self.lang_template[lang]
        in_context = self.build_incontext(ann_index=index)

        # print(in_context, end='\n\n')
        prompt = f"{in_context}\n\n{template[0]}{question}\n{template[1]}"

        answer = answer.split('#')[0]  # remove number result at the end of the answer.
        answer = f'{answer} {template[2].format_map({"number":number_answer})}'
        
        print('########### prompt: ', prompt)
        print("*********** answer: ", answer)
        print()

        example = prompt + answer
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        
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
    
    def choose_aligns(self, aligns):
        mask_num = min(len(aligns), random.randint(0, self.dataset_config.max_switch_num))
        align_idxs = np.random.choice(np.arange(len(aligns)), mask_num, replace=False)
        mask_aligns = [aligns[align_idx] for align_idx in align_idxs]
        return mask_aligns


    def remove_overlapped_aligns(self, phrases, src_item, tgt_item):
        if len(phrases) <= 1:
            return phrases
        span_mask_aligns = []  # remove intersected phrases
        if isinstance(phrases[0][0], int):
            span_mask_aligns = list({phrase[0]: phrase[1] for phrase in phrases}.items())
            span_mask_aligns = list({span_mask_align[1]: span_mask_align[0] for span_mask_align in span_mask_aligns}.items())
            span_mask_aligns = [[span_mask_align[1], span_mask_align[0]] for span_mask_align in span_mask_aligns]
            span_mask_aligns = list(span_mask_aligns)
        else:
            # src mask
            src_masks = torch.zeros(len(src_item), dtype=torch.bool)
            # tgt mask
            tgt_masks = torch.zeros(len(tgt_item), dtype=torch.bool)
            for span_mask_align in phrases:
                src_span_mask_align = span_mask_align[0]
                tgt_span_mask_align = span_mask_align[1]
                if not src_masks[src_span_mask_align[0]: src_span_mask_align[1]].any() and not tgt_masks[tgt_span_mask_align[0]: tgt_span_mask_align[1]].any():
                    span_mask_aligns.append(span_mask_align)
                    src_masks[src_span_mask_align[0]: src_span_mask_align[1]] = True
                    tgt_masks[tgt_span_mask_align[0]: tgt_span_mask_align[1]] = True
        return span_mask_aligns

    def prepare_phrase_level_pattern_based_sentence(self, aligns, phrases, src_item, tgt_item):
        if random.random() < self.dataset_config.switch_prob:  # ensure 1 alignment at least, switch_prob
            if random.random() <= self.dataset_config.subword_prob:
                if len(aligns) == 0 or len(aligns[0]) != 2:
                    return src_item, tgt_item
             
                mask_aligns = self.choose_aligns(aligns)
                mask_aligns = self.remove_overlapped_aligns(mask_aligns, src_item, tgt_item)
                # print(mask_aligns)
                replace_length = 0 
                origin_src_length = len(src_item)
                mask_aligns = sorted(mask_aligns, key=lambda x: x[0],reverse=True)
        
                for align in mask_aligns:
                    src_align, tgt_align = align 
                    # print(src_item[src_align], tgt_item[tgt_align])
                    src_item[src_align] = tgt_item[tgt_align]+'▁'
                    replace_length+= 1
                    if replace_length/origin_src_length > self.dataset_config.replace_percent_threshhold:
                        break 
            else:
                if len(phrases) == 0 or len(phrases[0]) != 2:
                    return src_item, tgt_item
                # Phrase level mask
                phrases = list(filter(lambda phrase: phrase[0][1] - phrase[0][0] <= self.dataset_config.max_span_length, phrases))
                # phrases = self.save_word_based_phrase(phrases, src_item, tgt_item)
                phrases.append([[1, len(src_item) - 1], [1, len(tgt_item) - 1]])  # add the SLOT for whole sentence
                # random.shuffle(phrases)
                span_mask_aligns = self.choose_aligns(phrases)
                span_mask_aligns = self.remove_overlapped_aligns(span_mask_aligns, src_item, tgt_item)
                replace_length = 0 
                origin_src_length = len(src_item)
                
                span_mask_aligns = sorted(span_mask_aligns, key=lambda x: x[0][0],reverse=True)
                
                for align in span_mask_aligns:
                    src_align, tgt_align = align 
                    # print(''.join(src_item[src_align[0]:src_align[1]]).replace('▁', ' '))
                    # print(''.join(tgt_item[tgt_align[0]:tgt_align[1]]).replace('▁', ' '))
                
                    src_item = src_item[:src_align[0]]+tgt_item[tgt_align[0]:tgt_align[1]]+['▁']+src_item[src_align[1]:]
                    replace_length+= (src_align[1]-src_align[0])
                    if replace_length/origin_src_length > self.dataset_config.replace_percent_threshhold:
                        break 
                        
        return src_item


    