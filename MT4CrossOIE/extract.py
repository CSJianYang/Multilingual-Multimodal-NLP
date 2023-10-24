import os
import torch
import numpy as np
import utils.bio as bio
from transformers import BertTokenizer
from tqdm import tqdm


def extract(args,
            model,
            loader,
            output_path, is_benchie=False, top_K=None):
    model.eval()
    
    sentence_no = 0
    os.makedirs(output_path, exist_ok=True)
    extraction_path = os.path.join(output_path, "extraction.txt")
    tokenizer = BertTokenizer.from_pretrained(args.bert_config)
    f = open(extraction_path, 'w') # write the extractions into file 'extraction.txt'
    # predict on 'loader' (dev_set/test_set)
    for step, batch in tqdm(enumerate(loader), desc='prediction_steps', total=len(loader)): # prediction_steps
        token_strs = [[word for word in sent] for sent in np.asarray(batch[-2]).T]
        sentences = batch[-1]
        token_ids, att_mask = map(lambda x: x.to(args.device), batch[:-2])

        with torch.no_grad():
            """
            We will iterate B(batch_size) times
            because there are more than one predicate in one batch.
            In feeding to argument extractor, # of predicates takes a role as batch size.

            pred_logit: (B, L, 3)
            pred_hidden: (B, L, D)
            pred_tags: (B, P, L) ~ list of tensors, where P is # of predicate in each batch
            """
            pred_logit, pred_hidden = model.extract_predicate(
                input_ids=token_ids, attention_mask=att_mask, top_K=top_K)
            pred_tags = torch.argmax(pred_logit, 2)
            pred_tags = bio.filter_pred_tags(pred_tags, token_strs) # e.g. [2,2,0,2,2,2,0,1,2,2,2,2,0] There are 3 predicates in total for  one sentence
            pred_tags = bio.get_single_predicate_idxs(pred_tags) # bsz [list type]: 32     for each element in list : [pred_num, L] e.g. [[2,2,0,2,2,2,2,2,2,2,2,2,2],[2,2,2,2,2,2,0,1,2,2,2,2,2],[2,2,2,2,2,2,2,2,2,2,2,2,0]]
            pred_probs = torch.nn.Softmax(2)(pred_logit)

            # iterate B times (one iteration means extraction for one sentence)
            for cur_pred_tags, cur_pred_hidden, cur_att_mask, cur_token_id, cur_pred_probs, token_str, sentence \
                    in zip(pred_tags, pred_hidden, att_mask, token_ids, pred_probs, token_strs, sentences):

                # generate temporary batch for this sentence and feed to argument module
                cur_pred_masks = bio.get_pred_mask(cur_pred_tags).to(args.device)
                n_predicates = cur_pred_masks.shape[0]
                sentence_no+=1
                if n_predicates == 0:
                    continue  # if there is no predicate, we cannot extract.
                cur_pred_hidden = torch.cat(n_predicates * [cur_pred_hidden.unsqueeze(0)]) # n duplicated hidden
                cur_token_id = torch.cat(n_predicates * [cur_token_id.unsqueeze(0)])
                cur_arg_logit = model.extract_argument(
                    input_ids=cur_token_id,
                    predicate_hidden=cur_pred_hidden,
                    predicate_mask=cur_pred_masks)

                # filter and get argument tags with highest probability
                cur_arg_tags = torch.argmax(cur_arg_logit, 2)
                cur_arg_probs = torch.nn.Softmax(2)(cur_arg_logit)
                cur_arg_tags = bio.filter_arg_tags(cur_arg_tags, cur_pred_tags, token_str)

                # get string tuples and write results
                cur_extractions, cur_extraction_idxs = bio.get_tuple(sentence, cur_pred_tags, cur_arg_tags, tokenizer)
                cur_confidences = bio.get_confidence_score(cur_pred_probs, cur_arg_probs, cur_extraction_idxs)
                
                for extraction, confidence in zip(cur_extractions, cur_confidences): ########### output format #########
                    if is_benchie:
                        # f.write("\t".join([sentence] + [str(confidence)] + extraction) + '\n')
                        # print(extraction)
                        if extraction[1] and extraction[0] and extraction[2]:
                            f.write(f'{sentence_no}\t{extraction[1]}\t{extraction[0]}\t{extraction[2]}\n')
                        
                    else:
                        if args.binary:
                            f.write("\t".join([sentence] + [str(1.0)] + extraction[:3]) + '\n')
                        else:
                            f.write("\t".join([sentence] + [str(confidence)] + extraction) + '\n')
                
    f.close()
    print("\nExtraction Done.\n")

