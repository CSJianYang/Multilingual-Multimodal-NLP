import os
import torch
import torch.nn as nn
import utils.bio as bio
from tqdm import tqdm
from extract import extract
from utils import utils
from test import do_eval
from benchie.src.benchie import Benchie
from icecream import ic 
import argparse
from dataset import load_data
from evaluate.evaluate import Benchmark
from evaluate.matcher import Matcher
from evaluate.generalReader import GeneralReader
from carb.carb import Benchmark as CarbBenchmark
from carb.matcher import Matcher as CarbMatcher
from carb.tabReader import TabReader
import numpy as np

LANG='ar' # en zh de pt es ar

def main(args):
    model = utils.get_models(
        bert_config=args.bert_config,
        pred_n_labels=args.pred_n_labels,
        arg_n_labels=args.arg_n_labels,
        n_arg_heads=args.n_arg_heads,
        n_arg_layers=args.n_arg_layers,
        pos_emb_dim=args.pos_emb_dim,
        use_lstm=args.use_lstm,
        device=args.device)
    
    model.load_state_dict(torch.load(args.model_path))
    model.zero_grad()
    model.eval()

    # loaded_matrix = np.load('./results/EN_matrix_file_1.npy')

    # # 打印加载的矩阵
    # print(loaded_matrix)
    # exit()


    # test_loaders = [
    #     load_data(
    #     data_path=cur_test_path,
    #     batch_size=args.batch_size,
    #     tokenizer_config=args.bert_config,
    #     dataset_name='benchie' if 'benchie' in cur_test_path else 'oie',
    #     train=False)
    #     for index, cur_test_path in enumerate(args.test_data_path)]
    
    trn_loader = load_data(
        data_path=args.trn_data_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        tokenizer_config=args.bert_config)
    
    matrix = torch.tensor([]).to(args.device)
    iterator = tqdm(enumerate(trn_loader))
    count = 0
    for i, batch in iterator:
        index = i + 1
        batch = map(lambda x: x.to(args.device), batch)
        token_ids, att_mask, single_pred_label, single_arg_label, all_pred_label = batch
        pred_mask = bio.get_pred_mask(single_pred_label) # only mask single predicate (including non-single word)  to 1 and 0 for others

        # feed to predicate model
        avg_vector = model(
            input_ids=token_ids,
            attention_mask=att_mask,
            predicate_mask=pred_mask,
            total_pred_labels=all_pred_label,
            arg_labels=single_arg_label) 
        
        matrix = torch.cat((matrix, avg_vector), dim=0) 

        if index % 100 == 0:
            count += 1
            numpy_array = matrix.cpu().detach().numpy()
            np.save(f'./results/2nd_stage/{LANG}_matrix_file_{count}.npy', numpy_array)  
            torch.cuda.empty_cache()
 
            print("cache empty")
            if count == 5:
                print("Finished!")
                exit()

            

            matrix = torch.tensor([]).to(args.device)
    ################################### top-k test ########################################


    # test_f1_for_each_K = []
    # for k in range(args.lora_num):
    #     top_K = k+1
    #     test_f1_sum = 0
    #     test_iter = zip(args.test_data_path, args.test_gold_path, test_loaders)

    #     for test_input, test_gold, test_loader in test_iter:
    #         test_name = test_input.split('/')[-1]
    #         if 'txt' in test_name:
    #             test_name = test_name.replace('.txt', '')
    #         else:
    #             test_name = test_name.replace('.pkl', '')

    #         test_output_path = os.path.join(args.save_path, f'top{top_K}/{test_name}')

    #         if test_gold == './evaluate/Re-OIE2016.json' or test_gold == './carb/CaRB_test.tsv':
    #             args.binary = False
    #         else:
    #             args.binary = True
            
            
    #         if 'sample' in test_name:
    #             print("###########################")
                
    #             extract(args, model, test_loader, test_output_path, is_benchie=True, top_K=top_K)  # do prediction                       
    #             pred_path = os.path.join(test_output_path, 'extraction.txt')
    #             # print('pred_path: ', pred_path)
    #             benchie = Benchie()
    #             benchie.load_gold_annotations(filename=test_gold) # test_gold: './benchie/data/gold/...'
    #             try:
    #                 benchie.add_oie_system_extractions(oie_system_name="CrossL-OIE", filename=pred_path)
    #                 # Compute scores
    #                 benchie.compute_precision()
    #                 benchie.compute_recall()
    #                 benchie.compute_f1()

    #                 # Print scores
    #                 test_result = benchie.print_scores()
    #             except:
    #                 test_result = [-1, -1, -1]
    #         else:
    #             extract(args, model, test_loader, test_output_path, top_K=top_K)  ############  prediction  ############
    #             test_result = do_eval(test_output_path, test_gold) # evaluation
    #             # logging.info(f'meizha step {step}\n')
    #         # except:
    #         #     test_result = [-1, -1, -1, -1]
    #             # logging.info(f'============= zhale step {step} ===============\n')

    #         if 'sample' in test_name:
    #             utils.print_results(f"Current K={top_K} {test_name}", test_result, ["F1  ", "PREC", "REC "])
    #         else:
    #             utils.print_results(f" Current K={top_K} {test_name}", test_result, ["F1  ", "PREC", "REC ", "AUC "])

    #         test_f1_sum += test_result[0]

    #     print(f" Top K={top_K} F1_sum={test_f1_sum}\n")
    #     test_f1_for_each_K.append(test_f1_sum)

    #     args.binary = False

    # max_value = max(test_f1_for_each_K)
    # for i in range(len(test_f1_for_each_K)):
    #     if test_f1_for_each_K[i] == max_value:
    #         print(f" Best K={i+1} total_testset_f1={max_value}")

    # print("#########################################################")

############################# non-TopK test ##################################

    # test_iter = zip(args.test_data_path, args.test_gold_path, test_loaders)

    # for test_input, test_gold, test_loader in test_iter:
    #     test_name = test_input.split('/')[-1]
    #     if 'txt' in test_name:
    #         test_name = test_name.replace('.txt', '')
    #     else:
    #         test_name = test_name.replace('.pkl', '')

    #     test_output_path = os.path.join(args.save_path, f'{test_name}')

    #     if test_gold == './evaluate/Re-OIE2016.json' or test_gold == './carb/CaRB_test.tsv':
    #         args.binary = False
    #     else:
    #         args.binary = True
            
            
    #     if 'sample' in test_name:
    #         print("###########################")
                
    #         extract(args, model, test_loader, test_output_path, is_benchie=True)  # do prediction                       
    #         pred_path = os.path.join(test_output_path, 'extraction.txt')
    #         # print('pred_path: ', pred_path)
    #         benchie = Benchie()
    #         benchie.load_gold_annotations(filename=test_gold) # test_gold: './benchie/data/gold/...'
    #         try:
    #             benchie.add_oie_system_extractions(oie_system_name="CrossL-OIE", filename=pred_path)
    #             # Compute scores
    #             benchie.compute_precision()
    #             benchie.compute_recall()
    #             benchie.compute_f1()

    #             # Print scores
    #             test_result = benchie.print_scores()
    #         except:
    #             test_result = [-1, -1, -1]
    #     else:
    #         extract(args, model, test_loader, test_output_path)  ############  prediction  ############
    #         test_result = do_eval(test_output_path, test_gold) # evaluation
    #         # logging.info(f'meizha step {step}\n')
    #         # except:
    #         #     test_result = [-1, -1, -1, -1]
    #             # logging.info(f'============= zhale step {step} ===============\n')

    #     if 'sample' in test_name:
    #         utils.print_results(f"Current  {test_name}", test_result, ["F1  ", "PREC", "REC "])
    #     else:
    #         utils.print_results(f" Current {test_name}", test_result, ["F1  ", "PREC", "REC ", "AUC "])

    ########################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--trn_data_path', default=f'./datasets/openie4_train_{LANG}.pkl')
    #parser.add_argument('--trn_data_path', default=['./datasets/openie4_train_en.pkl', './datasets/openie4_train_zh.pkl', './datasets/openie4_train_de.pkl', './datasets/openie4_train_es.pkl', './datasets/openie4_train_pt.pkl', './datasets/openie4_train_ar.pkl'])

    parser.add_argument('--binary', nargs='?', const=True, default=False, type=utils.str2bool)  # binary/n-ary
    parser.add_argument('--model_path', default='./results/stage_2/emb_pos_en_bert-muliti_maxlength100_bs128/model-epoch1-end-score1.9494.bin') # select model
    parser.add_argument('--save_path', default='./results/All_benchmarks_extractions') # test results


    # parser.add_argument('--test_data_path', default=['./benchie/data/sentences/sample300_en.txt',\
    #                                                  './benchie/data/sentences/sample300_zh.txt',\
    #                                                  './benchie/data/sentences/sample300_de.txt',\
    #                                                  './benchie/data/sentences/sample100_ar.txt',\
    #                                                  #'./datasets/re_oie2016_test.pkl',\
    #                                                  './datasets/carb_test.pkl',\
    #                                                  './datasets/re_oie2016_test_english.pkl',\
    #                                                  './datasets/re_oie2016_test_portuguese.pkl',\
    #                                                  './datasets/re_oie2016_test_spanish.pkl'
    #                                                  #'./datasets/re_oie2016_test_spanish_clean.pkl'
    #                                                  ])

    # parser.add_argument('--test_gold_path', nargs='+', default=['./benchie/data/gold/benchie_gold_annotations_en.txt',\
    #                                                             './benchie/data/gold/benchie_gold_annotations_zh.txt',\
    #                                                             './benchie/data/gold/benchie_gold_annotations_de.txt',\
    #                                                             './benchie/data/gold/benchie_gold_annotations_ar.txt',\
    #                                                             #'./evaluate/Re-OIE2016.json',\
    #                                                             './carb/CaRB_test.tsv',\
    #                                                             './evaluate/Re-OIE2016-Binary.json',\
    #                                                             './evaluate/Re-OIE2016-Portuguese-Binary.json',\
    #                                                             './evaluate/Re-OIE2016-Spanish-Binary.json'
    #                                                             #'./evaluate/Re-OIE2016-Spanish-Binary-Clean-original.json'
    #                                                             ])


    
    # parser.add_argument('--test_data_path', default=['./datasets/re_oie2016_test_spanish_clean.pkl'])

    # parser.add_argument('--test_gold_path', nargs='+', default=['./evaluate/Re-OIE2016-Spanish-Binary-Clean-original.json'])


    parser.add_argument('--bert_config', default='bert-base-multilingual-cased')
    #parser.add_argument('--device', default='cpu')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--visible_device', default="2") #
    # parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1) #
    parser.add_argument('--pos_emb_dim', type=int, default=64)
    parser.add_argument('--n_arg_heads', type=int, default=8)
    parser.add_argument('--n_arg_layers', type=int, default=4)
    parser.add_argument('--use_lstm', nargs='?', const=True, default=False, type=utils.str2bool)
    parser.add_argument('--lora_num', type=int, default=6)
    parser.add_argument('--max_len', type=int, default=100)
    main_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = main_args.visible_device

    main_args.pred_n_labels = 3
    main_args.arg_n_labels = 9
    device = torch.device(main_args.device if torch.cuda.is_available() else 'cpu')
    main_args.device = device
    main(main_args)
