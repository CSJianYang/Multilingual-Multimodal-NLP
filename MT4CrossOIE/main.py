import argparse
import os
import torch
from utils import utils
from utils.utils import SummaryManager
from dataset import load_data
from tqdm import tqdm
from train import train
from extract import extract
from test import do_eval

import loralib as lora
import random
from torch.utils.data import ConcatDataset
from icecream import ic 
        
def main(args):
    utils.set_seed(args.seed)
    model = utils.get_models(
        bert_config=args.bert_config,
        pred_n_labels=args.pred_n_labels,
        arg_n_labels=args.arg_n_labels,
        n_arg_heads=args.n_arg_heads,
        n_arg_layers=args.n_arg_layers,
        lstm_dropout=args.lstm_dropout,
        mh_dropout=args.mh_dropout,
        pred_clf_dropout=args.pred_clf_dropout,
        arg_clf_dropout=args.arg_clf_dropout,
        pos_emb_dim=args.pos_emb_dim,
        use_lstm=args.use_lstm,
        device=args.device)

    train_stage = args.train_stage
    print(f"args.train_stage: {args.train_stage}")
    # stage_1
    if train_stage == 1:
        for name, param in model.bert.named_parameters():
            print(name)
            if "embeddings.word_embeddings" in name: # finetune  word
                param.requires_grad = True
                print('only tune word_embeddings')
            else:
                param.requires_grad = False

        # use Multilingual datasets
        # trn_loader = load_data(
        #     data_path=args.trn_data_path,
        #     batch_size=args.batch_size,
        #     max_len=args.max_len,
        #     dataset_name='multi_oie',
        #     tokenizer_config=args.bert_config)

        # use mLoRA & English datasets
        trn_loader = load_data(
            data_path=args.trn_data_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            tokenizer_config=args.bert_config)

    # stage_2
    elif train_stage == 2:
        model.load_state_dict(torch.load(args.load_model_path), strict=False)
        print(f"Load Model from: {args.load_model_path}")

        for name, param in model.bert.named_parameters():
            print(name)
            if "embeddings.word_embeddings" in name: # finetune position
                param.requires_grad = False
                print('word_embeddings froze')
            else:
                param.requires_grad = True


        # use Multilingual datasets
        # trn_loader = load_data(
        #     data_path=args.trn_data_path,
        #     batch_size=args.batch_size,
        #     max_len=args.max_len,
        #     dataset_name='multi_oie',
        #     tokenizer_config=args.bert_config)


        # use mLoRA & English datasets
        trn_loader = load_data(
            data_path=args.trn_data_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            tokenizer_config=args.bert_config)

    # stage 3
    elif train_stage == 3:
        # load 2nd model & set mLoRA setting
        model.load_state_dict(torch.load(args.load_model_path), strict=False)
        print(f"Load Model from: {args.load_model_path}")
        # use mLoRA
        lora.mark_only_lora_as_trainable(model.bert)  # lora finetuning

        # use mLoRA & Multilingual datasets
        # trn_loader = load_data(
        #     data_path=args.trn_data_path,
        #     batch_size=args.batch_size,
        #     max_len=args.max_len,
        #     dataset_name='multi_oie',
        #     tokenizer_config=args.bert_config)
        
        # use mLoRA & English datasets
        trn_loader = load_data(
            data_path=args.trn_data_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            tokenizer_config=args.bert_config)

    else:
        # model.load_state_dict(torch.load(args.load_model_path), strict=False)
        print(" ======================== load model without LoRA training =========================")

        # use Multilingual datasets
        trn_loader = load_data(
            data_path=args.trn_data_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            dataset_name='multi_oie',
            tokenizer_config=args.bert_config)

        # use mLoRA & English datasets
        # trn_loader = load_data(
        #     data_path=args.trn_data_path,
        #     batch_size=args.batch_size,
        #     max_len=args.max_len,
        #     tokenizer_config=args.bert_config)
        


    dev_loaders = [
        load_data(
            data_path=cur_dev_path,
            batch_size=args.dev_batch_size,
            tokenizer_config=args.bert_config,
            train=False)
        for cur_dev_path in args.dev_data_path]

    test_loaders = [
        load_data(
        data_path=cur_test_path,
        batch_size=args.test_batch_size,
        tokenizer_config=args.bert_config,
        dataset_name='benchie' if 'benchie' in cur_test_path else 'oie',
        train=False)
        for index, cur_test_path in enumerate(args.test_data_path)]
    
    # test_loaders = [
    #     load_data(
    #     data_path=cur_test_path,
    #     batch_size=args.test_batch_size,
    #     tokenizer_config=args.bert_config,
    #     dataset_name='oie',
    #     train=False)
    #     for index, cur_test_path in enumerate(args.test_data_path)]

    args.total_steps  = round(len(trn_loader) * args.epochs)
    args.warmup_steps = round(args.total_steps / 10)

    optimizer, scheduler = utils.get_train_modules(
        model=model,
        lr=args.learning_rate,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps)
    model.zero_grad()
    summarizer = SummaryManager(args)
    print("\nTraining Starts\n")

    torch.save(model.state_dict(), os.path.join(args.save_path, "before_training.bin"))
    exit()
    for epoch in tqdm(range(1, args.epochs + 1), desc='epochs'):
        trn_results = train(
            args, epoch, model, trn_loader, dev_loaders,
            summarizer, optimizer, scheduler, test_loaders=test_loaders)

        # extraction on dev-set
        dev_iter = zip(args.dev_data_path, args.dev_gold_path, dev_loaders)
        dev_results = list()
        total_sum = 0
        for dev_input, dev_gold, dev_loader in dev_iter:
            dev_name = dev_input.split('/')[-1].replace('.pkl', '')
            output_path = os.path.join(args.save_path, f'epoch{epoch}_dev/end_epoch/{dev_name}')
            extract(args, model, dev_loader, output_path)
            dev_result = do_eval(output_path, dev_gold)
            utils.print_results(f"EPOCH{epoch} EVAL",
                                dev_result, ["F1  ", "PREC", "REC ", "AUC "])
            total_sum += dev_result[0] + dev_result[-1]
            dev_result.append(dev_result[0] + dev_result[-1])
            dev_results += dev_result
        summarizer.save_results([epoch] + trn_results + dev_results + [total_sum])
        model_name = utils.set_model_name(total_sum, epoch)
        torch.save(model.state_dict(), os.path.join(args.save_path, model_name))
    print("\nTraining Ended\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--save_path', default='./results')
    parser.add_argument('--save_path', default='./results/default') # stage 1 2 3
    parser.add_argument('--load_model_path', default='./results/stage_2/') # stage 2 3
    parser.add_argument('--bert_config', default='bert-base-multilingual-cased', help='or bert-base-cased')

    #parser.add_argument('--trn_data_path', default='./datasets/openie4_train.pkl')
    # re-preprocessing
    parser.add_argument('--trn_data_path', default=['./datasets/openie4_train_en.pkl', './datasets/openie4_train_zh.pkl', './datasets/openie4_train_de.pkl', './datasets/openie4_train_es.pkl', './datasets/openie4_train_pt.pkl', './datasets/openie4_train_ar.pkl'])
    

    parser.add_argument('--dev_data_path', nargs='+', default=['./datasets/oie2016_dev.pkl', './datasets/carb_dev.pkl'])
    parser.add_argument('--dev_gold_path', nargs='+', default=['./evaluate/OIE2016_dev.txt', './carb/CaRB_dev.tsv'])

    parser.add_argument('--test_data_path', default=['./benchie/data/sentences/sample300_en.txt',\
                                                     './benchie/data/sentences/sample300_zh.txt',\
                                                     './benchie/data/sentences/sample300_de.txt',\
                                                     './benchie/data/sentences/sample100_ar.txt',\
                                                     './datasets/re_oie2016_test.pkl',\
                                                     './datasets/carb_test.pkl',\
                                                     './datasets/re_oie2016_test_english.pkl',\
                                                     './datasets/re_oie2016_test_portuguese.pkl',\
                                                     './datasets/re_oie2016_test_spanish.pkl',\
                                                     './datasets/re_oie2016_test_spanish_clean.pkl'])

    parser.add_argument('--test_gold_path', nargs='+', default=['./benchie/data/gold/benchie_gold_annotations_en.txt',\
                                                                './benchie/data/gold/benchie_gold_annotations_zh.txt',\
                                                                './benchie/data/gold/benchie_gold_annotations_de.txt',\
                                                                './benchie/data/gold/benchie_gold_annotations_ar.txt',\
                                                                './evaluate/Re-OIE2016.json',\
                                                                './carb/CaRB_test.tsv',\
                                                                './evaluate/Re-OIE2016-Binary.json',\
                                                                './evaluate/Re-OIE2016-Portuguese-Binary.json',\
                                                                './evaluate/Re-OIE2016-Spanish-Binary.json',\
                                                                './evaluate/Re-OIE2016-Spanish-Binary-Clean-original.json'])

    # parser.add_argument('--test_data_path', default=['./benchie/data/sentences/sample300_en.txt',\
    #                                                  './benchie/data/sentences/sample300_zh.txt'
    #                                                     #'./datasets/re_oie2016_test_spanish.pkl',\
    #                                                 #'./datasets/re_oie2016_test_spanish_clean.pkl'
    #                                                 ])

    # parser.add_argument('--test_gold_path', nargs='+', default=['./benchie/data/gold/benchie_gold_annotations_en.txt',\
    #                                                             './benchie/data/gold/benchie_gold_annotations_zh.txt'
    #                                                                 #'./evaluate/Re-OIE2016-Spanish-Binary.json',\
    #                                                             #'./evaluate/Re-OIE2016-Spanish-Clean-Binary.json'
    #                                                             ])
    
    # parser.add_argument('--test_data_path', default=[
    #                                                  './datasets/re_oie2016_test.pkl',\
    #                                                  './datasets/carb_test.pkl',\
    #                                                  './datasets/re_oie2016_test_english.pkl',\
    #                                                  './datasets/re_oie2016_test_portuguese.pkl',\
    #                                                  './datasets/re_oie2016_test_spanish.pkl'])

    # parser.add_argument('--test_gold_path', nargs='+', default=[
    #                                                             './evaluate/Re-OIE2016.json',\
    #                                                             './carb/CaRB_test.tsv',\
    #                                                             './evaluate/Re-OIE2016-Binary.json',\
    #                                                             './evaluate/Re-OIE2016-Portuguese-Binary.json',\
    #                                                             './evaluate/Re-OIE2016-Spanish-Binary.json'])

    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--visible_device', default="1")
    parser.add_argument('--summary_step', type=int, default=100)
    parser.add_argument('--use_lstm', nargs='?', const=True, default=False, type=utils.str2bool)
    parser.add_argument('--binary', nargs='?', const=True, default=False, type=utils.str2bool)

    # hyper-parameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lstm_dropout', type=float, default=0.)
    parser.add_argument('--mh_dropout', type=float, default=0.2)
    parser.add_argument('--pred_clf_dropout', type=float, default=0.)
    parser.add_argument('--arg_clf_dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)   # 64 for only 3rd
    parser.add_argument('--dev_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--n_arg_heads', type=int, default=8)
    parser.add_argument('--n_arg_layers', type=int, default=4)
    parser.add_argument('--pos_emb_dim', type=int, default=64)

    parser.add_argument('--train_stage', type=int, default=0)

    parser.add_argument('--lora_num', type=int, default=6)
    main_args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = main_args.visible_device
    main_args = utils.clean_config(main_args)
    main(main_args)