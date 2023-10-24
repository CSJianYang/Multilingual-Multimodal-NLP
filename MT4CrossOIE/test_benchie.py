import argparse
import os
import time
import torch
from utils import utils
from dataset import load_data
from extract import extract
from evaluate.evaluate import Benchmark
from evaluate.matcher import Matcher
from evaluate.generalReader import GeneralReader
from carb.carb import Benchmark as CarbBenchmark
from carb.matcher import Matcher as CarbMatcher
from carb.tabReader import TabReader
from benchie.src.benchie import Benchie

def get_performance(output_path, gold_path):
    auc, precision, recall, f1 = [None for _ in range(4)]

    
    return auc, precision, recall, f1


def do_eval(output_path, gold_path):
    auc, prec, rec, f1 = get_performance(output_path, gold_path)
    eval_results = [f1, prec, rec, auc]
    return eval_results


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

    test_loader = load_data(
        data_path=args.test_data_path,
        batch_size=args.batch_size,
        tokenizer_config=args.bert_config,
        dataset_name='benchie',
        train=False)
    


    start = time.time()
    extract(args, model, test_loader, args.save_path, is_benchie=True) # do prediction
    pred_path = os.path.join(args.save_path, 'extraction.txt')
    benchie = Benchie()
    benchie.load_gold_annotations(filename=args.test_gold_path)
    benchie.add_oie_system_extractions(oie_system_name="CrossL-OIE", filename=pred_path)
    # benchie.add_oie_system_extractions(oie_system_name="xiang", filename='benchie/data/oie_systems_explicit_extractions/clausie_explicit.txt')

    # Compute scores
    benchie.compute_precision()
    benchie.compute_recall()
    benchie.compute_f1()

    # Print scores
    # print(f"K={top_K}")
    benchie.print_scores()

    # test_results = do_eval(args.save_path, args.test_gold_path)
    # print(test_results)
    # utils.print_results("TEST RESULT", test_results, ["F1  ", "PREC", "REC ", "AUC "])

    print("TIME: ", time.time() - start)
    print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--binary', nargs='?', const=True, default=False, type=utils.str2bool)  # binary/n-ary
    parser.add_argument('--model_path', default='./results/stage_1/unfreeze_emb/mbert/model-epoch1-end-score2.0049.bin') # select model
    parser.add_argument('--save_path', default='./results/benchie_test') # test results
    parser.add_argument('--test_data_path', default='') #     .pkl     test data
    parser.add_argument('--test_gold_path', default='') #     .tsv     test gold

    parser.add_argument('--bert_config', default='bert-base-multilingual-cased')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--visible_device', default="3") ###
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pos_emb_dim', type=int, default=64)
    parser.add_argument('--n_arg_heads', type=int, default=8)
    parser.add_argument('--n_arg_layers', type=int, default=4)
    parser.add_argument('--use_lstm', nargs='?', const=True, default=False, type=utils.str2bool)
    parser.add_argument('--lora_num', type=int, default=6)

    main_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = main_args.visible_device

    main_args.pred_n_labels = 3
    main_args.arg_n_labels = 9
    device = torch.device(main_args.device if torch.cuda.is_available() else 'cpu')
    main_args.device = device
    main(main_args)

