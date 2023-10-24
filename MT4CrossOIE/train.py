import os
import torch
import torch.nn as nn
import utils.bio as bio
from tqdm import tqdm
from extract import extract
from utils import utils
from test import do_eval
from benchie.src.benchie import Benchie
import time
from icecream import ic 

def train(args,
          epoch,
          model,
          trn_loader,
          dev_loaders,
          summarizer,
          optimizer,
          scheduler,
          test_loaders=None):
    total_pred_loss, total_arg_loss, trn_results = 0, 0, None
    epoch_steps = int(args.total_steps / args.epochs)

    iterator = tqdm(enumerate(trn_loader), desc='steps', total=epoch_steps)

    highest_totatl_testset_f1 = 0
    # old_highest_score = 0
    for step, batch in iterator:
        batch = map(lambda x: x.to(args.device), batch)
        token_ids, att_mask, single_pred_label, single_arg_label, all_pred_label = batch
        pred_mask = bio.get_pred_mask(single_pred_label) # only mask single predicate (including non-single word)  to 1 and 0 for others

        model.train()
        model.zero_grad()

        # feed to predicate model
        batch_loss, pred_loss, arg_loss = model(
            input_ids=token_ids,
            attention_mask=att_mask,
            predicate_mask=pred_mask,
            total_pred_labels=all_pred_label,
            arg_labels=single_arg_label)

        # get performance on this batch
        total_pred_loss += pred_loss.item()
        total_arg_loss += arg_loss.item()

        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        trn_results = [total_pred_loss / (step + 1), total_arg_loss / (step + 1)]
        if step > epoch_steps:
            break

        # interim evaluation
        if step % 1000 == 0 and step != 0:
            dev_iter = zip(args.dev_data_path, args.dev_gold_path, dev_loaders)
            dev_results = list()
            total_sum = 0
            for dev_input, dev_gold, dev_loader in dev_iter:
                dev_name = dev_input.split('/')[-1].replace('.pkl', '')
                output_path = os.path.join(args.save_path, f'epoch{epoch}_dev/step{step}/{dev_name}')

                extract(args, model, dev_loader, output_path) ###########  prediction  ############

                try:
                    dev_result = do_eval(output_path, dev_gold) # evaluation
                    # logging.info(f'meizha step {step}\n')
                except:
                    dev_result = [-1, -1, -1, -1]
                    # logging.info(f'============= zhale step {step} ===============\n')

                utils.print_results(f"EPOCH{epoch} STEP{step} EVAL", dev_result, ["F1  ", "PREC", "REC ", "AUC "])
                total_sum += dev_result[0] + dev_result[-1] # score: total_sum = auc + f1
                dev_result.append(dev_result[0] + dev_result[-1])
                dev_results += dev_result

            summarizer.save_results([step] + trn_results + dev_results + [total_sum]) # train & dev results saved in 'train_results.csv'
            model_name = utils.set_model_name(total_sum, epoch, step)


            torch.save(model.state_dict(), os.path.join(args.save_path, model_name)) ###

########################################## test on benchmarks ##############################################
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

            #         test_output_path = os.path.join(args.save_path, f'epoch{epoch}_dev/step{step}/{test_name}')

            #         if test_gold == './evaluate/Re-OIE2016.json' or test_gold == './carb/CaRB_test.tsv':
            #             args.binary = False
            #         else:
            #             args.binary = True
                    
            #         # top_K=1
            #         if 'sample' in test_name:
                    
            #             #top_K = 1
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
            #             utils.print_results(f"EPOCH{epoch} STEP{step} Current K={top_K} {test_name}", test_result, ["F1  ", "PREC", "REC "])
            #         else:
            #             utils.print_results(f"EPOCH{epoch} STEP{step} Current K={top_K} {test_name}", test_result, ["F1  ", "PREC", "REC ", "AUC "])

            #         test_f1_sum += test_result[0]

            #     print(f"Current K={top_K} STEP{step} F1_sum={test_f1_sum}\n")
            #     test_f1_for_each_K.append(test_f1_sum)

            #     args.binary = False

            # max_value = max(test_f1_for_each_K)
            # for i in range(len(test_f1_for_each_K)):
            #     if test_f1_for_each_K[i] == max_value:
            #         print(f"Step {step}: Best K={i+1} total_testset_f1={max_value}")

            # if max_value >= highest_totatl_testset_f1:
            #     highest_totatl_testset_f1 = max_value
            #     print(f"Current highest totatl_testset_f1={highest_totatl_testset_f1}")
            # print("#########################################################")

#########################################################################################
        if step % args.summary_step == 0 and step != 0:
            utils.print_results(f"EPOCH{epoch} STEP{step} TRAIN",
                                trn_results, ["PRED LOSS", "ARG LOSS "])

    # end epoch summary

    utils.print_results(f"EPOCH{epoch} TRAIN", trn_results, ["PRED LOSS", "ARG LOSS "])
    return trn_results

