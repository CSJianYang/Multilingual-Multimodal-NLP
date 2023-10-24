# model_dir='./results/stage_3/emb_pos_6lora'
# checkpoint_name='/model-epoch1-end-score1.9671.bin'
# CUDA_VISIBLE_DEVICES=0
# model_dir='./results'
# checkpoint_name='/Multi2OIE_multilingual.bin'

# model_dir='./results/stage_3/emb_pos_multi_bert-muliti_maxlength100_bs64_without_mLoRA'
# checkpoint_name='/model-epoch1-end-score1.9964.bin'
# export CUDA_VISIBLE_DEVICES=1
# i=1

######################################### Benchie Binary #############################################
# #Benchie-en
# python test_benchie.py --test_data_path './benchie/data/sentences/sample300_en.txt' \
#                       --test_gold_path './benchie/data/gold/benchie_gold_annotations_en.txt'\
#                       --model_path "${model_dir}${checkpoint_name}"\
#                       --save_path "${model_dir}/benchie_test_result_en"\
#                       --binary 'True' \
#                       --visible_device $i


#Benchie-zh

# python test_benchie.py --test_data_path './benchie/data/sentences/sample300_zh.txt' \
#                        --test_gold_path './benchie/data/gold/benchie_gold_annotations_zh.txt'\
#                        --model_path "${model_dir}${checkpoint_name}"\
#                        --save_path "${model_dir}/benchie_test_result_zh"\
#                        --binary 'True' \
#                        --visible_device $i




##Benchie-de
# python test_benchie.py --test_data_path './benchie/data/sentences/sample300_de.txt' \
#                        --test_gold_path './benchie/data/gold/benchie_gold_annotations_de.txt'\
#                        --model_path "${model_dir}${checkpoint_name}"\
#                        --save_path "${model_dir}/benchie_test_result_de"\
#                        --binary 'True' \
#                        --visible_device $i



#Benchie-ar
# python test_benchie.py --test_data_path './benchie/data/sentences/sample100_ar.txt' \
#                        --test_gold_path './benchie/data/gold/benchie_gold_annotations_ar.txt'\
#                        --model_path "${model_dir}${checkpoint_name}"\
#                        --save_path "${model_dir}/benchie_test_result_ar"\
#                        --binary 'True'\
#                        --visible_device $i



########################################### N-ary ##########################################################

#Re-oie2016-EN-nary (Lexical Match) mutil2oie(multilingual): f1 83.76 84.93 82.62    auc 76.23 

# python test.py --test_data_path './datasets/re_oie2016_test.pkl' \
#                --test_gold_path './evaluate/Re-OIE2016.json'\
#                --model_path "${model_dir}${checkpoint_name}"\
#                --save_path "${model_dir}/re_oie2016_test_result_en_nary"





# CaRB-EN-nary (Tuple Match)        mutil2oie(multilingual): f1 51.85 59.53 45.92    auc 31.54

# python test.py --test_data_path './datasets/carb_test.pkl' \
#               --test_gold_path './carb/CaRB_test.tsv'\
#               --model_path "${model_dir}${checkpoint_name}"\
#               --save_path "${model_dir}/carb_test_result_en_nary" \
#               --visible_device $i

# ########################################## Binary(Tuple Match)...Change to LM test now ##########################################################

#Re-oie2016-EN-Binary

# python test.py --test_data_path './datasets/re_oie2016_test_english.pkl' \
#               --test_gold_path './evaluate/Re-OIE2016-Binary.json'\
#               --model_path "${model_dir}${checkpoint_name}"\
#               --save_path "${model_dir}/re_oie2016_test_result_en_binary"\
#               --binary 'True' \
#               --visible_device $i


# Re-oie2016-PT-Binary
# python test.py --test_data_path './datasets/re_oie2016_test_portuguese.pkl' \
#                --test_gold_path './evaluate/Re-OIE2016-Portuguese-Binary.json'\
#                --model_path "${model_dir}${checkpoint_name}"\
#                --save_path "${model_dir}/re_oie2016_test_result_pt_binary"\
#                --binary 'True' \
#                --visible_device $i

# # Re-oie2016-ES-Binary
# #
# python test.py --test_data_path './datasets/re_oie2016_test_spanish.pkl' \
#                --test_gold_path './evaluate/Re-OIE2016-Spanish-Binary.json'\
#                --model_path "${model_dir}${checkpoint_name}"\
#                --save_path "${model_dir}/re_oie2016_test_result_es_binary"\
#                --binary 'True' \
#                --visible_device $i
####################################################################################################

# # Re-oie2016-ES-Binary-Clean(LM)
# #
# python test.py --test_data_path './datasets/re_oie2016_test_spanish_clean.pkl' \
#                --test_gold_path './evaluate/Re-OIE2016-Spanish-Binary-Clean-original.json'\
#                --model_path "${model_dir}${checkpoint_name}"\
#                --save_path "${model_dir}/re_oie2016_test_result_es_binary"\
#                --binary 'True'\
#                --visible_device $i
################################# Select Best K in model ##########################################################
# test Top-K
# model_dir='./results/stage_3/emb_pos_multi_lora6_T100_rank8'
# checkpoint_name='/model-epoch1-step23000-score2.0138.bin' 
# export CUDA_VISIBLE_DEVICES=1

# python test_model_topk.py  --model_path   "${model_dir}${checkpoint_name}"\
#                            --visible_device 1 > test_multi_lora6_T100_rank8.log 2>&1 &
                           

# model_dir='./results/stage_3/emb_pos_multi_bert-muliti_maxlength100_bs64_without_mLoRA'
# checkpoint_name='/model-epoch1-end-score1.9964.bin' 
# export CUDA_VISIBLE_DEVICES=1

# python test_model_topk.py  --model_path   "${model_dir}${checkpoint_name}"\
#                            --visible_device 1 \
#                            --save_path './results/stage_3/emb_pos_multi_bert-muliti_maxlength100_bs64_without_mLoRA' > test_emb_pos_multi_bert-muliti_maxlength100_bs64_without_mLoRA.log 2>&1 &


# Ablation without TopK without rank / revise test_model_topk.py
# model_dir='./results/stage_3/emb_pos_en_mLoRA_bestrank4_without_openie4plusplus'
# checkpoint_name='/model-epoch1-end-score1.9697.bin' 
# export CUDA_VISIBLE_DEVICES=2

# python test_model_topk.py  --model_path   "${model_dir}${checkpoint_name}"\
#                            --visible_device 2 \
#                            --save_path './results/ablations/emb_pos_en_mLoRA_bestrank4_without_openie4plusplus' > en_mLoRA_bestrank4_without_openie4plusplus.log 2>&1 &

     