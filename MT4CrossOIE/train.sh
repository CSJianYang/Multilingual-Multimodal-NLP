# save the whole model before training ...
export CUDA_VISIBLE_DEVICES=2 
python main.py  --visible_device 2 \
                --save_path './results'\
                --train_stage 0

# # bs=128
# export CUDA_VISIBLE_DEVICES=3 
# python main.py  --visible_device 3 \
#                 --save_path './results/multi2oie_en_bert-muliti_maxlength100_bs128'\
#                 --train_stage 0

# CUDA_VISIBLE_DEVICES=2
# python main.py  --visible_device 2 \
#                 --save_path './results/stage_1/emb_en_bert-muliti_maxlength100_bs128'\
#                 --train_stage 1

# CUDA_VISIBLE_DEVICES=2
# python  main.py                       --visible_device 2 \
#                                       --load_model_path './results/stage_1/emb_en_bert-muliti_maxlength100_bs128/model-epoch1-end-score2.0053.bin'\
#                                       --save_path './results/stage_2/emb_pos_en_bert-muliti_maxlength100_bs128'\
#                                       --train_stage 2

# export CUDA_VISIBLE_DEVICES=2
# python main.py --visible_device 2 \
#                --load_model_path './results/stage_2/emb_pos_en_bert-muliti_maxlength100_bs128/model-epoch1-end-score1.9494.bin' \
#                 --train_stage 3 \
#                --save_path './results/stage_3/emb_pos_en_bert-muliti_maxlength100_bs64_rank64' # > train_0718_en_mLoRA_bestrank4_without_openie4plusplus.log 2>&1 &


# export CUDA_VISIBLE_DEVICES=0
# python main.py --visible_device 0 \
#                --load_model_path './results/stage_2/emb_pos_en_bert-muliti_maxlength100_bs128/model-epoch1-end-score1.9494.bin' \
#                 --train_stage 3 \
#                --save_path './results/stage_3/emb_pos_multi_bert-muliti_maxlength100_bs64_without_mLoRA' # > train_0718_en_mLoRA_bestrank4_without_openie4plusplus.log 2>&1 &



