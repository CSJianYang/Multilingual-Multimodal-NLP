export CUDA_VISIBLE_DEVICES=1

python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name /home/wangzixiang-b17/data/chai/LLM_checkpoint/Llama-2-7b-hf --output_dir ./ckpt
# python length_stat.py  --use_peft --peft_method lora --quantization --model_name /home/wangzixiang-b17/data/chai/LLM_checkpoint/Llama-2-7b-hf --output_dir ./ckpt >length.txt

# python llama_test_dataset.py --model_name /home/wangzixiang-b17/data/chai/LLM_checkpoint/Llama-2-7b-hf --output_dir ./ckpt