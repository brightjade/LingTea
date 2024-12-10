#!bin/bash
export CUDA_VISIBLE_DEVICES=0

model_type="xglm-564M"
task="flores"
langs=("en fr es zh ar vi eu ur te sw")

python run.py \
    --model_name_or_path facebook/${model_type} \
    --model_type ${model_type} \
    --cache_dir ../../.cache/ \
    --data_dir data/${task}/ \
    --task ${task} \
    --method lingtea \
    --forget_lang $langs \
    --retain_lang $langs \
    --do_train \
    --do_test \
    --max_seq_len 256 \
    --forget_num 32 \
    --retain_multiplier 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --bf16 \
    --wandb_mode disabled
