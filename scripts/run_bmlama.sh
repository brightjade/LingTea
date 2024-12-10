#!bin/bash
export CUDA_VISIBLE_DEVICES=0

model_type="xglm-564M"
task="bmlama"
langs=("en fr es pt ar vi ca hi bn")

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
    --max_seq_len 32 \
    --forget_num 32 \
    --retain_multiplier 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --bf16 \
    --wandb_mode disabled
