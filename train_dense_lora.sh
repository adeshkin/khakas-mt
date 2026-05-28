#!/bin/bash

MODEL_SIZE="1.8B"
model_path="tencent/Hy-MT2-1.8B"
output_path="./experiments/hy-mt2-1.8b-kjh-ru-finetune"
HIDDEN_SIZE=2048
INTERMEDIATE_SIZE=6144
NUM_ATTENTION_HEADS=16
NUM_KEY_VALUE_HEADS=4
NUM_LAYERS=32

tokenizer_path=${model_path}
train_data_file="./experiments/khakas_russian_train_data.jsonl"

# ============== Output & Logging ==============
mkdir -p ${output_path}

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${output_path}/"log_${current_time}.txt"


echo "============================================"
echo "Dense ${MODEL_SIZE} LoRA fine-tuning"
echo "Model path: ${model_path}"
echo "Output path: ${output_path}"
echo "============================================"

# ============== Launch Training ==============
python -u train_dense.py \
    --do_train \
    --model_size ${MODEL_SIZE} \
    --model_name_or_path ${model_path} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --train_data_file ${train_data_file} \
    --output_dir ${output_path} \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 100 \
    --max_steps 30000 \
    --save_steps 2000 \
    --learning_rate 2e-4 \
    --min_lr 1e-5 \
    --warmup_steps 0.01 \
    --save_strategy steps \
    --bf16 \
    --hidden_size ${HIDDEN_SIZE} \
    --intermediate_size ${INTERMEDIATE_SIZE} \
    --num_attention_heads ${NUM_ATTENTION_HEADS} \
    --num_key_value_heads ${NUM_KEY_VALUE_HEADS} \
    --num_layers ${NUM_LAYERS} \
    --model_max_length 4096 \
    --max_seq_length 4096 \
    --use_qk_norm \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 | tee ${log_file}
