for i in {2000..24000..2000}
do
    echo "Processing checkpoint-$i..."
    python evaluate.py \
        --model_path ../experiments/hy-mt2-1.8b-kjh-ru-finetune/checkpoint-${i}_merged_hy_lora_weight
done