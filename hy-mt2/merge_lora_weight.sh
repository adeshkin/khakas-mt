# based on https://github.com/Tencent-Hunyuan/Hy-MT2/blob/c30b36c59b19252dae7f3afca34d8478dbe67de9/train/deepspeed_support/merge_lora_weight.sh

for i in {2000..22000..2000}
do
    echo "Processing checkpoint-$i..."
    python merge_lora_weight.py \
        --base_model_path tencent/Hy-MT2-1.8B \
        --adapter_model_path ./experiments/hy-mt2-1.8b-kjh-ru-finetune/checkpoint-$i \
        --output_path ./experiments/hy-mt2-1.8b-kjh-ru-finetune/checkpoint-${i}_merged_hy_lora_weight \
        --save_dtype fp32
done