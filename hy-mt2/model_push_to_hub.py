from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from dotenv import load_dotenv

load_dotenv()


def main():
    tokenizer = AutoTokenizer.from_pretrained("tencent/Hy-MT2-1.8B", trust_remote_code=True)
    model_path = '../experiments/hy-mt2-1.8b-kjh-ru-lora-finetune/checkpoint-24000_merged_hy_lora_weight'

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    repo_id = 'adeshkin/Hy-MT2-1.8B-kjh-ru-lora'
    model.push_to_hub(repo_id, token=os.getenv('WRITE_HF_TOKEN'))
    tokenizer.push_to_hub(repo_id, token=os.getenv('WRITE_HF_TOKEN'))


if __name__ == "__main__":
    main()
