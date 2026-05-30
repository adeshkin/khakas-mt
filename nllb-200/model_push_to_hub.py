from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from google.colab import userdata


def main():
    tokenizer = NllbTokenizer.from_pretrained('/content/drive/MyDrive/experiments/khakas-mt/tokenizer_kjh_Cyrl')
    model_path = '/content/drive/MyDrive/experiments/khakas-mt/checkpoints/last'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    repo_id = 'adeshkin/nllb-200-distilled-600M-kjh-ru'
    model.push_to_hub(repo_id, token=userdata.get('WRITE_HF_TOKEN'))
    tokenizer.push_to_hub(repo_id, token=userdata.get('WRITE_HF_TOKEN'))


if __name__ == "__main__":
    main()
