import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import sacrebleu
from datasets import load_dataset
import json
from dotenv import load_dotenv

load_dotenv()


def translate(model, tokenizer, text, src_lang, tgt_lang):
    prompt = f"Translate the following {src_lang} text into {tgt_lang}, output only the translation result without additional explanation:\n\n{text}"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    return response


def compute_metrics(dev_corpus, tokenizer, model, src_lang, tgt_lang):
    bleu_calc = sacrebleu.BLEU()
    chrf_calc = sacrebleu.CHRF(word_order=2)

    texts = [x[src_lang] for x in dev_corpus]
    preds = [translate(model, tokenizer, x, src_lang, tgt_lang) for x in tqdm(texts)]
    refs = [x[tgt_lang] for x in dev_corpus]

    chrfpp_value = chrf_calc.corpus_score(preds, [refs])
    bleu_value = bleu_calc.corpus_score(preds, [refs])

    return bleu_value.score, chrfpp_value.score


def prepare_data_flores(split):
    ds_kjh = load_dataset('openlanguagedata/flores_plus', name='kjh_Cyrl', split=split,
                          token=os.getenv('READ_HF_TOKEN'))
    ds_rus = load_dataset('openlanguagedata/flores_plus', name='rus_Cyrl', split=split,
                          token=os.getenv('READ_HF_TOKEN'))
    rus_dict = {row['id']: row['text'] for row in ds_rus}
    corpus = []
    for row in ds_kjh:
        idx = row['id']
        if idx in rus_dict:
            corpus.append({'Khakas': row['text'], 'Russian': rus_dict[idx]})

    assert len(corpus) == len(rus_dict)

    return corpus


def main():
    model_name = 'adeshkin/Hy-MT2-1.8B-kjh-ru-lora'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # model.eval()
    dev_corpus = prepare_data_flores('devtest')
    eval_results = {}

    src_lang = 'Khakas'
    tgt_lang = 'Russian'
    bleu_value, chrfpp_value = compute_metrics(dev_corpus, tokenizer, model, src_lang, tgt_lang)
    eval_results[f'{src_lang}-{tgt_lang}'] = {'bleu': bleu_value, 'chrf++': chrfpp_value}
    print(f'{src_lang}-{tgt_lang}', f"bleu = {bleu_value:.1f}, chrf++ = {chrfpp_value:.1f}")

    src_lang = 'Russian'
    tgt_lang = 'Khakas'
    bleu_value, chrfpp_value = compute_metrics(dev_corpus, tokenizer, model, src_lang, tgt_lang)
    print(f'{src_lang}-{tgt_lang}', f"bleu = {bleu_value:.1f}, chrf++ = {chrfpp_value:.1f}")
    eval_results[f'{src_lang}-{tgt_lang}'] = {'bleu': bleu_value, 'chrf++': chrfpp_value}
    eval_str = "\n".join([f"{k}: bleu = {v['bleu']:.1f}, chrf++ = {v['chrf++']:.1f}"
                          for k, v in eval_results.items()])
    print(eval_str)
    print()
    output_path = 'hy-mt2_devtest_metrics.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)

    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
