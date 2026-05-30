from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
from tqdm.auto import trange
import sacrebleu
import json
from google.colab import userdata


def prepare_data_flores(split):
    ds_kjh = load_dataset('openlanguagedata/flores_plus', name='kjh_Cyrl', split=split,
                          token=userdata.get('READ_HF_TOKEN'))
    ds_rus = load_dataset('openlanguagedata/flores_plus', name='rus_Cyrl', split=split,
                          token=userdata.get('READ_HF_TOKEN'))
    rus_dict = {row['id']: row['text'] for row in ds_rus}
    corpus = []
    for row in ds_kjh:
        idx = row['id']
        if idx in rus_dict:
            corpus.append({'kjh_Cyrl': row['text'], 'rus_Cyrl': rus_dict[idx]})

    assert len(corpus) == len(rus_dict)

    return corpus


def translate(text, tokenizer, model, src_lang='rus_Cyrl', tgt_lang='eng_Latn', a=32, b=3, max_input_length=1024,
              num_beams=4, **kwargs):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    model.eval()
    with torch.no_grad():
        result = model.generate(
            **inputs.to(model.device),
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
            num_beams=num_beams,
            **kwargs
        )
    return tokenizer.batch_decode(result, skip_special_tokens=True)


def batched_translate(texts, tokenizer, model, batch_size=16, **kwargs):
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in trange(0, len(texts2), batch_size):
        results.extend(translate(texts2[i: i + batch_size], tokenizer, model, **kwargs))
    return [p for i, p in sorted(zip(idxs, results))]


def compute_metrics(dev_corpus, tokenizer, model, batch_size, num_beams, src_lang, tgt_lang):
    bleu_calc = sacrebleu.BLEU()
    chrf_calc = sacrebleu.CHRF(word_order=2)

    texts = [x[src_lang] for x in dev_corpus]
    preds = batched_translate(texts, tokenizer, model, batch_size, src_lang=src_lang, tgt_lang=tgt_lang,
                              num_beams=num_beams)
    refs = [x[tgt_lang] for x in dev_corpus]

    chrfpp_value = chrf_calc.corpus_score(preds, [refs])
    bleu_value = bleu_calc.corpus_score(preds, [refs])

    return bleu_value.score, chrfpp_value.score


def evaluate(dev_corpus, tokenizer, model, batch_size, num_beams, langs):
    eval_results = {}

    src_lang = langs[0]
    tgt_lang = langs[1]
    bleu_value, chrfpp_value = compute_metrics(dev_corpus, tokenizer, model, batch_size, num_beams, src_lang, tgt_lang)
    eval_results[f'{src_lang}-{tgt_lang}'] = {'bleu': bleu_value, 'chrf++': chrfpp_value}

    src_lang = langs[1]
    tgt_lang = langs[0]
    bleu_value, chrfpp_value = compute_metrics(dev_corpus, tokenizer, model, batch_size, num_beams, src_lang, tgt_lang)
    eval_results[f'{src_lang}-{tgt_lang}'] = {'bleu': bleu_value, 'chrf++': chrfpp_value}

    return eval_results


def main():
    batch_size = 16
    num_beams = 4
    langs = ['kjh_Cyrl', 'rus_Cyrl']

    devtest_corpus = prepare_data_flores(split='devtest')
    model_name = 'adeshkin/nllb-200-distilled-600M-kjh-ru'
    tokenizer = NllbTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.cuda()

    eval_results = evaluate(devtest_corpus, tokenizer, model, batch_size, num_beams, langs)
    eval_str = "\n".join([f"{k}: bleu = {v['bleu']:.1f}, chrf++ = {v['chrf++']:.1f}"
                          for k, v in eval_results.items()])
    print(eval_str)
    print()
    output_path = 'nllb-200_devtest_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Metrics saved to {output_path}")


if __name__ == '__main__':
    main()
