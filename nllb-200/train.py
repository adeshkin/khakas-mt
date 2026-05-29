from datasets import load_dataset
import random
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor
import os
import sacrebleu
from tqdm.auto import trange
import numpy as np
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from google.colab import userdata
import gc
import re


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def get_word_num(text):
    return len(re.findall(r'(\w+|[^\w\s])', text))


def filter_func(examples):
    results = []
    for kjh_sent, ru_sent in zip(examples['kjh'], examples['ru']):
        cond1 = kjh_sent is not None and len(kjh_sent) >= 5 and get_word_num(kjh_sent) <= 64
        cond2 = ru_sent is not None and len(ru_sent) >= 5 and get_word_num(ru_sent) <= 64
        results.append(cond1 and cond2)

    return results


def prepare_data_train(ds):
    ds = ds.select_columns(['kjh', 'ru'])
    ds = ds.filter(filter_func, batched=True, num_proc=4)
    corpus = [{'kjh_Cyrl': row['kjh'], 'rus_Cyrl': row['ru']} for row in ds]
    assert len(corpus) == len(ds)

    return corpus


def prepare_data():
    train_corpus = []
    ds_krpc = load_dataset('adeshkin/khakas-russian-parallel-corpus', split='train')
    train_corpus.extend(prepare_data_train(ds_krpc))

    ds_smoldoc = load_dataset('adeshkin/google-smol-en-ru-kjh', name='smoldoc', split='train')
    train_corpus.extend(prepare_data_train(ds_smoldoc))

    ds_smolsent = load_dataset('adeshkin/google-smol-en-ru-kjh', name='smolsent', split='train')
    train_corpus.extend(prepare_data_train(ds_smolsent))

    dev_corpus = prepare_data_flores(split='dev')
    devtest_corpus = prepare_data_flores(split='devtest')

    return train_corpus, dev_corpus, devtest_corpus


def prepare_model(tokenizer, model_name, similar_lang_code, new_lang_code):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    embeds = model.model.shared.weight.data

    nllb_tokenizer = NllbTokenizer.from_pretrained(model_name)
    similar_lang_id = nllb_tokenizer.convert_tokens_to_ids(similar_lang_code)
    new_lang_id = tokenizer.convert_tokens_to_ids(new_lang_code)
    embeds[new_lang_id] = embeds[similar_lang_id]

    moved_tokens = [x.content for x in nllb_tokenizer.added_tokens_decoder.values()]
    moved_ids = nllb_tokenizer.convert_tokens_to_ids(moved_tokens)

    embeds[tokenizer.convert_tokens_to_ids(moved_tokens)] = embeds[moved_ids]

    added_vocab = set(tokenizer.get_vocab().keys()).difference(set(nllb_tokenizer.get_vocab().keys()))
    for t in added_vocab:
        if t == new_lang_code:
            continue

        if t in moved_tokens:
            continue

        tt = nllb_tokenizer(t, add_special_tokens=False).input_ids
        if len(tt) == 0:
            continue

        if nllb_tokenizer.unk_token_id in tt:
            continue

        idx = tokenizer.convert_tokens_to_ids(t)
        embeds[idx] = embeds[tt].mean(0)

    return model


def get_batch_pairs(batch_size, data, langs):
    src_lang, tgt_lang = np.random.choice(langs, size=2, replace=False, p=[0.6, 0.4])
    xx, yy = [], []
    for _ in range(batch_size):
        example = random.choice(data)
        xx.append(example[src_lang])
        yy.append(example[tgt_lang])
    return xx, yy, src_lang, tgt_lang


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


def save_best_model(model, i, eval_results, best_results, langs_first_second, model_dir):
    if (eval_results[langs_first_second]['chrf++'] >
            best_results[langs_first_second]['chrf++']):
        model_path = f"{model_dir}/best-{langs_first_second}"
        model.save_pretrained(model_path)

        print(
            f"[Best] New best for {langs_first_second}: step {i}, chrf++ {eval_results[langs_first_second]['chrf++']:.2f}")
        print(f"[*] Saved best model: {os.path.basename(model_path)}")

        best_results[langs_first_second] = eval_results[langs_first_second]

        with open(f'{model_path}-{i}-metrics.json', 'w') as f:
            json.dump(best_results, f, indent=2)

    return best_results


def main():
    set_seed(19)
    train_corpus, dev_corpus, devtest_corpus = prepare_data()
    langs = ['kjh_Cyrl', 'rus_Cyrl']

    batch_size = 16
    max_length = 128
    num_beams = 4
    warmup_steps = 1_000
    train_steps = 200_000
    eval_steps = 5_000
    accum_steps = 2
    log_steps = 100

    exp_dir = '/content/drive/MyDrive/experiments/khakas-mt'
    writer = SummaryWriter(f'{exp_dir}/logs')
    model_dir = f'{exp_dir}/checkpoints'
    os.makedirs(model_dir, exist_ok=True)
    tokenizer = NllbTokenizer.from_pretrained(f'{exp_dir}/tokenizer_kjh_Cyrl')

    model_name = 'facebook/nllb-200-distilled-600M'
    new_lang_code = 'kjh_Cyrl'
    similar_lang_code = 'kaz_Cyrl'  # 'kir_Cyrl'
    model = prepare_model(tokenizer, model_name, similar_lang_code, new_lang_code)
    model.cuda()

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3
    )
    optimizer.zero_grad(set_to_none=True)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=train_steps // accum_steps)
    scaler = torch.amp.GradScaler(device='cuda', enabled=True)
    model.train()
    losses = []
    best_results = None
    for i in trange(0, train_steps):
        xx, yy, src_lang, tgt_lang = get_batch_pairs(batch_size, train_corpus, langs)
        try:
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang

            inputs = tokenizer(
                text=xx,
                text_target=yy,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(model.device)

            inputs["labels"][inputs["labels"] == tokenizer.pad_token_id] = -100

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = model(**inputs).loss
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0 and i > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            losses.append(loss.item() * accum_steps)

        except RuntimeError as e:  # usually, it is out-of-memory
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            cleanup()
            print(f"Error: max_len={max(len(s) for s in xx + yy)} | {e}")
            continue

        except KeyboardInterrupt as e:
            print('\nKeyboardInterrupt detected.')
            break

        if i % log_steps == 0 and i > 0:
            writer.add_scalar("train/loss", losses[-1], i)
            writer.add_scalar(f"train/loss_mean_last_{log_steps}", np.mean(losses[-log_steps:]), i)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], i)

            print(f"step {i}\nloss = {losses[-1]:.4f} | loss_mean_last_{log_steps} = {np.mean(losses[-log_steps:]):.4f}\n")

        if i % eval_steps == 0 and i > 0:
            cleanup()
            model.eval()
            last_model_path = f"{model_dir}/last"
            model.save_pretrained(last_model_path)

            eval_results = evaluate(dev_corpus, tokenizer, model, batch_size, num_beams, langs)
            with open(f'{last_model_path}-{i}-metrics.json', 'w') as f:
                json.dump(eval_results, f, indent=2)

            langs_first_second = f'{langs[0]}-{langs[1]}'
            langs_second_first = f'{langs[1]}-{langs[0]}'

            writer.add_scalar(f"eval/bleu_{langs_first_second}", eval_results[f'{langs_first_second}']['bleu'], i)
            writer.add_scalar(f"eval/chrf++_{langs_first_second}", eval_results[f'{langs_first_second}']['chrf++'], i)
            writer.add_scalar(f"eval/bleu_{langs_second_first}", eval_results[f'{langs_second_first}']['bleu'], i)
            writer.add_scalar(f"eval/chrf++_{langs_second_first}", eval_results[f'{langs_second_first}']['chrf++'], i)
            writer.flush()

            if best_results is None:
                best_results = eval_results

            best_results = save_best_model(model, i, eval_results, best_results, langs_first_second, model_dir)
            best_results = save_best_model(model, i, eval_results, best_results, langs_second_first, model_dir)

            eval_str = "\n".join([f"{k}: bleu = {v['bleu']:.1f}, chrf++ = {v['chrf++']:.1f}"
                                  for k, v in eval_results.items()])
            print(
                f"step {i}\nloss = {losses[-1]:.4f} | loss_mean_last_{eval_steps} = {np.mean(losses[-eval_steps:]):.4f}\n{eval_str}\n")

            model.train()

    writer.close()


if __name__ == '__main__':
    main()
