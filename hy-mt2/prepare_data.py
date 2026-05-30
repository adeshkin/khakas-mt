from datasets import load_dataset
import re
import random
import json


def get_word_num(text):
    return len(re.findall(r'(\w+|[^\w\s])', text))


def filter_func(examples):
    results = []
    for kjh_sent, ru_sent in zip(examples['kjh'], examples['ru']):
        cond1 = kjh_sent is not None and len(kjh_sent) >= 5 and get_word_num(kjh_sent) <= 64
        cond2 = ru_sent is not None and len(ru_sent) >= 5 and get_word_num(ru_sent) <= 64
        results.append(cond1 and cond2)

    return results


def prepare_ds_train(ds):
    ds = ds.select_columns(['kjh', 'ru'])
    ds = ds.filter(filter_func, batched=True, num_proc=4)
    corpus = [{'kjh_Cyrl': row['kjh'], 'rus_Cyrl': row['ru']} for row in ds]
    assert len(corpus) == len(ds)

    return corpus


def prepare_corpus():
    train_corpus = []
    ds_krpc = load_dataset('adeshkin/khakas-russian-parallel-corpus', split='train')
    train_corpus.extend(prepare_ds_train(ds_krpc))

    ds_smoldoc = load_dataset('adeshkin/google-smol-en-ru-kjh', name='smoldoc', split='train')
    train_corpus.extend(prepare_ds_train(ds_smoldoc))

    ds_smolsent = load_dataset('adeshkin/google-smol-en-ru-kjh', name='smolsent', split='train')
    train_corpus.extend(prepare_ds_train(ds_smolsent))

    return train_corpus


def main():
    train_corpus = prepare_corpus()

    inputs = []
    for example in train_corpus:
        kjh_sent = example['kjh_Cyrl']
        rus_sent = example['rus_Cyrl']
        inputs.append({"messages": [{"role": "user", "content":
            f"Translate the following Khakas text into Russian, output only the translation result without additional explanation:\n\n{kjh_sent}"},
                                    {"role": "assistant", "content": f"{rus_sent}"}]})
        inputs.append({"messages": [{"role": "user", "content":
            f"Translate the following Russian text into Khakas, output only the translation result without additional explanation:\n\n{rus_sent}"},
                                    {"role": "assistant", "content": f"{kjh_sent}"}]})

    random.shuffle(inputs)

    with open('../experiments/khakas_russian_train_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in inputs:
            json.dump(entry, f)
            f.write('\n')


if __name__ == '__main__':
    main()
