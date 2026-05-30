# based on https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865

# pip install transformers==4.57.3

import os
import json
from collections import Counter
import shutil
import sentencepiece as spm
from datasets import load_dataset
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import NllbTokenizer
from tqdm.auto import tqdm
import re
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer

mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]


def get_non_printing_char_replacer(replace_by: str = " "):
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


replace_nonprint = get_non_printing_char_replacer(" ")


# based on https://github.com/facebookresearch/stopes/blob/2be1bb8f67a38588eef3dfce204679d180a2c921/stopes/pipelines/monolingual/monolingual_line_processor.py#L214
def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)

    return clean


def update_sp_model(model_name, spm_prefix_path, output_dir):
    tkn_dir_temp1 = f'{output_dir}/tokenizer_temp1'
    assert not os.path.isdir(tkn_dir_temp1)
    tkn_dir_temp2 = f'{output_dir}/tokenizer_temp2'
    os.makedirs(tkn_dir_temp2, exist_ok=False)
    tkn_dir = f'{output_dir}/tokenizer'
    assert not os.path.isdir(tkn_dir)

    tokenizer = NllbTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tkn_dir_temp1)

    additional_special_tokens = tokenizer.additional_special_tokens

    old_spm = sp_pb2_model.ModelProto()
    old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

    sp_trained = spm.SentencePieceProcessor(model_file=f'{spm_prefix_path}.model')
    added_spm = sp_pb2_model.ModelProto()
    added_spm.ParseFromString(sp_trained.serialized_model_proto())

    old_tokens_set = {p.piece for p in old_spm.pieces}
    prev_min_score = old_spm.pieces[-1].score

    added_tokens = []
    for p in added_spm.pieces:
        if p.type != 1:
            continue
        piece = p.piece
        if piece not in old_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = p.score + prev_min_score
            old_spm.pieces.append(new_p)
            added_tokens.append(piece)

    new_tok_spm_path = f'{output_dir}/new_spm_nllb.model'
    with open(new_tok_spm_path, 'wb') as f:
        f.write(old_spm.SerializeToString())

    with open(f"{tkn_dir_temp1}/tokenizer_config.json", "r") as f:
        cfg = json.load(f)
    cfg["added_tokens_decoder"] = {
        k: v
        for k, v in cfg["added_tokens_decoder"].items()
        if k in ["0", "1", "2", "3"]
    }

    cfg["additional_special_tokens"] = []
    with open(f"{tkn_dir_temp2}/tokenizer_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    shutil.copy(new_tok_spm_path, f"{tkn_dir_temp2}/sentencepiece.bpe.model")

    tokenizer = NllbTokenizer.from_pretrained(tkn_dir_temp2)
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens},
                                 replace_additional_special_tokens=True)

    no_added_tokens = set(added_tokens).difference(set(tokenizer.get_vocab().keys()))
    assert len(no_added_tokens) == 0

    tokenizer.save_pretrained(tkn_dir)

    return tkn_dir


def add_new_lang_token(tkn_dir, new_lang_code):
    tkn_dir_temp = f'{tkn_dir}_{new_lang_code}_temp'
    assert not os.path.isdir(tkn_dir_temp)

    tkn_final_dir = f'{tkn_dir}_{new_lang_code}'
    assert not os.path.isdir(tkn_final_dir)

    tokenizer = NllbTokenizer.from_pretrained(tkn_dir)
    tok_len_prev = len(tokenizer)
    special_tokens_prev = [x.content for x in tokenizer.added_tokens_decoder.values()]

    assert new_lang_code not in special_tokens_prev
    special_ids_prev = tokenizer.convert_tokens_to_ids(special_tokens_prev)

    tokenizer.add_special_tokens({"additional_special_tokens": [new_lang_code]},
                                 replace_additional_special_tokens=False)
    new_lang_id = tokenizer.convert_tokens_to_ids(new_lang_code)
    assert tok_len_prev + 1 == len(tokenizer)
    assert special_ids_prev == tokenizer.convert_tokens_to_ids(special_tokens_prev)
    assert new_lang_code in tokenizer.additional_special_tokens
    assert new_lang_id == len(tokenizer) - 1
    assert tokenizer.convert_ids_to_tokens([len(tokenizer) - 3, len(tokenizer) - 2, new_lang_id]) == ['zho_Hant',
                                                                                                      'zul_Latn',
                                                                                                      new_lang_code]
    assert tokenizer.convert_tokens_to_ids(['zho_Hant', 'zul_Latn', new_lang_code]) == [len(tokenizer) - 3,
                                                                                        len(tokenizer) - 2,
                                                                                        new_lang_id]

    tokenizer.save_pretrained(tkn_dir_temp)

    new_tokenizer = NllbTokenizer.from_pretrained(tkn_dir_temp)
    new_tokenizer.save_pretrained(tkn_final_dir)

    return tkn_final_dir


def check_tokenizer(model_name, tkn_final_dir, new_lang_code):
    tokenizer = NllbTokenizer.from_pretrained(model_name)
    special_tokens_prev = ["<s>", "<pad>", "</s>", "<unk>"]

    all_tokens_prev = list(tokenizer.get_vocab().keys())
    all_special_tokens_prev = [x.content for x in tokenizer.added_tokens_decoder.values()]
    tokens_prev = list(set(all_tokens_prev).difference(set(all_special_tokens_prev)))

    special_ids_prev = tokenizer.convert_tokens_to_ids(special_tokens_prev)

    new_tokenizer = NllbTokenizer.from_pretrained(tkn_final_dir)
    new_lang_id = new_tokenizer.convert_tokens_to_ids(new_lang_code)
    all_special_tokens = [x.content for x in new_tokenizer.added_tokens_decoder.values()]

    for i in range(0, len(tokens_prev), 1000):
        token_batch = tokens_prev[i:i + 1000]
        ids_prev = tokenizer.convert_tokens_to_ids(token_batch)
        assert ids_prev == new_tokenizer.convert_tokens_to_ids(token_batch)

    assert special_ids_prev == new_tokenizer.convert_tokens_to_ids(special_tokens_prev)
    assert "<mask>" in all_special_tokens_prev
    assert "<mask>" in all_special_tokens
    assert new_lang_code not in tokenizer.additional_special_tokens
    assert new_lang_code in new_tokenizer.additional_special_tokens
    assert new_lang_id == len(new_tokenizer) - 1


def main():
    save_dir = '/content/drive/MyDrive/experiments/khakas-mt'
    os.makedirs(save_dir, exist_ok=False)
    new_lang_code = 'kjh_Cyrl'
    dataset_name = 'adeshkin/kjh-mono-sents'
    model_name = 'facebook/nllb-200-distilled-600M'

    text_path = f'{save_dir}/kjh-mono-sents.txt'

    ds = load_dataset(dataset_name, split='train')

    def filter_func(examples):
        kjh_sent = examples['kjh']
        return kjh_sent is not None and len(kjh_sent) >= 5

    ds = ds.filter(filter_func)

    all_texts = ds['kjh']
    all_text_normalized = [preproc(t) for t in tqdm(all_texts)]
    chars_cnt = Counter(c for t in all_text_normalized for c in t)
    required_chars = ''.join([
        k for k, v in chars_cnt.most_common()
        if v >= 3 and k not in ' '
    ])
    print(f"Total characters: {len(chars_cnt)}, required: {len(required_chars)}")

    with open(text_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text_normalized))

    spm_prefix_path = f"{save_dir}/spm_kjh_16k"
    assert not os.path.exists(f'{spm_prefix_path}.model')

    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=spm_prefix_path,
        vocab_size=2 ** 14,  # 16K
        character_coverage=1,
        num_threads=16,
        train_extremely_large_corpus=False,
        add_dummy_prefix=False,
        max_sentencepiece_length=128,
        max_sentence_length=4192 * 4,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        required_chars=required_chars,
    )

    tkn_dir = update_sp_model(model_name, spm_prefix_path, save_dir)
    tkn_final_dir = add_new_lang_token(tkn_dir, new_lang_code)

    check_tokenizer(model_name, tkn_final_dir, new_lang_code)

    print(f"Success! Tokenizer saved to: {tkn_final_dir}")


if __name__ == '__main__':
    main()
