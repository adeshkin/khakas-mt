from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = "adeshkin/nllb-200-distilled-600M-kjh-ru"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

src_lang = "kjh_Cyrl"  # Khakas
tgt_lang = "rus_Cyrl"  # Russian
text = '54. "Ат ӱгредерде арғамҷың пик ползын, чонға чоохтирда чооғың сын ползын" сӧспектің тузазын чарыда пас пиріңер.'

tokenizer.src_lang = src_lang
tokenizer.tgt_lang = tgt_lang
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024)

model.eval()
with torch.no_grad():
    outputs = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(32 + 3 * inputs.input_ids.shape[1]),
        num_beams=4,
    )
result = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
print(result)
# 54. Объясните значение пословицы "При обучении коня пусть будет крепкий верёв твой, при обращении к народу пусть будет истинно слово твоё."
