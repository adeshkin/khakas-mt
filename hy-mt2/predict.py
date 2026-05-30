from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name= "adeshkin/Hy-MT2-1.8B-kjh-ru-lora"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

src_lang = "Khakas"
tgt_lang = "Russian"
text = '54. "Ат ӱгредерде арғамҷың пик ползын, чонға чоохтирда чооғың сын ползын" сӧспектің тузазын чарыда пас пиріңер.'
prompt = f"Translate the following {src_lang} text into {tgt_lang}, output only the translation result without additional explanation:\n\n{text}"

messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

model.eval()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
    )

result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(result)
# 54. Объясните значение пословицы "Чтобы научить коня, привязь твоя должна быть крепкой, чтобы народу говорить - твой рассказ был правдивым."
