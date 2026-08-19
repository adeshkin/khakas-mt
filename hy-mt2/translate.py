import requests


def translate_requests(text):
    url = "http://localhost:8000/v1/chat/completions"
    src_lang = "Khakas"
    tgt_lang = "Russian"
    text = '54. "Ат ӱгредерде арғамҷың пик ползын, чонға чоохтирда чооғың сын ползын" сӧспектің тузазын чарыда пас пиріңер.'
    prompt = f"Translate the following {src_lang} text into {tgt_lang}, output only the translation result without additional explanation:\n\n{text}"
    payload = {
        "model": "adeshkin/Hy-MT2-1.8B-kjh-ru-lora",
        "messages": [
            {"role": "user", "content": prompt}
        ],    }

    response = requests.post(url, json=payload)
    return response.json()['choices'][0]['message']['content']


print(translate_requests('54. "Ат ӱгредерде арғамҷың пик ползын, чонға чоохтирда чооғың сын ползын" сӧспектің тузазын чарыда пас пиріңер.'))
