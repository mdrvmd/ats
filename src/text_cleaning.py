import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\b(company name|city|state)\b", '', text)   # удалить company name, city, state
    text = re.sub(r'[^a-zа-яё0-9+#. ]', ' ', text)   # заменить символы на пробел
    text = re.sub(r'\s+', ' ', text)   # группы пробелов на один пробел
    text = text.strip()
    return text

