import html
import re


def preprocess_markdown(text):
    new_text = html.unescape(text)
    return re.sub(r"(?<=\S)  +(?=\S)", r" ", new_text)
