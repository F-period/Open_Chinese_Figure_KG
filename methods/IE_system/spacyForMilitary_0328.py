import os
import re

import spacy as sp
from spacy import Language
from spacy.tokens import Span
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sp.require_gpu()


# 根据正则规定实体
@Language.component("get_military_terms")
def get_military_terms(doc):
    expression = r"[\w]+-[\w]+"
    # expression = r"[A-Z]+-.*?[\u673a]"
    new_span = []
    chars_to_tokens = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token.text)):
            chars_to_tokens[i] = token.i

    for match in re.finditer(expression, doc.text, re.A):
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            # print("Found match:", span.text, type(span))
            new_span.append(Span(doc, span.start, span.end, label='military'))
        else:
            start_token = chars_to_tokens.get(start)
            end_token = chars_to_tokens.get(end)
            if start_token is not None and end_token is not None:
                span = doc[start_token:end_token + 1]
                # print("Found closest match:", span.text, type(span))
    # print(new_span)
    doc.ents = new_span
    return doc


# 加载模型
nlp = sp.load("zh_core_web_trf")
nlp.add_pipe("get_military_terms", after="ner")
nlp.add_pipe("merge_entities")

