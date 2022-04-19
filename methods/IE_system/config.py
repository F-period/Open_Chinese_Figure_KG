import spacy

spacy.require_gpu()
nlp = spacy.load("zh_core_web_trf")

