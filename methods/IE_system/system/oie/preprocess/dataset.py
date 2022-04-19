import os

from tqdm import tqdm


def data_loader(input_fp):
    with open(input_fp, "r", encoding='utf-8') as f:
        for line in f.readlines():
            sample = eval(line)
            yield sample['text']

