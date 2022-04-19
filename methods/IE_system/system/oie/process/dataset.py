import os

from tqdm import tqdm


def data_loader(input_fp):
    with open(input_fp, "r", encoding='utf-8') as f:
        print("processing 9137 data")
        for line in tqdm(f.readlines()):
            sample = eval(line)
            yield sample['text']