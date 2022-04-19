from pathlib import Path
import torch
from torch import nn
import random

# 【基于模型的预测结果】

# 【目标分类的列表及其对于编号】
main_tags = ['药物', '疾病', '症状', '检查科目', '细菌', '病毒', '医学专科']
main_pred_tags = {'药物': 0, '疾病': 1, '症状': 2, '检查科目': 3, '细菌': 4, '病毒': 5, '医学专科': 6}

# 【对模型结构进行一个投票】
def do_avg_vote(logit_paths, output_path, avg_th=0.5, tags=None, pred_tags=None):
    data = []
    f_sigmoid = nn.Sigmoid()
    for f_path in logit_paths:
        # 【逐次遍历每个要考虑的模型结果取出相应的logits】
        cur_data = []
        with open(f_path, 'r', encoding='utf-8') as input:
            for line in input:
                items = line.rstrip('\n').split('\t')
                cur_data.append((items[0],
                                f_sigmoid(torch.Tensor([float(num) for num in items[1:]]))
                                 ))
        data.append(cur_data)

    # replace default None with meaningful parameters
    # 【如果没结果，用该模型预测出的结果替换】
    if tags is None:
        tags = main_tags
    if pred_tags is None:
        pred_tags = main_pred_tags

    vote_pred = []
    data_num = len(data)
    total = len(data[0])
    num_tags = len(pred_tags)
    # 【对每个实体做下列操作】
    for i in range(total):
        cur_word = data[0][i][0]

        cur = torch.Tensor([0.0] * num_tags)
        for j in range(data_num):
            cur += data[j][i][1]
        cur /= float(data_num)

        mx, idx = cur.max(-1)
        mx = mx.item()
        idx = idx.item()

        # 【如果大于阈值，采纳当前结果，否则视为无类型】
        if mx >= avg_th:
            vote_pred.append((cur_word, tags[idx]))
        else:
            vote_pred.append((cur_word, 'NoneType'))

    # 【写入预测结果】
    with open(output_path, 'w', encoding='utf-8') as predict_file:
        for k, v in vote_pred:
            predict_file.write(f"{k}\t{v}\n")


# 【对模型进行投票】
def do_ensemble():
    # test try 19
    # 【logits: 未进入softmax的概率】
    # 【以下是全部考虑的模型】
    vote_paths = [
        './pred_data/logits_test_m1_e6.txt',
        './pred_data/logits_test_m1_e26.txt',
        './pred_data/logits_test_m2_e42.txt',
        './pred_data/logits_test_m3_e17.txt',
        './pred_data/logits_test_m4_e1.txt',
        './pred_data/logits_test_m4_e2.txt',
        './pred_data/logits_test_m5_e4.txt',
        './pred_data/logits_test_m6_e2.txt',
        './pred_data/logits_test_m7_e13.txt',
        './pred_data/logits_test_m7_e17.txt',
        './pred_data/logits_test_m8_e74.txt',
        './pred_data/logits_test_m8_e89.txt',
        './pred_data/logits_test_m9_e73.txt',
        './pred_data/logits_test_m10_e54.txt',
    ]
    # 【结果是否可靠的概率阈值】
    vote_th = 0.5
    # 【投票结束的预测结果】
    vote_output = './pred_data/results_test_model_vote.txt'

    do_avg_vote(vote_paths, vote_output, avg_th=vote_th)

if __name__ == '__main__':
    do_ensemble()

    pass