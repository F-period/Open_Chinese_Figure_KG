#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time:    2020/9/29
# @Author:  kerrzwu

# 【基于规则的生成成果】

# 【通过正则表达式进行类型推断】
import re
import numpy as np
from collections import defaultdict

# 【类型推断的结果】
ccks_label_names = ["病毒",
                    "细菌",
                    "疾病",
                    "药物",
                    "医学专科",
                    "检查科目",
                    "症状"]

# 【同名转换，映射到类型】
name_mapping = {"病因": "病毒",
                "药物": "药物",
                "成份": "药物",
                "商品名": "药物",
                "治疗": "药物",

                "检查": "检查科目",
                "实验室检查": "检查科目",
                "诊疗": "检查科目",

                "症状体征": "症状",
                "疾病": "疾病",
                "科室": "医学专科"}

# 【根据实体名进行正则匹配推断类型】
# 【这里主要是根据名称的尾缀进行一个直接的匹配】
def rule_factory(names, preds):
    for i in range(len(names)):

        cur_name = names[i]

        if re.match('.*(菌属|霉)$', cur_name):
            preds[i] = '细菌'

        if re.match('.*(噬菌体)$', cur_name):
            preds[i] = '病毒'

        if re.match('.*(病|综合征|炎|瘤|症|癌|癣|囊肿|中毒|青光眼)$', cur_name):
            preds[i] = '疾病'

        if re.match('.*科$', cur_name):
            preds[i] = '医学专科'

        if re.match('.*(实验|检查|测定|抗体|试验|计数|抗原)$', cur_name):
            preds[i] = '检查科目'

        if re.match('.*(片|尼|散|胶囊|颗粒|丸|剂|膏|汀|胶|糖浆|药|酊|茶|口服液|钠|丁|酮|酚|胺|西林|宁)$', cur_name) \
                or re.search('复方|注射用|注射液|口服液|球蛋白|沙星|盐酸|试剂盒|药盒', cur_name):
            preds[i] = '药物'

# 【输入:两个带类型的实体txt】
# 【输出 {实体:(类型)}的defaultdict词典】
def kg_build():
    # 【这个csv 是实体名】
    tmail_kg = "./rules/tmail_kg.csv"
    # 【同样是是实体名，类别，类边中含有NoneType】
    c_new_data = "./rules/query_on_c_test.txt"
    kg_raw_data = open(tmail_kg, 'r', encoding='utf-8')
    # 【defaltdict: 如果查找的key不存在，返回默认值set()】
    kg_dict = defaultdict(set)

    for line in kg_raw_data.readlines():
        line = line.strip('\n').split(',')
        # 【head_name：实体名】
        head_name = line[0]
        # 【字典的值: 实体的类型，列表，可能有多个】
        kg_dict[head_name] = set(line[1:])

    with open(c_new_data, 'r', encoding='utf-8') as f:

        for line in f.readlines():
            line = line.strip('\n').split(',')
            # 【name：实体名】
            name = line[0]
            # 【preds：预测的类型名】
            preds = line[1]

            # 【对于并非NoneType的类型，加入词典】
            if preds != "NoneType":
                if len(kg_dict[name]) == 0:
                    kg_dict[name].add(preds)

    return kg_dict


def rule_kg():
    # 【测试数据: 实体名称】
    test_dir = "./rules/entity_test.txt"
    # 【训练数据: 带类型的实体名称】
    train_dir = "./rules/entity_type.txt"
    # 【结果数据：规则预测结果】
    result_dir = './rules/pre_rules.txt'
    
    # 【先构造知识图谱字典】
    kg_dict = kg_build()
    
    # 【测试数据预处理】
    with open(test_dir, 'r', encoding='utf-8') as f:
        # 【实体列表：去除\n、\t，获得实体名列表】
        test_data = [elem.strip('\n') for elem in f.readlines()]
        test_data = np.array([elem.strip('\n').split('\t') for elem in test_data])
        test_name = test_data[:, 0]

    # 【训练数据预处理】
    with open(train_dir, 'r', encoding='utf-8') as f:
        # 【实体列表+标签列表：去除\n、\t】
        train_data = [elem.strip('\n') for elem in f.readlines()]
        train_data = np.array([elem.strip().split('\t') for elem in train_data])
        train_name = list(train_data[:, 0])
        train_labels = list(train_data[:, 1])

    # f = open('../data/tmail_kg_new.csv', 'w', encoding='utf-8')

    test_preds = []
    # 【count:推断出类型的实体名】
    # 【这个count定义了是干嘛的？】
    count = 0
    for i, elem in enumerate(test_name):

        if elem in kg_dict.keys():
            # 【kg_dict里的是实体名称的候选】
            candidate_list = list(kg_dict[elem])

            # 【应该是产生的一个噪声项】
            if 'c' in candidate_list:
                candidate_list.remove('c')

            # f.write(elem + ',' + ','.join(candidate_list) + '\n')

            if len(candidate_list) == 1:
                # 【如果在构造的图谱字典里，并且可以通过映射转换到类型】
                if candidate_list[0] in name_mapping.keys():
                    test_preds.append(name_mapping[candidate_list[0]])
                    count += 1
                else:
                    # 【否则推断不出来】
                    test_preds.append("NoneType")
            else:
                # 【不在预先构造的词典里，同样通过映射推断】
                mapped_candidate_list = [name_mapping[elem] for elem in candidate_list if elem in name_mapping.keys()]

                if len(mapped_candidate_list) != 1:
                    test_preds.append("NoneType")
                else:
                    test_preds.append(mapped_candidate_list[0])
                    count += 1
        else:
            test_preds.append("NoneType")
    f.close()
    # 【这些都做完了之后，通过正则表达式判断】
    # 【这么看来正则表达式的优先级是更高的？】
    rule_factory(test_name, test_preds)

    # 【将推断的内容写入结果】
    with open(result_dir, 'w', encoding='utf-8') as f:
        for i in range(len(test_name)):
            if test_name[i] in train_name:
                index_in_train = train_name.index(test_name[i])
                test_preds[i] = train_labels[index_in_train]
            f.write(test_name[i] + '\t' + test_preds[i] + '\n')


if __name__ == "__main__":
    rule_kg()
