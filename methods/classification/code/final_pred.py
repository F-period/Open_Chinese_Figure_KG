from pathlib import Path
import torch
from torch import nn
import random
import re
import numpy as np

# 【模型+正则的结合方式】

# 【外部数据源精修的一个环节】
def train_refine(final_results_dir, names_list, preds_list):
    train_path = './rules/entity_type.txt'
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data_raw = f.readlines()
        train_data = np.array([elem.strip('\n').split('\t') for elem in train_data_raw])
        train_name = list(train_data[:, 0])
        train_label = list(train_data[:, 1])

    with open(final_results_dir, 'w', encoding='utf-8') as ff:

        # 【对预测结果和训练集不一致的部分进行精修】
        for index, name in enumerate(names_list):
            if name in train_name:
                dup_label = train_label[train_name.index(name)]
                if preds_list[index] != dup_label:
                    # print(f"name: {name}, preds: {preds_list[index]}, label: {dup_label}")
                    preds_list[index] = dup_label

            # 【写出最终结果】
            ff.write(name + '\t' + preds_list[index] + '\n')


def other_src_refine(g_disease, g_symptom, target, final_results_dir):
    # 【主要是依据疾病与症状列表】
    disease = set()
    symptom = set()

    # 【同样将这两者构成列表，并进行交集操作】
    with open(g_disease, 'r', encoding='utf-8') as input:
        for line in input:
            k = line.rstrip('\n')
            disease.add(k)

    with open(g_symptom, 'r', encoding='utf-8') as input:
        for line in input:
            k = line.rstrip('\n')
            symptom.add(k)

    overlap = set.intersection(set(disease), set(symptom))

    if len(overlap) != 0:
        print(f"find {len(overlap)} overlap items between disease and symptom\n")

    final_results_dict = dict()
    names_list = []

    count = 0
    overlap_count = 0
    all_count = 0

    # 【target:上面生成的final_result】
    with open(target, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            name = line[0]
            preds = line[1]

            # 【对位于这些列表中的实体名进行判断】
            if name in overlap:
                # if preds == '症状':
                #     print(name, preds)
                overlap_count += 1
                # src_preds = None
                src_preds = '疾病'
            elif name in disease:
                src_preds = '疾病'
            elif name in symptom:
                src_preds = '症状'
            else:
                src_preds = None

            if src_preds:
                all_count += 1
                if src_preds != preds:
                    count += 1
                    # print(f'name:{name}, pred:{preds}, src pred:{src_preds}')
                final_results_dict[name] = src_preds
            else:
                final_results_dict[name] = preds

            names_list.append(name)

    # print(f"changed:{count}, overlap:{overlap_count}, all:{all_count}\n")
    # 【检查完这两个列表之后去根据训练数据进一步处理】
    train_refine(final_results_dir, names_list, [final_results_dict[name] for name in names_list])


def post_rule_factory(names, preds):
    for i in range(len(names)):

        cur_name = names[i]

        if re.findall(
                '《|》|·|盲点|假尾孢|金龟|光鞘|木霉|栉大蚊|分娩|筛查系统$|'
                '褐蛉$|鸻$|酵母$|玉米油$|蜾蠃$|元方$|长喙壳$|马勃$|信天翁$|'
                '能力测试$|蘑$|耳属$|青霉$|.曲霉$|腐霉$|[氧氦]-[0-9]{0,3}|'
                '.同位素$|HZC$|护理$|迟萌$|毒品$|一氧..$|奶粉$',
                cur_name) or cur_name in ['手', '口', '足', '头', '耳', '液', '眼', '胸']:
            preds[i] = 'NoneType'

        # if re.match('.*(病|综合征|炎|瘤|症|癌|癣|囊肿|中毒|青光眼)$', cur_name):
        #     if preds[i] != '疾病':
        #         print(cur_name, preds[i])
        #     preds[i] = '疾病'

        # if re.findall('鱼眼$|尿糖$|痉挛$|衰竭$|肥大$|脱落$|谊妄$|收缩$', cur_name):
        #     preds[i] = '症状'

        if re.findall(
                '鱼眼$|痉挛$|宫颈肥大$|撕脱伤$|直肠重复畸形|神经功能障碍$|'
                '宫腔粘连$|结节性硬化$|样皮疹$|性肥胖$|增殖体肥大$|冠状动脉供血不足|'
                '神经性头痛|血瘀体质$|营养代谢缺乏$|强迫型人格障碍$|晕动病$',
                cur_name):
            preds[i] = '症状'

        if re.match('.*(小儿脾大$|骨质增生$|丛集性头痛$|高弓足$|记忆障碍$|急性肾功能不全$|肛周脓肿$|支气管扩张$)', cur_name):
            preds[i] = '疾病'

        if re.findall('球蛋白', cur_name) and re.findall('[病症]$', cur_name):
            preds[i] = '疾病'

        if re.findall('柞蚕蛾$|菠萝蛋白酶$|凝血活酶$|.上腺素$|尿.酶$|.*诊断红细胞$', cur_name):
            preds[i] = '药物'

        if re.match('噬菌体$|.状病毒$|.*流感病毒$', cur_name):
            preds[i] = '病毒'
        if re.match('.*(诊断用噬菌体)$', cur_name):
            preds[i] = '药物'

        if re.match('链霉菌$|沙门氏菌$|.*血弧菌$', cur_name):
            preds[i] = '细菌'
        if re.findall('[杆球]菌$', cur_name) and re.findall('[血肠歧乳葡萄金芽]', cur_name):
            preds[i] = '细菌'

        if re.match('总补体$|胆红素$|血清钠$|胍基化合物$|甘氨酸$|脑脊液钠$|乳糜微粒$|胰多肽$',
                cur_name):
            preds[i] = '检查科目'

        if re.findall('球蛋白$|球蛋白[A-Z]$|球蛋白.*类$', cur_name) and re.findall('.[血清抗体甲乙丙试验反应A-Z]', cur_name):
            preds[i] = '检查科目'

        if re.findall('时间$|测定$|结合反应$|[碘蛋白]试验$|渗滤法$|抗凝因子$|转移酶$|血糖素$|..冬氨酸$|组氨酸$', cur_name):
            preds[i] = '检查科目'

        if re.findall('^尿.*酮$|^[尿粪][糖锰钙钠素酰胺酮]$|^尿.*肾上腺素$|^尿.*酰胺$|^尿维生素|免疫复合物$', cur_name):
            preds[i] = '检查科目'

        if re.findall('^粪.*结晶$|^血.*紧张素|^促.*激素$|^人绒毛.*激素$', cur_name):
            preds[i] = '检查科目'

        if re.findall('[一二三四]酮$', cur_name):
            preds[i] = '检查科目'

        if re.findall('^胰.*素$', cur_name):
            preds[i] = '检查科目'

# 【根据规则进行矫正】
def post_rule_refine(target, final_results_dir):
    preds_list = []
    names_list = []

    # 【将结果数据转换为字典】
    with open(target, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            name = line[0]
            pred = line[1]
            preds_list.append(pred)
            names_list.append(name)

    # 【根据正则表达式的预测结果】
    post_rule_factory(names_list, preds_list)
    # 【校正】
    train_refine(final_results_dir, names_list, preds_list)

# 【预测】
def final_pred(pre_rule, results_model, g_disease, g_symptom, post_rule, train_file, output_path):
    train_map = {}
    prerule_map = {}
    model_map = {}
    gd = set()
    gs = set()
    postrule_map = {}

    pred = {}
    pred_source = {}

    # 【将训练数据转换为字典形式】
    with open(train_file, 'r', encoding='utf-8') as input:
        for line in input:
            k, v, *_ = line.rstrip('\n').split('\t')
            train_map[k] = v

    with open(pre_rule, 'r', encoding='utf-8') as input:
        for line in input:
            k, v, *_ = line.rstrip('\n').split('\t')
            prerule_map[k] = v

    # 【将刚才模型投票的结果转换为字典形式】
    with open(results_model, 'r', encoding='utf-8') as input:
        for line in input:
            k, v, *_ = line.rstrip('\n').split('\t')
            model_map[k] = v

    # 【一个疾病的集合】
    with open(g_disease, 'r', encoding='utf-8') as input:
        for line in input:
            k = line.rstrip('\n')
            gd.add(k)

    # 【一个症状的集合】
    with open(g_symptom, 'r', encoding='utf-8') as input:
        for line in input:
            k = line.rstrip('\n')
            gs.add(k)

    # 【将两个集合的交集单独抽取出来】
    both = gd.intersection(gs)
    gd = gd.difference(both)
    gs = gs.difference(both)

    for k, v in prerule_map.items():
        # 【如果规则不能处理的再通过模型预测】
        if v == 'NoneType':
            pred[k] = model_map[k]
            pred_source[k] = 'model'
        else:
            pred[k] = v
            pred_source[k] = 'pre'


        # data based rules should be put in the last step
        # 【最后一步根据已有数据源界定：最可靠】
        if pred[k] == '疾病' and k in gs:
            pred[k] = '症状'
            # if pred_source[k] == 'post':
            #     print(k)
            pred_source[k] = 'github_symptom'

        if pred[k] == '症状' and k in gd:
            pred[k] = '疾病'
            # if pred_source[k] == 'post':
            #     print(k)
            pred_source[k] = 'github_disease'

        if k in train_map:
            pred[k] = train_map[k]
            pred_source[k] = 'train'

    source_stat_dict = {}
    for k, v in pred_source.items():
        if v in source_stat_dict:
            source_stat_dict[v] += 1
        else:
            source_stat_dict[v] = 1

    source_stat = sorted(source_stat_dict.items(), key=lambda x: x[1], reverse=True)
    print(source_stat)

    # 写入预测结果
    with open(output_path, 'w', newline='', encoding='utf-8') as output:
        for k, v in pred.items():
            output.write(f"{k}\t{v}\n")

# 【最终的矫正】
def final_pred_post(g_disease, g_symptom, final_results, final_results_other_src, final_results_post):
    # 【根据其它数据源的数据进行校正】
    other_src_refine(g_disease, g_symptom, final_results, final_results_other_src)
    # 【根据规则对结果进行校正】
    post_rule_refine(final_results_other_src, final_results_post)


if __name__ == '__main__':
    # 【进行预测-->详见上方的函数定义】
    final_pred('./rules/pre_rules.txt',
               './pred_data/results_test_model_vote.txt',
               './rules/github_disease.txt',
               './rules/github_symptom.txt',
               './rules/post_rules.txt',
               './rules/entity_type.txt',
               './rules/pre_rule+model+post_result.txt')

    # 【对预测的结果进行最终的矫正】
    final_pred_post('./rules/github_disease.txt',
                    './rules/github_symptom.txt',
                    './rules/pre_rule+model+post_result.txt',
                    './rules/pre_rule+model+post+other_src_result.txt',
                    './result.txt')
