# encoding:utf-8

"""
1 从raw 三元组中提取出出现次数大于10的关系
"""
total_sentences = 0
total_triples = 0
rel_stats = {}
ent_stats = {}
triple_stats = {}
with open('spacyOpenIE-ouput-v5.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = eval(line.strip())
        total_sentences += 1
        total_triples += len(line)
        for triple in line:
            if triple[1] in rel_stats.keys():
                rel_stats[triple[1]] += 1
            else:
                rel_stats[triple[1]] = 1

            if triple[0] in ent_stats.keys():
                ent_stats[triple[0]] += 1
            else:
                ent_stats[triple[0]] = 1

            if triple[2] in ent_stats.keys():
                ent_stats[triple[2]] += 1
            else:
                ent_stats[triple[2]] = 1

        if len(line) in triple_stats.keys():
            triple_stats[len(line)] += 1
        else:
            triple_stats[len(line)] = 1


print('关系数目：', len(rel_stats))

# 分别计算实体、关系
# rel_stats = ent_stats
# print(sorted(rel_stats.items(), key=lambda item: item[1]))
# print(sorted(ent_stats.items(), key=lambda item: item[1]))
sort_by_len = sorted(rel_stats.items(), key=lambda item: item[1])
rel_fre = {}
words = []
more_than_10 = 0
for item in sort_by_len:
    if item[1] in rel_fre.keys():
        rel_fre[item[1]] += 1
    else:
        rel_fre[item[1]] = 1

    if item[1] >= 10:
        words.append(item[0])

print("出现次数大于等于10", len(words))

with open('relations.txt', 'w', encoding='utf-8') as f:
    for i in words:
        f.write(i)
        f.write('\n')

import spacy as sp

"""
1 计算关系和关系之间的相似度，获取相似度大于阈值
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sp.require_gpu()

# text = "‘飞豹’作为中国最新一代轰炸机，完全由中国自主研发，是中国载弹能力最强的轰炸机，目前已部署在中国沿海基地，完全可以覆盖台湾，封锁其港口"
# word1 = "轰炸"
# word2 = "攻击"
nlp = sp.load("zh_core_web_lg")

words = []
with open('relations.txt', 'r', encoding='utf8') as f:
    for word in f.readlines():
        words.append(word)

del_words = []
res_file = open('relations_sim_valid.txt', 'w', encoding='utf8')
with open('relation_sim.txt', 'w', encoding='utf8') as f:
    for i in range(0, len(words)):
        for j in range(0, len(words)):
            if i != j and words[j] not in del_words:
                doc1 = nlp(words[i])
                doc2 = nlp(words[j])
                sim = doc1[0].similarity(doc2[0]).tolist()
                if sim > 0.5:
                    # 记录相似度达到阈值的关系
                    res_file.write(str((doc1[0], doc2[0], sim)))
                    res_file.write('\n')
                    del_words.append(words[j])
                # 记录所有的关系相似度
                f.write(str((doc1[0], doc2[0], sim)) + '\n')
res_file.close()


"""
2 获取这些关系对应的实体
"""

# 整理关系，rel2rels 用于记录关系：相似关系们
with open('relations_sim_valid.txt', 'r', encoding='utf8') as f:
    rel2rels = {}
    history_words = []
    for line in f.readlines():
        line = line.strip().split(',')
        first = line[0][1:]  # 第一个词
        second = line[1][1:]  # 第二个相似的词

        if first in history_words:
            # 不考虑关系相似的传递性，因此直接加入历史words不再考虑此word
            history_words.append(second)
            continue

        history_words.append(second)
        history_words.append(first)

        if first not in rel2rels.keys():
            rel2rels[first] = []
            rel2rels[first] = [first, second]
        else:
            rel2rels[first].append(second)

print("提取出的关系数目:", len(rel2rels))
print(list(rel2rels.keys()))

# 从三元组数据中，找到这些关系对应的三元组们，然后将头实体和尾实体分别整理
triples = []
with open('spacyOpenIE-ouput-military-simplify.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = eval(line)
        triples.extend(line)
# print(triples)

entities = {}
for rel in rel2rels.keys():
    each_rel_entities = {'head': [], 'tail': [], 'relation': [], 'domain': [], 'range': []}
    for tri in triples:
        if tri[1] in rel2rels[rel]:
            each_rel_entities['head'].append(tri[0])
            each_rel_entities['tail'].append(tri[2])
    each_rel_entities['relation'] = rel2rels[rel]
    entities[rel] = each_rel_entities
print(entities)

# vocab = {'国家地区': ['中国', '英国', '台湾', '美国', '美', '中', '叙利亚', '俄罗斯', '朝鲜', '马来西亚'],
#          '人物': ['奥巴马'],
#          '战斗机':, }

# 对实体进行domain和range的确定
# 计算每个rel的头实体们和尾实体们的相似度，保留和每个实体相似度最大的两个实体作为domain以及range。
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# sp.require_gpu()
nlp = sp.load("zh_core_web_lg")


def get_sim_ents(words):
    ret_words = {}
    del_words = []
    for i in range(0, len(words)):
        for j in range(0, len(words)):
            # 如果两个词不是同一个词；且不在已经有相似词的词内
            if words[i] != words[j] and words[j] not in del_words:
                doc1 = nlp(words[i])
                doc2 = nlp(words[j])
                sim = doc1[0].similarity(doc2[0])
                if sim > 0.55:
                    # 记录相似度达到阈值的实体
                    if words[i] not in ret_words.keys():
                        ret_words[words[i]] = [words[j]]
                    else:
                        ret_words[words[i]].append(words[j])
                    del_words.append(words[j])
                    if words not in del_words:
                        del_words.append(words[i])
    return ret_words


print(rel2rels['提升'])
for rel in entities.keys():
    # 处理头实体的type 从而获取domain
    # print(entities[rel]['head'])
    entities[rel]['domain'] = get_sim_ents(entities[rel]['head'])
    # print(entities[rel]['domain'])
    # 处理尾实体的type 从而获取range
    entities[rel]['range'] = get_sim_ents(entities[rel]['tail'])

# 将包含了{rel:{头、尾、domain、range}}的rel2rel写入文件
with open('rel2rels.txt', 'w', encoding='utf-8') as f:
    f.write(str(entities))

"""
3 最终构造出<domain, relaiton, range>三元组
"""
# 将得到的domain和range与关系进行组合，构造出<domain, relaiton, range>三元组
with open('rel2rels.txt', 'r', encoding='utf-8') as f:
    entities = eval(f.readlines()[0])

# 敏感词表
sensitive_words = ['习近平', '李克强', '胡锦涛', '温家宝', '江泽民']
wrong_words = ['KAI', '他', '她', '它', '1', '2']+sensitive_words
triples = []
for rel in entities.keys():
    each = entities[rel]
    domains = list(each['domain'].keys())
    ranges = list(each['range'].keys())
    for d in domains:
        for r in ranges:
            if d in wrong_words or r in wrong_words:
                continue
            triples.append([d, rel, r])


import pandas as pd
df = pd.DataFrame(triples)
print(df)
df.to_excel('军事关系抽取结果.xlsx')



