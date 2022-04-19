# coding=utf-8
# @Time : 2021/8/1 15:09

import json
# import sys

import Levenshtein  # 专门用来算编辑距离的
import urllib.parse
from tableExtract.table import *
import pymysql
import jieba


# reload(sys)
# sys.setdefaultencoding("utf-8")


class Candidate(object):
    # entity_path: 实体知识库路径
    # synonym_path: 同义词重定向知识库的路径
    # threshold: 筛选候选实体与 mention 的字符串相似度阈值
    def __init__(self, synonym_path, cursor, stopwords):
        self.synonym_path = synonym_path
        self.threshold = 0.7
        self.entitySynonym = []  # 列表每行存着一本字典 [{'label': label, 'synonym': [synonym list]}]
        self.synonymDict = {}
        self.cursor = cursor
        self.sql0 = "select * from label where label = %s limit 1"
        self.sql1 = "select * from label where label like %s"
        self.sql2 = "select `infobox`, `abstract` from baike where file_id = %s"
        self.sql4 = "select * from label where label = %s"
        self.stopwords = stopwords
        synonymFile = open(self.synonym_path, 'r')

        count = 0
        L = ['1'] * 5
        for line in synonymFile.readlines():
            count += 1
            count %= 4
            line = line.strip().strip('.')
            line = line.split('/')[-1].strip(' ').strip('>')
            line = urllib.parse.unquote(line, encoding="utf8")
            L[count] = line
            # 第一行是同义词  # 第三行是实际会跳转到的页面
            if count == 0:
                if self.synonymDict.__contains__(L[1]):
                    print(L[1])
                else:
                    self.synonymDict[L[1]] = L[3]
        print('重定向处理完毕')
        synonymFile.close()

    # calculate String Similarity by edit distance
    def stringSimilarity(self, s1, s2):
        edit_distance = Levenshtein.distance(s1, s2)
        len_s1 = len(s1)
        len_s2 = len(s2)
        if len_s1 > len_s2:
            max = len_s1
        else:
            max = len_s2
        # 编辑距离越大 相似值是越低的
        stringSimilarity = 1.0 - float(edit_distance) / max
        return stringSimilarity

    # Table表格类
    def generateCandidate(self, table):
        # 二维数组 entity_candidate[j][k]
        # 每一格存的是一本字典{'mention': m, 'candidates': [c1, c2, c3...]}
        cursor = self.cursor
        entity_candidate = []
        if table.unfoldDirection == 'COL':
            table.flip();

        nRow = table.rowNumber
        nCol = table.colNumber

        # 为传入的表格中的每个单元格中的 mention 生成候选实体
        # i: row number; j: column number
        for i in range(nRow):
            row = []  # 当前行的所有字典
            for j in range(nCol):
                dict = {}
                candidates = []
                file_ids = []
                cell = table.cell[i][j]

                temp_list = []
                # 表头不做候选实体生成
                if i == 0:
                    dict['header'] = cell.content
                    row.append(dict)
                    continue;

                # 如果为空 就直接不候选实体了
                if cell.content.isspace() or cell.content == '':
                    dict['mention'] = cell.content
                    dict['candidates'] = [[], []]
                    row.append(dict)
                    continue

                temp_content = cell.content
                if cell.multi:
                    temp_content = cell.key_content.split('_;_')[0]
                    if temp_content.isspace() or temp_content == '':
                        temp_content = cell.key_content.split('_;_')[1]
                originContent = temp_content
                temp_content = temp_content.strip('《》')
                temp_content = temp_content.replace(" ", "")
                pattern = re.compile(r'^((\d)+)(\.)?(-[01]?\d+)?(-?\d{2})?年?(\d{1,2}月?)?(\d{2}日?)?')
                # 数字(含浮点数) 年月就不链接了
                if re.match(pattern, temp_content):
                    dict['mention'] = temp_content
                    dict['candidates'] = [[], []]
                    row.append(dict)
                    continue

                # 看是否在同义词列表中
                for key in self.synonymDict:
                    if temp_content in key:
                        stringSimilarity = self.stringSimilarity(temp_content, key)
                        if stringSimilarity < self.threshold:
                            continue
                        cursor.execute(self.sql0, self.synonymDict[key])  # 精确查找且limit为1
                        result = cursor.fetchone()
                        if result is None:  # 没搜到
                            continue
                        file_id = result[0]
                        entity = result[1]
                        if file_id in file_ids:  # 重复了
                            continue
                        file_ids.append(file_id)
                        candidates.append(entity)

                # 查找 实体(消歧义)
                if temp_content.isspace() or temp_content == '':
                    dict['mention'] = temp_content
                    dict['candidates'] = [[], []]
                    row.append(dict)
                    continue
                content = temp_content + "（%"
                cursor.execute(self.sql1, content)
                results = cursor.fetchall()
                for result in results:
                    # 完整的实体，包括消岐义内容 real_entity(disambiguation)
                    file_id = result[0]
                    entity = result[1]
                    if entity in candidates:
                        continue
                    real_entity = entity
                    if real_entity[-1] == '）':  # 防止实体里本身就有带括号而不是那些消歧的括号
                        split = entity.split('（')
                        real_entity = split[0]  # 真实的实体，去除了消岐义内容 real_entity
                    stringSimilarity = self.stringSimilarity(temp_content, real_entity)
                    if stringSimilarity >= self.threshold:
                        candidates.append(entity)
                        file_ids.append(file_id)

                # 最后在按mention本身来找一下试试
                cursor.execute(self.sql0, temp_content)
                results = cursor.fetchall()
                for result in results:
                    file_id = result[0]
                    entity = result[1]
                    if entity in candidates:
                        continue
                    real_entity = entity
                    if real_entity[-1] == '）':  # 防止实体里本身就有带括号而不是那些消歧的括号
                        split = entity.split('（')
                        real_entity = split[0]  # 真实的实体，去除了消岐义内容 real_entity
                    stringSimilarity = self.stringSimilarity(temp_content, real_entity)
                    if stringSimilarity >= self.threshold:
                        candidates.append(entity)
                        file_ids.append(file_id)

                dict['mention'] = originContent
                candidates = list(set(candidates))  # 去除重复元素

                # 如果候选实体过多了, 则会影响消歧的效率 就干脆精确查找一次
                if len(candidates) > 50:
                    candidates = []
                    file_ids = []
                    cursor.execute(self.sql4, content)
                    results = cursor.fetchall()
                    for result in results:
                        # 完整的实体，包括消岐义内容 real_entity[disambiguation]
                        file_id = result[0]
                        entity = result[1]
                        if entity in candidates:
                            continue
                        real_entity = entity
                        if real_entity[-1] == '）':  # 防止实体里本身就有带括号而不是那些消歧的括号
                            split = entity.split('（')
                            real_entity = split[0]  # 真实的实体，去除了消岐义内容 real_entity
                        stringSimilarity = self.stringSimilarity(temp_content, real_entity)
                        if stringSimilarity >= self.threshold:
                            candidates.append(entity)
                            file_ids.append(file_id)

                temp_list.append(candidates)
                temp_list.append(file_ids)
                dict['candidates'] = temp_list

                row.append(dict)

            entity_candidate.append(row)
        print('当前表格候选实体生成完成')
        print(entity_candidate)
        return entity_candidate

    # L是候选实体
    def generateContext(self, _table, L):
        entity_context = {}
        cursor = self.cursor
        flag = 0
        for row in L:
            if (flag == 0):  # 跳过属性那一行
                flag = 1
                continue
            for col in row:
                if len(col['candidates']) != 0:
                    candidate_entities = col['candidates'][0]
                    file_ids = col['candidates'][1]
                    for i in range(len(file_ids)):
                        cursor.execute(self.sql2, file_ids[i])
                        result = cursor.fetchone()
                        if result is None:
                            info_box = ''
                            abstract = ''
                        else:
                            info_box = result[0]
                            abstract = result[1]
                        abstract = self.decomposition(abstract, candidate_entities[i])
                        entity_context[candidate_entities[i]] = self.info_boxToList(info_box)
                        entity_context[candidate_entities[i]].extend(abstract)
        print('候选实体上下文完毕')
        return entity_context

    def decomposition(self, sentence, entity):
        entity = entity.split('[')[0]  # 去除 [防歧义]部分
        sentence = sentence.replace(entity, '', 1)  # 把实体在摘要中去除一次 如果还有剩就当作上下文
        sentence_depart = jieba.lcut(sentence.strip())
        output = []  # 输出结果
        # 去停用词
        for word in sentence_depart:
            if word not in self.stopwords:
                if word != '\t':
                    output.append(word)
        return output

    def info_boxToList(self, info_box):
        out_put_list = []
        _list = info_box.split('_;_')
        for i in range(0, len(_list) - 1):
            pair = _list[i]
            split = pair.split('_:_')
            out_put_list.append(split[1])
        return out_put_list
