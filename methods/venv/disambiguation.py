# coding=utf-8
# @Time : 2021/8/4 11:06


import Levenshtein
import time  # 从目前代码上看 并没有派上用场
import copy
import networkx as nx  # 将表格转换为图
from networkx.drawing.nx_agraph import write_dot
import numpy as np
import os
import json
from tableExtract.table import *

class EntityDisambiguationGraph(object):
    # table: 第i张表格，Table 类型对象
    # table_number: 表格的编号，即为 i
    # candidates: 当前表格中 mentions 的候选实体
    # graph_path: EDG 图片输出路径
    # EDG_out_path: EDG 输出路径
    # disambiguation_result_path: 消岐结果输出路径
    # mention_quantity: 当前表格中的 mention 数量
    # rowNumber: 当前表格的行数
    # colNumber: 当前表格的列数
    # EDG: 当前表格及其候选实体生成的完成的 EDG
    # miniEDG: 省略了 entity-entity Edge 的 EDG，为了更快速地画图 因为实体之间都是满边
    # mention_node_begin: mention node 编号的开始
    # mention_node_end: mention node 编号的结束
    # entity_node_begin: entity node 编号的开始
    # entity_node_end: entity node 编号的结束
    # node_quantity: 所有节点的总数
    # alpha1, beta1, alpha2, beta2: 计算语义相似度时的参数
    # A: 概率转移列表
    # r: 消岐结果概率列表
    def __init__(self, table_number, table : Table, candidates, entity_context):
        self.table_number = table_number
        self.table = table
        self.rowNumber = table.rowNumber
        self.colNumber = table.colNumber
        self.mention_quantity = (self.rowNumber - 1) * self.colNumber
        self.candidates = candidates
        self.entity_context = entity_context
        self.EDG = nx.Graph(number=table_number)  # 创建简单图
        self.miniEDG = nx.Graph(number=table_number)
        self.mention_node_begin = 0
        self.mention_node_end = self.mention_quantity - 1
        self.entity_node_begin = self.mention_quantity
        self.entity_node_end = 0
        self.node_quantity = 0
        self.A = []
        self.r = []
        self.damping_factor = 0.5
        self.iterations = 500
        self.delta = 0.0001
        self.alpha1 = 0.5
        self.beta1 = 0.5
        self.alpha2 = 0.5
        self.beta2 = 0.5
        self.bonus = 0.0

    # 获取当前表格中一个 mention 的上下文，该 mention 位于第r行第c列，r与c都从0开始
    def getMentionContext(self, r, c):
        table = self.table
        mention_context = table.getMentionContext(r, c)
        return mention_context

    # 获取一个 entity e 的上下文，来自 abstract 和 infobox_property
    # e: entity 字符串
    def getEntityContext(self, e):
        dict1 = self.entity_context
        if dict1.__contains__(e):
            return dict1[e]
        else:
            list = []
            return list

    # 获取实体字符串的消岐义内容
    # entity: entity 字符串
    def getEntityDisambiguation(self, entity):
        disambiguation = ''
        # 完整的实体 entity，包括消岐义内容 real_entity[disambiguation]
        if entity[-1] == '）':
            split = entity.split('（')
        else:
            return disambiguation
        if len(split) == 2:
            disambiguation = split[1]
            disambiguation = disambiguation[:-1]

        return disambiguation

    # Building Entity Disambiguation Graph
    # mNode: mention node
    # eNode: entity node
    # meEdge: mention-entity edge
    # eeEdge: entity-entity edge
    # node probability: mention node probability 为初始权重值。entity node probability 在 iterative_probability_propagation() 中计算
    # edge probability: 边两端节点间的语义相似度。有2种边，mention-entity edge 和 entity-entity edge
    def build_entity_disambiguation_graph(self):
        EDG = self.EDG
        table = self.table
        candidates = self.candidates
        nRow = self.rowNumber
        nCol = self.colNumber
        i = self.table_number
        mention_quantity = self.mention_quantity
        mention_node_initial_importance = float(1) / mention_quantity

        # 编号范围 [0, mention_quantity - 1] [mention_quantity, entity_node_end]
        # 节点类型     mention node                      entity node
        mention_id = 0
        entity_id = mention_quantity

        # 逐行逐列遍历给定表格中的每个单元格
        for r in range(nRow):
            if r == 0:  # 表头不作为 EDG 中的节点
                continue

            for c in range(nCol):
                mention = table.cell[r][c].content  # unicode
                entity_candidates = candidates[r][c]['candidates'][0]  # unicode
                candidate_index = 0

                flag_NIL = False
                if len(entity_candidates) == 0:  # 当前提及没有候选实体
                    flag_NIL = True

                # 在 EDG 中添加 mention node
                # ranking: [(entity node index i, the probability for the node i to be the referent entity of the mention)] 候选实体根据概率逆序排列的列表
                EDG.add_node(mention_id, type='mNode', mention=mention, NIL=flag_NIL, table=i, row=r, column=c,
                             ranking=[], probability=float(mention_node_initial_importance), context=[])
                # EDG.nodes[mention_id]['label'] = 'mention: ' + EDG.nodes[mention_id]['mention']
                EDG.nodes[mention_id]['context'] = self.getMentionContext(r, c)

                if not flag_NIL:
                    # 在 EDG 中添加 entity node 且 提及要与他自己的每个候选实体都连边
                    # candidate: 候选实体字符串
                    # mNode_index: entity node 相邻的唯一一个 mention node 的编号
                    # disambiguation: 实体名称中的消岐义部分 entity [disambiguation]
                    for candidate in entity_candidates:
                        candidate_index += 1
                        EDG.add_node(entity_id, type='eNode', candidate=candidate, index=candidate_index,
                                     mNode_index=mention_id, probability=float(0), context=[], disambiguation='')
                        # EDG.nodes[entity_id]['label'] = 'candidate' + str(EDG.nodes[entity_id]['index']) + ': ' + EDG.nodes[entity_id]['candidate']
                        EDG.nodes[entity_id]['context'] = self.getEntityContext(candidate)
                        EDG.nodes[entity_id]['disambiguation'] = self.getEntityDisambiguation(candidate)

                        # 在 EDG 中添加 mention-entity edge
                        EDG.add_edge(mention_id, entity_id, type='meEdge', probability=float(0))

                        entity_id += 1
                mention_id += 1
        self.entity_node_end = entity_id - 1
        self.node_quantity = entity_id
        self.miniEDG = copy.deepcopy(EDG)

        # 在 EDG 中添加 entity-entity edge
        for p in range(self.entity_node_begin, self.entity_node_end + 1):
            for q in range(self.entity_node_begin, self.entity_node_end + 1):
                if p < q:
                    EDG.add_edge(p, q, type='eeEdge', probability=float(0))
                    EDG[p][q]['label'] = str(EDG[p][q]['probability'])

        self.EDG = EDG

    def stringSimilarity(self, s1, s2):
        # s1 = s1.decode('utf8')
        # s2 = s2.decode('utf8')
        edit_distance = Levenshtein.distance(s1, s2)
        len_s1 = len(s1)
        len_s2 = len(s2)

        if len_s1 > len_s2:
            max = len_s1
        else:
            max = len_s2

        stringSimilarity = 1.0 - float(edit_distance) / max
        return stringSimilarity

    # 计算 mention 和 entity 之间的字符串相似度特征 (String Similarity Feature)
    # m: mention node index
    # e: entity node index
    def strSim(self, m, e):
        mention = self.EDG.nodes[m]['mention']  # unicode
        entity = self.EDG.nodes[e]['candidate']  # unicode

        split = entity.split('(')
        real_entity = split[0]  # 真实的实体 (unicode)，去除了消岐义内容 real_entity

        stringSimilarity = self.stringSimilarity(mention, real_entity)
        return stringSimilarity

    # x y都是存着字符串的列表
    def jaccard_similarity(self, x, y):
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        jaccard_similarity = intersection_cardinality / float(union_cardinality)
        return jaccard_similarity

    # 计算 mention 和 entity 之间的上下文相似度特征 (Mention-Entity Context Similarity Feature)
    # m: mention node infdex
    # e: entity node index
    def contSim_me(self, m, e):
        mention_context = self.EDG.nodes[m]['context']
        entity_context = self.EDG.nodes[e]['context']

        if len(entity_context) == 0:
            context_similarity_me = 0.0
            return context_similarity_me

        context_similarity_me = self.jaccard_similarity(mention_context, entity_context)
        return context_similarity_me

    # 计算 mention 和 entity 之间的语义相似度 (Mention-Entity Semantic Relatedness)
    # m: mention node index
    # e: entity node index
    def SR_me(self, m, e):
        alpha1 = self.alpha1
        beta1 = self.beta1
        sr_me = 0.99 * (alpha1 * self.strSim(m, e) + beta1 * self.contSim_me(m, e)) + 0.01
        return sr_me

    # 计算 mention node 和其所有相邻 entity node 之间的语义相似度之和
    # m: mention node index
    def SR_me_star(self, m):
        sr_me_star = 0.0

        if self.EDG.nodes[m]['NIL'] == False:  # 先确保该提及有候选实体
            for e in self.EDG.neighbors(m):
                sr_me_star += self.EDG[m][e]['probability']

        return sr_me_star

    # 计算 2 entities 之间的三元组关系特征 (Triple Relation Feature)
    # ???: e1 和 e2 存在于同一个 RDF 中是否需要存在于不同部分
    # e1: entity1 node index
    # e2: entity2 node index
    def isRDF(self, e1, e2):
        dict1 = self.entity_context
        is_rdf = 0
        e1 = self.EDG.nodes[e1]['candidate']
        e2 = self.EDG.nodes[e2]['candidate']

        if dict1.__contains__(e1):
            for i in dict1[e1]:
                if e2 == i:
                    is_rdf = 1
                    break
        if is_rdf == 0 and dict1.__contains__(e2):
            for i in dict1[e2]:
                if e1 == i:
                    is_rdf = 1
                    break
        return is_rdf

    # 计算 2 entities 之间的上下文相似度特征 (Entity-Entity Context Similarity Feature)
    # e1: entity1 node index
    # e2: entity2 node index
    def contSim_ee(self, e1, e2):
        entity1_context = self.EDG.nodes[e1]['context']
        entity2_context = self.EDG.nodes[e2]['context']

        if len(entity1_context) == 0 or len(entity2_context) == 0:
            context_similarity_ee = 0.0
            return context_similarity_ee

        context_similarity_ee = self.jaccard_similarity(entity1_context, entity2_context)
        return context_similarity_ee

    # 计算 2 entities 之间的语义相似度 (Entity-Entity Semantic Relatedness)
    # e1: entity1 node index
    # e2: entity2 node index
    def SR_ee(self, e1, e2):
        alpha2 = self.alpha2
        beta2 = self.alpha2
        sr_ee = 0.99 * (alpha2 * self.isRDF(e1, e2) + beta2 * self.contSim_ee(e1, e2)) + 0.01
        return sr_ee

    # 计算 entity node 和其相邻的唯一一个 mention node 之间的语义相似度
    # e: entity node index
    def SR_em(self, e):
        m = self.EDG.nodes[e]['mNode_index']
        sr_em = self.EDG[m][e]['probability']
        return sr_em

    # 计算 entity node 和其所有相邻 entity node 之间的语义相似度之和
    # e: entity node index
    def SR_ee_star(self, e):
        sr_ee_star = 0.0

        m = self.EDG.nodes[e]['mNode_index']
        sr_me = self.EDG[m][e]['probability']

        entities = self.EDG.neighbors(e)

        for ee in entities:  # 刚看到这里还让我有些许困惑
            sr_ee_star += self.EDG[e][ee]['probability']  # 怎么不是调用SR_ee
            # 后面发现调用SR_ee的地方比star早 所以已经存好了

        sr_ee_star -= sr_me

        return sr_ee_star

    # Computing EL Impact Factors
    def compute_el_impact_factors(self):
        EDG = self.EDG

        # compute semantic relatedness between mentions and entities
        # k: mention node 编号
        # i: entity node 编号
        for k in range(self.mention_node_begin, self.mention_node_end + 1):
            if EDG.nodes[k]['NIL'] == False:  # 同样要先确保当前提及有候选实体
                for i in EDG.neighbors(k):
                    EDG[k][i]['probability'] = self.SR_me(k, i)

        # compute semantic relatedness between entities  暴力双循环
        # p: entity1 node 编号
        # q: entity2 node 编号
        for p in range(self.entity_node_begin, self.entity_node_end + 1):
            for q in range(self.entity_node_begin, self.entity_node_end + 1):
                if p < q:
                    EDG[p][q]['probability'] = self.SR_ee(p, q)

        self.EDG = EDG

    # Iterative Probability Propagation
    # 计算 entity node probability (该 entity 成为 mention 的对应实体的概率)
    def iterative_probability_propagation(self):
        EDG = self.EDG
        n = self.node_quantity
        damping_factor = self.damping_factor
        iterations = self.iterations
        delta = self.delta
        A = [[0.0 for col in range(n)] for row in range(n)]
        E = [[1.0 for col in range(n)] for row in range(n)]
        r = [0.0 for i in range(n)]
        flag_convergence = False

        # compute A[i][j]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                i_type = EDG.nodes[i]['type']
                j_type = EDG.nodes[j]['type']

                if i_type == 'mNode' and j_type == 'mNode':
                    continue

                elif i_type == 'mNode' and j_type == 'eNode':
                    if EDG.nodes[i]['NIL'] == False:
                        if EDG.has_edge(i, j) == True:
                            A[i][j] = EDG[i][j]['probability'] / self.SR_me_star(i)

                elif i_type == 'eNode' and j_type == 'mNode':
                    if EDG.nodes[j]['NIL'] == False:
                        if EDG.has_edge(i, j) == True:
                            A[i][j] = A[j][i]  # 这个应该是因为i一定是从提及开始 所有轮到这里时 右边一定有值

                elif i_type == 'eNode' and j_type == 'eNode':
                    A[i][j] = (1.0 - self.SR_em(i)) * EDG[i][j]['probability'] / self.SR_ee_star(i)

        self.A = A
        # initialize r(i)
        # epoch 0
        for i in range(n):
            if i >= self.mention_quantity:  # 实体节点初始值为0
                break
            r[i] = 1.0 / self.mention_quantity

        matrix_r = np.matrix(r).T
        matrix_A = np.matrix(A)
        matrix_E = np.matrix(E)

        # update r(i)
        for epoch in range(1, iterations + 1):
            matrix_r_next = \
                ((1.0 - damping_factor) * (matrix_E / n) + damping_factor * matrix_A) * matrix_r

            r_list = matrix_r.tolist()
            r_next_list = matrix_r_next.tolist()
            max_difference = 0.0

            for i in range(n):
                if EDG.nodes[i]['type'] == 'eNode':
                    difference = abs(r_list[i][0] - r_next_list[i][0])

                    if difference > max_difference:
                        max_difference = difference

            if max_difference <= delta:
                matrix_r = matrix_r_next
                flag_convergence = True
                break

            matrix_r = matrix_r_next

        r_list = matrix_r.tolist()

        for i in range(n):
            r[i] = r_list[i][0]

        self.r = r
        self.EDG = EDG

    # 给 mention 的候选实体排名
    def rank_candidates(self):
        EDG = self.EDG
        r = self.r

        for i in range(self.mention_node_begin, self.mention_node_end + 1):
            if EDG.nodes[i]['NIL'] == True:
                continue

            mention = EDG.nodes[i]['mention']
            mention_context = EDG.nodes[i]['context']
            neighbors = list(EDG.neighbors(i))  # 源代码中把这个赋给candidates不知为何会出现问题
            ranking = []
            max = 0.0  # 等下作为概率归一化的分母 
            self.bonus = 0.0  # bonus: 实体节点上概率奖励增量，为其所在的候选实体集合中所有实体节点概率的平均值

            counter = 0
            for e in neighbors:
                self.bonus += r[e]
                counter += 1  # 不知道怎么得到邻居数 所以干脆自己统计

            self.bonus /= counter

            for e in neighbors:
                entity = EDG.nodes[e]['candidate']
                disambiguation = EDG.nodes[e]['disambiguation']
                probability = r[e]

                if entity == mention:
                    probability += self.bonus  # 候选实体与 mention 完全相同，奖励该候选实体
                for c in mention_context:
                    if c in disambiguation:
                        probability += 2 * self.bonus  # mention 的上下文中元素出现在候选实体的消岐义内容中，奖励该候选实体

                r[e] = probability
                tuple = (e, probability)  # (实体节点编号，实体成为 mention 的对应实体的概率)
                ranking.append(tuple)

                # 为实体结果的概率归一化做准备
                if r[e] > max:
                    max = r[e]

            newranking = []
            for t in ranking:
                t = list(t)
                e = t[0]
                p = t[1] / max
                tuple = (e, p)
                newranking.append(tuple)
            newranking.sort(key=lambda x: x[1], reverse=True)  # newranking 根据概率逆序排序，概率值越大下标越小
            EDG.nodes[i]['ranking'] = newranking
        # 计算 eNode 上的概率 并 打上标签
        for p in range(self.entity_node_begin, self.entity_node_end + 1):
            EDG.nodes[p]['probability'] = r[p]
            self.miniEDG.nodes[p]['label'] = 'candidate' + str(EDG.nodes[p]['index']) + ': ' + EDG.nodes[p][
                'candidate'] + '\n' + str(EDG.nodes[p]['probability'])

        self.EDG = EDG

    # 挑选出 mention 的候选实体中概率最高的一个 entity
    # 将消岐后的结果文件存储于 disambiguation_output_path
    def pick_entity(self):

        EDG = self.EDG
        table = self.table
        nRow = self.rowNumber
        nCol = self.colNumber
        i = self.mention_node_begin
        t = []

        for m in range(nRow):
            row = []
            for n in range(nCol):
                table.cell[m][n].href = {} # 因为百度里本身固有的连接也是有错误的 所以这个没有价值
                if m == 0:
                    continue
                else:
                    mention = EDG.nodes[i]['mention']

                    if EDG.nodes[i]['NIL'] == True:
                        entity = 'null'
                    else:
                        eNode_index = EDG.nodes[i]['ranking'][0][0]
                        entity = EDG.nodes[eNode_index]['candidate']
                        table.cell[m][n].href[mention] = entity

                    i += 1


class Disambiguation(object):
    def __init__(self):
        pass

    def disambiguation(self, table_number, table, candidates, entity_context):
        t = []
        if table.tableType == '获奖关系表':
            return t
        EDG_master = EntityDisambiguationGraph(table_number, table, candidates, entity_context)
        if EDG_master.mention_quantity == 0:
            pass
        else:
            time1 = time.time()
            EDG_master.build_entity_disambiguation_graph()
            EDG_master.compute_el_impact_factors()
            EDG_master.iterative_probability_propagation()
            EDG_master.rank_candidates()
            EDG_master.pick_entity()
            time2 = time.time()
            print('消歧代价', time2 - time1)
        print('消歧完毕')
