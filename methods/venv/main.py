# coding=utf-8
# @Time : 2021/7/29 19:14

import json
import time
import csv
import os
import pickle
from bs4 import BeautifulSoup
from preprocess import *

from tableExtract.table import *
from candidate import *
from disambiguation import *
from pre1 import *

if __name__ == '__main__':
    # 连接数据库
    db = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='123456',
        database='baikepedia'
    )
    cursor = db.cursor()  # 数据库的接口

    table_path_in = r'/data1/baikepedia/data/tables1'  # 表格数据
    synonym_path = r'data/KB2/3.0_baidubaike_synonym_zh.txt'  # 同义词文件
    triple_path = r'/data1/baikepedia/data/temp_'  # 三元组
    # 载入停止词文件s
    path = r'data/hit_stopwords.txt'
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]

    # 第一个执行 这样candidate_generator.label_dict之后就可以直接使用了
    candidate_generator = Candidate(synonym_path, cursor, stopwords)
    D = Disambiguation()

    path_list = os.listdir(table_path_in)  # 读取输入文件夹
    path_list.sort(key=lambda x: int(x.split('.')[0]))  # 遵循数字大小将目录内容排序
    file_count = 0  # 统计当前处理到第几个文件了
    record_count = 0
    start_flag = 0
    for suffix_path in path_list:
        if suffix_path != '19480775_3.pickle' and not start_flag:  # 测试用
            continue
        file_count += 1
        url = table_path_in + r'/' + suffix_path
        with open(url, 'rb') as file:
            _table = pickle.load(file)
            table_number = suffix_path.strip('.pickle')
            print("suffix_path: ", suffix_path)
            print("file_count: ", file_count)
            try:
                _table, flag = dealWithTableList(_table)  # 规整 判断方向 转置 清理 排除数据表
            except Exception as e:
                print('发生异常1')
                print(_table.prefix)
                print(e)
                continue
            if not flag:
                print('--此表格被淘汰--表格是:', _table.prefix, '的', _table.name)
                continue

            if _table.tableType != '获奖关系表' and _table.rowNumber * _table.colNumber <= 100:  # 当表格太大时 效率起见就不做实体链接了
                # 候选实体生成
                L = candidate_generator.generateCandidate(_table)
                entity_context = candidate_generator.generateContext(_table, L)

                # 实体消歧
                D.disambiguation(table_number, _table, L, entity_context)

            # 关系抽取
            try:
                local = triple_path + r'/' + table_number
                record_count += _table.extractRelationship(local)
                print('record_count', record_count)
            except Exception as e:
                print('发生异常2')
                print(e)
                print(_table.__dict__)
                print('三元组数量', record_count)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('处理完成')
