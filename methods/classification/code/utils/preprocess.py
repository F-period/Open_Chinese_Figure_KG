import re
import os
import json
import random

# 【训练模型的辅助模块】
# 【实现预处理】

# 【数据载入】
class MyDataLoader(object):
    def __init__(self):
        pass

    @staticmethod
    # 【获取训练数据】
    def getTrainData(train_path):
        train_size = 0
        val_size = 0
        test_size = 0
        data = []
        with open(train_path, 'r', encoding='utf-8') as input:
            head = True
            for line in input:
                # 【去除训练数据中的空白符】
                item = line.rstrip('\n').split('\t')
                if head:
                    # 【对开头进行特殊处理，获取元数据】
                    train_size, val_size, test_size = int(item[0]), int(item[1]), int(item[2])
                    head = False
                else:
                    # 【非开头直接读入，转换为列表形式】
                    data.append((list(item[0]), item[1])) # convert raw words to list

        # 【根据是否有测试数据进行不同的返回】
        if test_size > 0:
            return data[:train_size], data[train_size:-test_size], data[-test_size:]
        else:
            return data[:train_size], data[train_size:], []

    @staticmethod
    # 【获取测试数据】
    def getTestData(test_path):
        data = []
        # 【去除训练数据中的空白符，转换为列表形式】
        with open(test_path, 'r', encoding='utf-8') as input:
            for line in input:
                item = line.rstrip('\n')
                data.append(list(item))  # convert raw words to list

        return data

# 【数据内容载入】
class MyDataContentLoader(object):
    def __init__(self):
        pass

    @staticmethod
    # 【这里和上面是一致的】
    def getTrainData(train_path):
        train_size = 0
        val_size = 0
        test_size = 0
        data = []
        with open(train_path, 'r', encoding='utf-8') as input:
            head = True
            for line in input:
                item = line.rstrip('\n').split('\t')
                if head:
                    train_size, val_size, test_size = int(item[0]), int(item[1]), int(item[2])
                    head = False
                else:
                    data.append((item[0], item[1], item[2])) # convert raw words to list

        if test_size > 0:
            return data[:train_size], data[train_size:-test_size], data[-test_size:]
        else:
            return data[:train_size], data[train_size:], []

    @staticmethod
    # 【跟上面比起来多加了验证集】
    def getTestData(test_path):
        data = []
        val_item = set()
        with open(test_path, 'r', encoding='utf-8') as input:
            for line in input:
                item = line.rstrip('\n').split('\t')
                val_item.add(item[0])
                data.append((item[0], item[1]))  # convert raw words to list

        return data, val_item
