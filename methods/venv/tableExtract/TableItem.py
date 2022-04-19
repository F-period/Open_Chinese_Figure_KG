#!/user/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
os.environ['JAVA_HOME'] = '/home/ghn/java/jdk1.8.0_251'
from pyhanlp import HanLP
from FileIO import FileIO


class TableItem:
    """
    表格单元类
    """

    def __init__(self, content, rowLoc, rowspan, colLoc, colspan,
                 href=None, imgSrc=None, type_=None, tagName=None):
        """
        初始化函数，得到一个单元格
        :param content: 单元格的内容
        :param rowLoc: 所在行位置
        :param rowspan: 行跨度
        :param colLoc: 所在列位置
        :param colspan: 列跨度
        :param href: 链接实体的url
        :param type_: 表格类型
        :param tagName: 标签名
        """
        if href is None:
            href = {}
        self.content = content  # 表格单元内容
        self.rowLoc = rowLoc  # 表格单元的行位置
        self.rowspan = rowspan  # 表格单元的行占格
        self.colLoc = colLoc  # 表格单元的列位置
        self.colspan = colspan  # 表格单元的列占格
        self.span = False
        self.absoluteRow = self.rowLoc  # 表格单元绝对行位置
        self.absoluteCol = self.colLoc  # 表格单元绝对列位置
        self.href = href  # 表格单元中含有的链接
        self.type_ = type_  # 表格单元的类型
        self.wordType = None  # 表格单词类型
        self.tagName = tagName  # 表格单元的标签名
        self.particular = {}  # 主要是看有没有加粗<b>
        self.hasIndex = False  # 有没有那种类似[12]这种
        self.multi = False  # 一个单元格里有好多条信息(由于百科编辑的不规范而导致)
        self.key_content = ''
        self.hasPic = False


    def getTableItemType(self) -> str:
        """
        求得表格单元的类型
        :return: 返回类型值
        """
        # if self.type_:
        #     return self.type_
        typeSymbol = re.compile(r"^[\W]*$")  # 匹配符号类型
        typeNumber = re.compile(r"^([\$\uFFE5]?)(-?)(\d+)(\.\d+)?([\u4e00-\u9fa5\%]?)$")  # 匹配数字类型
        typeNumLess0 = re.compile(r"^((-\d+(\.\d+)?)|(0+(\.0+)?))$")  # 小于等于0的数字范围
        typeNum0_1 = re.compile(r"^0(\.\d+)?$")  # 0-1的数字范围
        typeNumGreater1 = re.compile(r"^(([1-9]\d+)|[1-9])(\.[\d]*)?$")  # 大于1的数字范围
        typeChinese = re.compile(r"[\u4e00-\u9fa5]+$")  # 匹配纯汉字
        typeEnglish = re.compile(r"[A-Za-z]+$")  # 匹配英语
        typeEngLowCase = re.compile(r"[a-z]+$")  # 匹配英语小写
        typeEngUpperCase = re.compile(r"[A-Z]+$")  # 匹配英语大写
        typeCharacterAndNum = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9]+$")  # 字符数字类型表达式
        typeHypeLink = re.compile(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]")
        content = str(self.content).strip()
        if re.match(typeHypeLink, content):
            self.type_ = "超链接"
        elif re.match(typeSymbol, content):
            self.type_ = "标点类型"
        elif re.match(typeCharacterAndNum, content):
            if re.match(typeNumber, content):  # 数字类型
                if re.match(typeNumLess0, content):
                    self.type_ = "<=0"
                elif re.match(typeNum0_1, content):
                    self.type_ = "0-1"
                elif re.match(typeNumGreater1, content):
                    self.type_ = ">=1"
                else:
                    self.type_ = "数字类型"
            else:  # 字符类型
                if re.match(typeChinese, content):
                    self.type_ = "中文"
                elif re.match(typeEnglish, content):
                    if re.match(typeEngUpperCase, content):
                        self.type_ = "大写"
                    elif re.match(typeEngLowCase, content):
                        self.type_ = "小写"
                    else:
                        self.type_ = "大小写混合"
                else:
                    self.type_ = "字符类型"
        else:
            self.type_ = "其他类型"
        return self.type_

    def getTableItemWordType(self):
        """
        获得单元格的单词类型
        :return: 单元格的单词类型
        """

        path = r'configuration/other/WordMap.pkl'
        if self.wordType:
            return self.wordType
        segment = HanLP.newSegment()
        segment.enableNameRecognize(True)
        result = list(segment.seg(str(self.content)))
        typeList = [str(pair.nature) for pair in result]
        wordDict = FileIO.readPkl(path)
        numSum = 0
        for type_ in typeList:
            numSum += wordDict[type_]
        # typeString = "".join(typeList)
        # self.wordType = typeString
        self.wordType = numSum
        return self.wordType
