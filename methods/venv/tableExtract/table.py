import json
import os
import re
from copy import deepcopy
import csv
import Levenshtein
import sys
import numpy as np
from bs4 import NavigableString
from bs4 import Tag
from treelib import Tree

from FileIO import FileIO
from tableExtract.TableItem import TableItem


class Table:
    """
    表格类
    """
    def __init__(self, rowNumber: int = 0, colNumber: int = 0, name: str = "未命名表格",
                 table=None, unfoldDirection=None):
        """
        初始化函数
        :param rowNumber: 表格的行数
        :param colNumber: 表格的列数
        :param name: 表格的名称
        :param table: 表格的单元格数组，这是一个二维数组
        :param unfoldDirection: 表格的展开方向
        """
        self.rowNumber = rowNumber  # 表格的行数
        self.colNumber = colNumber  # 表格的列数
        if table is None:  # 表格所在的二维数组
            self.cell = [[TableItem(content=0, rowLoc=j, rowspan=1, colLoc=i, colspan=1)
                          for i in range(self.colNumber)]
                         for j in range(self.rowNumber)]
        else:
            self.cell = table
        self.name = name  # 表格的名称
        self.prefix = None  # label
        self.unfoldDirection = unfoldDirection  # 表格的展开方向
        self.__isCorrect = True  # 当前表格是否正确，行列单元数相同
        self.__isNormal = True  # 当前表格是否正常，行列数均大于等于2 不且含图片
        self.propertyList = []  # 属性单元列表
        self.propertyNameList = []  # 属性名列表
        self.propertyLineNum = 1  # 属性行数
        self.tableType = None  # 表格类型
        self.centerWord = None  # 中心词汇,例如人物表，中心词汇就是人名，如“李渊”
        self.hrefMap = ''  # label的网址
        self.personNameIndex = -1
        self.hasPic = False
        self.deleted = False
        self.occupation = []
        self.getAbsolutePosition()  # 获取表格单元的绝对位置
        self.initialNormal()  # 判断表格是否正常
        self.initialTableItemsType()  # 初始化表格单元的类型

    def extractRelationship(self, out_path):
        """
        抽取实体关系
        :return:从当前表格中抽取的实体列表和关系列表
        """
        relationship = []
        record_count = 0
        typeName = self.getTableType()

        if typeName == "标题关系表":  #
            if self.isPropertyRelationShipTable():  # 先判断有无 父子 夫妻等关系可以提取
                relationship = self.extractPropertyRelationship()  # 提取亲戚关系表
                if len(relationship) != 0:
                    record_count += self.persistence3(relationship, out_path)
                    return record_count

            # 若没有 直接提取标题关系
            relationship = self.extractCaptionRelationship()  # 提取标题关系组
            if len(relationship):
                record_count += self.persistence1(relationship, out_path)
            else:
                print('未能提取标题关系')

        elif typeName == "艺术形象关系表":
            relationship = self.extractYSXXRelationship()
            if len(relationship):
                record_count += self.persistence2(relationship, out_path)
            else:
                print('未能提取标题关系')
        elif typeName == "实体关系表":
            relationship = self.extractEntityRelationship()
            record_count += self.persistence4(relationship, out_path)
        elif typeName == "获奖关系表":
            relationship= self.extractRewardRelationship()
            record_count += self.persistence5(relationship, out_path)
        else:  # 其他表
            pass
        return record_count

    def getTableType(self):
        """
        识别表格类型
        :return:表格的类型
        """
        if self.tableType:  # 获奖属性表会在这里直接返回
            print('此表格类型:', self.tableType)
            return self.tableType
        else:
            if self.isXXRelationshipTable():
                self.tableType = "艺术形象关系表"
            elif self.isCaptionRelationShipTable():
                self.tableType = "标题关系表"
            elif self.isEntityRelationshipTable():
                self.tableType = "实体关系表"
            else:
                self.tableType = "其他表"
        print('此表格类型:', self.tableType)
        return self.tableType

    def getMentionContext(self, r, c):
        mention_context = []

        for i in range(self.rowNumber - 1):  # 第一排的属性就不拿了
            if i + 1 == r:
                continue
            mention_context.append(self.cell[i][c].content)  # unicode

        for j in range(self.colNumber):
            if j == c:
                continue
            mention_context.append(self.cell[r][j].content)  # unicode

        return mention_context

    # 处理rowspan之类的问题
    def extendTable(self):
        """
        将当前表格扩展为规范表格
        :return: 扩展后的表格
        """
        # 行扩展
        for rows in self.cell:
            for item in rows:
                item.span = False

        for rows in self.cell:
            before = 0
            for item in rows:
                if item.rowspan > 1:
                    rowspan = item.rowspan
                    item.span = True
                    item.rowspan = 1
                    for row in range(item.absoluteRow + 1, item.absoluteRow + rowspan):
                        newItem = deepcopy(item)
                        newItem.rowLoc = row
                        newItem.absoluteRow = row
                        self.cell[row].insert(before, newItem)
                before += 1
        # 列扩展
        for rows in self.cell:
            for item in rows:
                if item.colspan > 1:
                    colspan = item.colspan
                    item.span = True
                    item.colspan = 1
                    for col in range(item.absoluteCol + 1, item.absoluteCol + colspan):
                        newItem2 = deepcopy(item)
                        newItem2.colLoc = col
                        newItem2.absoluteCol = col
                        self.cell[item.absoluteRow].insert(item.absoluteCol, newItem2)
        self.initialNormal()
        self.initialCorrect()
        return self

    def isCorrect(self):
        """
        判断当前表格是否正确，即行列单元数相同
        :return:
        """
        return self.__isCorrect

    def isNormal(self):
        """
        判断当前表格是否正常，即行列数均大于等于2
        :return:
        """
        return self.__isNormal

    def deleteOneRow(self, index: int):
        """
        删除指定行
        :param index:要删除的索引号，例如Index=0代表第1行
        :return:
        """
        if self.__isCorrect and self.__isNormal:
            if index < 0 or index >= self.rowNumber:
                raise Exception(f"要删除的行<{index}>超出行数范围<0,{self.rowNumber - 1}>")
            del self.cell[index]
            self.rowNumber -= 1
            self.getAbsolutePosition()
            self.initialPropertyList()
        else:
            raise Exception("当前表格未规整，无法删除行")

    def deleteOneCol(self, index: int) -> list:
        """
        删除指定列
        :param index: 要删除的索引号，例如Index=0代表第1列
        :return:无
        """
        if self.__isCorrect and self.__isNormal:
            if index < 0 or index >= self.colNumber:
                raise Exception(f"要删除的列<{index}>超出列数范围<0,{self.colNumber - 1}>")
            column = []
            for i in range(self.rowNumber):
                temp = []
                temp.append(self.cell[i][index].content)
                if self.cell[i][index].href.__contains__(self.cell[i][index].content):
                    temp.append(self.cell[i][index].href[self.cell[i][index].content])
                else:
                    temp.append('null')
                column.append(temp)
                del self.cell[i][index]
            if index < self.personNameIndex:
                self.personNameIndex -= 1
            self.getAbsolutePosition()
            self.colNumber -= 1
            self.initialPropertyList()
            return column
        else:
            raise Exception("当前表格未规整，无法删除列")

    def flip(self):
        """
        翻转表格方向,并返回一个新的矩阵
        :return:返回翻转方向后的矩阵
        """
        newTable = Table(rowNumber=self.colNumber, colNumber=self.rowNumber, name=self.name)
        for i in range(self.rowNumber):
            for j in range(self.colNumber):
                newTable.cell[j][i] = deepcopy(self.cell[i][j])
        if self.unfoldDirection == "ROW":
            newTable.unfoldDirection = "COL"
        if self.unfoldDirection == "COL":
            newTable.unfoldDirection = "ROW"

        newTable.prefix = self.prefix  # 表格的前驱
        newTable.propertyList = self.propertyList  # 属性单元列表
        newTable.propertyNameList = self.propertyNameList  # 属性名列表
        newTable.propertyLineNum = self.propertyLineNum  # 属性行数
        newTable.tableType = self.tableType  # 表格类型
        newTable.centerWord = self.centerWord  # 中心词汇,例如人物表，中心词汇就是人名，如“李渊”
        newTable.hrefMap = self.hrefMap  # 超链接映射
        newTable.initialNormal()  # 判断表格是否正常
        newTable.initialCorrect()
        return newTable

    def getTableItemLengthCharacter(self):
        """
        计算矩阵的几何特征，返回行方差均值和列方差均值，方差越小，则按照该方式展开的可能性越大
        :return: 方差均值和列方差均值
        """
        # 这段的意思是去掉表格的首行首列(因为属性这一行的字段可能会和值不太相像)
        row_bias = col_bias = 1
        if self.rowNumber < 3:
            row_bias = 0
        if self.colNumber < 3:
            col_bias = 0

        data = np.zeros((self.rowNumber - row_bias, self.colNumber - col_bias), dtype=int)
        for i in range(self.rowNumber - row_bias):
            for j in range(self.colNumber - col_bias):
                data[i, j] = len(str(self.cell[i + row_bias][j + col_bias].content))
        colVarianceMean = np.mean(np.std(data, axis=0))  # 列方差均值
        rowVarianceMean = np.mean(np.std(data, axis=1))  # 行方差均值
        sumNumber = rowVarianceMean + colVarianceMean
        if sumNumber == 0:
            return rowVarianceMean, colVarianceMean
        return rowVarianceMean / sumNumber, colVarianceMean / sumNumber

    def getTableItemTypeCharacter(self):
        """
        计算矩阵的类型特征，返回行方差均值和列方差均值，方差越小，则按照该方式展开的可能性越大
        :return: 方差均值和列方差均值
        """
        _typeTree = TypeTree()
        return _typeTree.getTypeCharacter(self)

    def getTableItemWordTypeCharacter(self):
        """
        获得行列的单词类型差异
        :return:
        """
        self.initialTableItemWordType()
        for row in self.cell:
            for col in row:
                print(col.wordType, end=" ")
            print()
        data = np.zeros((self.rowNumber, self.colNumber), dtype=int)
        for i in range(self.rowNumber):
            for j in range(self.colNumber):
                data[i, j] = self.cell[i][j].wordType

        colVarianceMean = np.mean(np.std(data, axis=0))  # 列方差均值
        rowVarianceMean = np.mean(np.std(data, axis=1))  # 行方差均值
        sumNumber = rowVarianceMean + colVarianceMean
        if sumNumber == 0:
            return rowVarianceMean, colVarianceMean
        return rowVarianceMean / sumNumber, colVarianceMean / sumNumber

    def getRowAt(self, row: int):
        """
        获取表格第row行的数据列表,如果获取不到则抛出异常
        :param row: 行数，从0开头
        :return: 第row行对应的数据列表
        """
        if self.__isNormal and self.__isCorrect:
            if 0 <= row < self.rowNumber:
                return self.cell[row]
            else:
                raise Exception(f"row={row},此时超出表格索引范围")
        else:
            raise Exception(f"当前表格不正常，无法获取第{row}行的数据列表")

    def getColAt(self, col: int):
        """
        获取表格第col列的数据列表,如果获取不到则抛出异常
        :param col: 列数，从0开头
        :return: 第col列对应的数据列表
        """
        if self.__isNormal and self.__isCorrect:
            if 0 <= col < self.colNumber:
                res = []
                for row in range(self.rowNumber):
                    res.append(self.cell[row][col])
                return res
            else:
                raise Exception(f"col={col},此时超出表格索引范围")
        else:
            raise Exception(f"当前表格不正常，无法获取第{col}列的数据列表")

    def getUnfoldDirection(self) -> str:
        """
        返回表格的展开方向,只能判断为横向展开或者纵向展开
        :return: "ROW"表示横向展开，"COL"表示纵向展开
        """
        if self.unfoldDirection:
            return self.unfoldDirection

        # 检测当前表中含了多少高频的实体表关键字
        entity_count = 0
        base_count = 0
        path = r'configuration/relationshipConf/entityProperty.json'
        entityPropertyList = FileIO.readJson(path)
        dict_ = {}
        for rows in self.cell:
            for item in rows:
                content = item.content
                if not (content == '' or content.isspace()):
                    base_count += 1
                for property_ in entityPropertyList:
                    if property_ in content:
                        if dict_.__contains__(property_):
                            continue
                        else:
                            dict_[property_] = 1
                            entity_count += 1
                            print(property_)
                            break
        if entity_count > 5 or (base_count != 0 and entity_count / base_count >= 0.44):
            self.unfoldDirection = 'ENTITY'
            self.tableType = '实体关系表'
            return self.unfoldDirection

        firstRow = self.getRowAt(0)
        firstCol = self.getColAt(0)

        # 属性行里一定没有时间或数字
        # 如果含有冒号 则一般是属性行了
        pattern = re.compile(r'^((\d)+)(\.)?(-[01]?\d+)?(-?\d{2})?年?(\d{1,2}月?)?(\d{2}日?)?')
        col_colon_flag = row_colon_flag = True
        col_time_flag = row_time_flag = True
        for row in firstRow:  # 先检查第一行
            if re.match(pattern, row.content) or row.hasIndex == True:
                row_time_flag = False
            if not re.search(r'[：:]', row.content):
                row_colon_flag = False
        for col in firstCol:  # 再检查第一列
            if re.match(pattern, col.content) or col.hasIndex == True:
                col_time_flag = False
            if not re.search(r'[：:]', col.content):
                col_colon_flag = False

        if row_time_flag ^ col_time_flag:
            if row_time_flag:
                self.unfoldDirection = "ROW"
            else:
                self.unfoldDirection = "COL"
            return self.unfoldDirection

        if row_colon_flag & col_colon_flag:  # 如果第一行和第一列都有冒号 表示每格都有 则应该是实体表
            self.unfoldDirection = "ENTITY"
            self.tableType = '实体关系表'
            return self.unfoldDirection
        if row_colon_flag | col_colon_flag:
            if row_colon_flag:
                self.unfoldDirection = "ROW"
            else:
                self.unfoldDirection = "COL"
            return self.unfoldDirection

        # 标签识别
        rowRes = [item.tagName == 'th' for item in firstRow]
        if rowRes[0] and len(set(rowRes)) == 1:
            self.unfoldDirection = "ROW"
            return self.unfoldDirection
        colRes = [item.tagName == 'th' for item in firstCol]
        if colRes[0] and len(set(colRes)) == 1:
            self.unfoldDirection = "COL"
            return self.unfoldDirection

        # 如果只有第一列(行)全都被加粗了 那就是属性
        row_b_flag = col_b_flag = 1  # 哪个最后仍为1表示可以作为属性列
        row_href_flag = col_href_flag = 1
        for row in firstRow:
            if not row.particular.__contains__('b'):
                row_b_flag = 0
            if len(row.href) != 0:  # 当前单元格有链接 基本不可能是属性行
                row_href_flag = 0
        for col in firstCol:
            if not col.particular.__contains__('b'):
                col_b_flag = 0
            if len(col.href) != 0:
                col_href_flag = 0

        if col_b_flag ^ row_b_flag:  # 异或操作 表示有一个含有b
            if row_b_flag:
                self.unfoldDirection = "ROW"
            else:
                self.unfoldDirection = "COL"
            return self.unfoldDirection
        if row_href_flag ^ col_href_flag:  # 异或操作 表示其中之一含有href
            if row_href_flag:
                self.unfoldDirection = "ROW"
            else:
                self.unfoldDirection = "COL"
            return self.unfoldDirection

        # 除第一行第一列外, 如果有某列重复出现某值 则应该是行展开的表格了
        row_repeat_flag = col_repeat_flag = 0
        for i in range(self.rowNumber):
            if not i:  # 第一行第一列就不循环了
                continue
            row_dict = {}
            for j in range(self.colNumber):
                if not j:
                    continue
                cell = self.cell[i][j]
                if cell.content.isspace() or cell.content == '' or cell.span:
                    continue
                if row_dict.__contains__(cell.content):
                    row_repeat_flag = 1
                else:
                    row_dict[cell.content] = 1
        for j in range(self.colNumber):
            if not j:
                continue
            col_dict = {}
            for i in range(self.rowNumber):
                if not i:
                    continue
                cell = self.cell[i][j]
                if cell.content.isspace() or cell.content == '' or cell.span:
                    continue
                if col_dict.__contains__(cell.content):
                    col_repeat_flag = 1
                else:
                    col_dict[cell.content] = 1
        if row_repeat_flag ^ col_repeat_flag:  # 异或操作 表示其中之一含有重复值
            if not row_repeat_flag:
                self.unfoldDirection = "ROW"
            else:
                self.unfoldDirection = "COL"
            return self.unfoldDirection

        # 这个就有点凭直觉的样子
        if self.rowNumber < 3 and self.colNumber >= 3:
            self.unfoldDirection = "ROW"
            return self.unfoldDirection
        if self.colNumber < 3 and self.rowNumber >= 3:
            self.unfoldDirection = "COL"
            return self.unfoldDirection

        # cell长度和类型判断法
        rowVarianceMean, colVarianceMean = self.getTableItemLengthCharacter()
        rowTypeCharacter, colTypeCharacter = self.getTableItemTypeCharacter()
        W1 = 0.4
        W2 = 0.6
        Row = W1 * rowVarianceMean + W2 * rowTypeCharacter
        Col = W1 * colVarianceMean + W2 * colTypeCharacter
        if Row < Col:
            direction = "COL"
        elif Row == Col:
            # 词性和判断法
            rowWordTypeVarianceMean, colWordTypeVarianceMean = self.getTableItemWordTypeCharacter()
            if rowWordTypeVarianceMean < colWordTypeVarianceMean:
                direction = "ROW"
            elif rowWordTypeVarianceMean > colWordTypeVarianceMean:
                direction = "COL"
            else:
                direction = "ROW"  # 如果无法判断，则判断为横向
        else:
            direction = "ROW"
        self.unfoldDirection = direction
        return self.unfoldDirection

    def getAbsolutePosition(self):
        """
        获得表格中每个项目所在的绝对位置，其中行绝对位置为self.absoluteRow,列绝对位置为self.absoluteCol
        :return:无
        """
        positionList = []
        for i in range(len(self.cell)):
            colIndex = 0
            before = 0  # 记录从这一行开始，到现在，之前有几个元素进入队列
            for j in range(len(self.cell[i])):
                data = self.cell[i][j]
                colStart = 0
                for position in positionList:
                    colStart += position[1]
                data.absoluteCol = colStart + j - before
                data.absoluteRow = i
                if data.rowspan > 1 or data.colspan > 1:
                    positionList.append([data.rowspan, data.colspan])
                    before += 1
                colIndex += 1

            for x in reversed(range(len(positionList))):
                if positionList[x][0] > 1:
                    positionList[x][0] -= 1
                else:
                    positionList.pop(x)

    # 返回属性的列表 可选择是否为纯净版(非纯净版指的就是带有tableItem痕迹的)
    def getPropertyList(self, isPropertyName=False) -> list:
        """
        获取属性所在的列表
        :isPropertyName:是否只返回纯净版属性名的列表 默认是全部返回的
        :return:属性单元格列表
        """
        if not isPropertyName:
            if self.propertyList:
                return self.propertyList
        else:
            if self.propertyNameList:
                return self.propertyNameList
        self.initialPropertyList()
        if not isPropertyName:
            return self.propertyList
        else:
            self.propertyNameList = [str(item.content) for item in self.propertyList]
            return self.propertyNameList

    def clearPersonNameList(self, personNameList: list):
        """
        将人名变成清晰干净的名字
        :param personNameList:
        :return:
        """
        punctuation = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+"
        for i in range(len(personNameList)):
            personNameList[i] = re.sub(u"\[.*\]|\(.*\)|\<.*\>|（.*）|【.*】|\{.*\}|（.*\)|\(.*）", "",
                                       personNameList[i])  # 去除括号
            personNameList[i] = str(personNameList[i]).split("/")[0]
            personNameList[i] = re.sub(punctuation, "", personNameList[i])
        return personNameList

    def clearPunctuation(self):
        """
        把单元格中的东西清理一下
        :param personNameList:
        :return:
        """
        for row in self.cell:
            for col in row:
                pattern2 = re.compile(r'\s*[:：]\s*')  # 为了把冒号左右的空格消掉
                content = re.sub(pattern2, ':', col.content)
                content = content.strip(' ')
                book = []  # 存带有书名号的内容(含书名号)

                left = right = 0  # 记录单对书名号的下标
                for i in range(len(content)):
                    if content[i] == '《':
                        left = i
                    elif content[i] == '》':
                        right = i
                        book.append(content[left: right + 1])

                if len(book):
                    _string = ''
                    for b in book:
                        _string += (b + '_;_')
                    col.key_content = _string
                    col.multi = True
                    continue

                # content = re.sub(u"\(.?\)|\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】||\\<.*?\\>", "", content)  # 去除括号及当中的内容
                content = re.sub(u"\[.*\]|\(.*\)|\<.*\>|（.*）|【.*】|\{.*\}|（.*\)|\(.*）", "", content)  # 去除括号及当中的内容
                content = content.split(',，')[0]  # 逗号后面的东西都扔了
                col.content = content
                col.key_content = content
                L = []
                left = 0

                for i in range(len(content)):  # 视 空格 顿号两边文字为并列
                    if content[i] == ' ':
                        if i - 1 >= 0 and (content[i - 1].isalpha() or content[i - 1].isdigit()):  # 如果前一个下标存在且为字母
                            continue
                        else:
                            L.append(content[left: i])
                        left = i + 1
                    elif content[i] == '、':
                        L.append(content[left: i])
                        left = i + 1

                if len(L):
                    L.append(content[left: len(content)])
                    _string = ''
                    for b in L:
                        _string += (b + '_;_')
                    col.key_content = _string
                    col.multi = True

    def getPersonColList(self, deleteCol=False, removeHeader=False, getName=False) -> list:
        """
        获取人名列表
        :param deleteCol:是否删除人名的这一列
        :param removeHeader:是否去除表头,一般是属性栏
        :param getName: 是否获取人名  即是否从tableItem中提取纯净的文字出来
        :return:人名的那一列
        """
        personList = []
        personNameIndex = self.__getPersonNameIndex()
        if personNameIndex != -1:
            personList = [person for person in self.getColAt(personNameIndex)]  # 获得人名所在的表格列
        if len(personList) == 0:
            return personList
        if removeHeader:
            propertyLineNum = self.discriminatePropertyLineNum(self.getUnfoldDirection())
            personList.pop(propertyLineNum - 1)
        if getName:  # 从tableItem中提取纯净的文字出来
            personList = [str(person.content) for person in personList]
            personList = self.clearPersonNameList(personList)  # 清理人名
        if deleteCol:
            self.deleteOneCol(personNameIndex)
        return personList

    # 判断可能因为rowspan而导致多行的属性 返回属性实际行数
    def __tagDiscriminatePropertyLineNum(self, direction: str):
        """
        根据标签判断表格的属性行数，该方法执行前必须先判断表格的展开方向
        :param direction: 表格的展开方向
        :return:
        """
        res = 0
        if direction == "ROW":
            for i in range(self.rowNumber):
                for j in range(self.colNumber):
                    item = self.cell[i][j]
                    if item.tagName != "th":
                        return res
                res += 1
            return res
        elif direction == "COL":
            for j in range(self.colNumber):
                for i in range(self.rowNumber):
                    item = self.cell[i][j]
                    if item.tagName != "th":
                        return res
                res += 1
            return res
        else:
            raise Exception(f"不存在这种表格展开方向<{direction}>")

    def __typeDiscriminatePropertyLineNum(self, direction: str) -> int:
        """
        根据类型判断属性行列数
        :param direction: 展开方向，目前有"ROW"，即行展开，和"COL"，即列展开
        :return: 属性行列数 n，若无法判别，则返回 0
        """
        characterTypeSet = {"字符类型", "中文", "英文", "大写", "小写", "大小写混合"}
        res = 0
        if direction == "ROW":
            for i in range(self.rowNumber):
                for j in range(self.colNumber):
                    item = self.cell[i][j]
                    if item.type_ not in characterTypeSet:
                        return res
                res += 1
            if res == self.rowNumber:  # 如果遍历了所有行
                res = 0
        elif direction == "COL":
            for i in range(self.colNumber):
                for j in range(self.rowNumber):
                    item = self.cell[j][i]
                    if item.type_ not in characterTypeSet:
                        return res
                res += 1
            if res == self.colNumber:  # 如果遍历了所有列
                res = 0
        else:
            raise Exception(f"不存在这种表格展开方向<{direction}>")
        return res

    def discriminatePropertyLineNum(self, direction: str):
        """
        判断表格的属性行数，该方法执行前必须先判断表格的展开方向
        :param direction: 表格的展开方向
        :return:
        """
        if self.propertyLineNum:
            return self.propertyLineNum
        res = self.__tagDiscriminatePropertyLineNum(direction)
        if res == 0 or res > 2:  # 属性若是2行以上 就有点异常了 通过一些启发式方法测一下
            res = self.__typeDiscriminatePropertyLineNum(direction)
            if res == 0:
                res = 1
        self.propertyLineNum = res
        return self.propertyLineNum

    def checkDigitNumber(self):
        """
        检查单元格为数字的占比
        """
        count = 0
        total = (self.rowNumber - 1) * self.colNumber

        #  还有一类非常不规整的表 暂时放到这个函数下判断
        counter = 0  # counter统计符合这类不规整表格的特征数
        L = ['合作关系', '人物名称', '合作作品']
        if self.name == '人物关系' or self.name == '合作关系':
            counter += 1

        blank_count = 0  # 统计空单元格数量
        for rows in self.cell:
            for item in rows:
                pattern = re.compile(r'^(\d)+(\.?(\d)+)*$')
                if re.match(pattern, item.content):
                    count += 1
                elif item.content.isspace() or item.content == '':
                    total -= 1
                    blank_count += 1
                if item.content in L:
                    counter += 1

        if counter >= 3 or (counter >= 2 and self.deleted):
            return True

        if self.rowNumber - 1 <= 0:
            return False

        if blank_count / (self.rowNumber - 1) / self.colNumber > 0.45:
            return True

        if total != 0 and count / total > 0.5:
            return True
        else:
            return False

    def initialTableItemWordType(self):
        """
        获得单词类型，例如"水果"就是名词，“跑步”就是动词，如果是句子就会划分为多个词
        :return:无
        """
        for row in self.cell:
            for item in row:
                item.getTableItemWordType()

    def initialTableItemsType(self):
        """
        初始化表格每一个单元的类型，如“你好”就是中文，“123”就是数字>1，“hello”就是英文
        :return:无
        """
        for row in self.cell:
            for item in row:
                item.getTableItemType()

    def initialCorrect(self) -> bool:
        """
        判断表格是否正确，正确表格的行与列单位数都非常规整
        :return:表格正确则返回True，表格错误则返回False
        """
        colLenList = []
        for rows in self.cell:
            colLen = 0
            for col in rows:
                colLen += col.colspan
            colLenList.append(colLen)
        self.__isCorrect = (len(set(colLenList)) == 1)
        return self.__isCorrect

    def initialNormal(self) -> bool:
        """
        判断是否是一个正常的表格，正常表格必须行列数都大于2
        :return:正常表格则返回True，否则返回False
        """
        if self.hasPic:  # 有图片直接判负
            self.__isNormal = False
            return self.__isNormal

        if self.rowNumber >= 2 and self.colNumber >= 2:
            self.__isNormal = True
        else:
            self.__isNormal = False
        return self.__isNormal

    def initialPropertyList(self):
        """
        初始化表格的属性列表
        :return: 无
        """
        direction = self.getUnfoldDirection()
        propertyLineNum = self.discriminatePropertyLineNum(direction)
        if direction == "ROW":
            self.propertyList = self.getRowAt(propertyLineNum - 1)
        elif direction == "COL":
            self.propertyList = self.getColAt(propertyLineNum - 1)
        elif direction == "ENTITY":
            self.propertyList = self.getRowAt(propertyLineNum - 1)
        else:
            raise Exception(f"不存在该表格展开方向<{self.name, self.prefix}>")
        self.propertyNameList = [str(p.content) for p in self.propertyList]

    def isPropertyRelationShipTable(self) -> bool:
        """
        判断是否为标题关系表
        :return:是则返回True，不是则返回False
        """
        # 有列为关系
        path = r'configuration/relationshipConf/propertyRelationship.json'  # 这里的路径要以FileIO的位置为准
        propertyRelationShipList = FileIO.readJson(path)
        propertyList = self.getPropertyList(isPropertyName=True)
        for propertyName in propertyList:
            for relationshipName in propertyRelationShipList:
                if relationshipName in propertyName:
                    return True
        # 属性行为关系
        propertyNameList = self.getPropertyList(isPropertyName=True)
        path = r'configuration/relationshipConf/captionRelationship.json'
        CRList = FileIO.readJson(path)
        count = 0
        for propertyName in propertyNameList:
            for CR in CRList:
                if CR in propertyName:
                    count += 1
                    continue
        if count > len(propertyNameList) / 2:
            return True
        return False

    def isStatisticTable(self) -> bool:
        if self.name and self.prefix:
            predicate = self.name
            likelihood = 0
            path = r'configuration/relationshipConf/statisticRelation.json'
            path2 = r'configuration/relationshipConf/statisticProperty.json'
            relationshipList = FileIO.readJson(path)
            propertyList = FileIO.readJson(path2)

            for i in range(len(relationshipList)):
                if relationshipList[i] in predicate:
                    likelihood += (0.8 - i * 0.05)

            count1 = 0
            propertyNameList = self.getPropertyList(isPropertyName=True)
            for property in propertyNameList:
                for pro in propertyList:
                    if pro in property:
                        count1 += 1
                        break

            likelihood += (count1 / len(propertyNameList))

            if self.checkDigitNumber():
                likelihood += 1
            if likelihood >= 0.65:
                return True
        return False

    def isXXRelationshipTable(self) -> bool:

        flag = False
        path1 = r'configuration/relationshipConf/YSXX/title.json'

        if self.name and self.prefix:
            predicate = self.name
            key_title_list = FileIO.readJson(path1)

            for key_title in key_title_list:
                if key_title == predicate:
                    flag = True
                    return flag

        return flag

    def isCaptionRelationShipTable(self) -> bool:
        """
        判断是否为标题关系表
        :return:是则返回True，否则返回False
        """
        idx = -1
        path1 = r'configuration/relationshipConf/personName.json'
        path2 = r'configuration/relationshipConf/notPossible.json'
        path3 = r'configuration/relationshipConf/entityCaption.json'
        pivot_prob = [0] * self.colNumber  # 记录每列作为主列的可能性
        if self.name and self.prefix:
            predicate = self.name
            propertyNameList = self.getPropertyList(isPropertyName=True)

            if 'MV' in predicate or 'mv' in predicate:  # 把mv替换成中文名字更容易识别主列
                predicate = predicate.strip('MV').strip('mv')
                predicate += '曲' + '音乐' + '作品'

            # 有些表格的标题就使他更加有几率成为实体关系表
            captionList = FileIO.readJson(path3)
            for caption in captionList:
                if caption in predicate:
                    pivot_prob = [-0.3] * self.colNumber

            # 属性行不应该有重复
            _dict = {}
            for j in range(self.colNumber):
                if _dict.__contains__(propertyNameList[j]):
                    pivot_prob[_dict[propertyNameList[j]]] = -2
                    pivot_prob[j] = -2
                else:
                    _dict[propertyNameList[j]] = j

            # 如果孙燕姿的表, 孙燕姿自己出现在了表的某列作为值 那基本不可能是这列
            # 如果某列中有某个值重复多次 则大概率也不是
            for j in range(self.colNumber):
                _dict = {}  # 用来记录重复
                for i in range(self.rowNumber):
                    if i == 0:
                        continue
                    content = self.cell[i][j].content
                    if content.isspace() or content == '':
                        continue
                    if content == self.prefix:  # 如果孙燕姿的表, 孙燕姿自己出现在了表的某列作为值 那基本不可能是这列
                        pivot_prob[j] -= 2
                    if _dict.__contains__(content):  # 该列有重复
                        _dict[content] += 1
                        if _dict[content] < 5:
                            pivot_prob[j] -= 0.1
                    else:
                        _dict[content] = 1

            # 看属性中有无概率很高的关键词
            relationshipList = FileIO.readJson(path1)
            i = 0
            for _property in propertyNameList:
                j = 0
                for relation in relationshipList:
                    j += 1
                    if relation in _property:
                        pivot_prob[i] += 0.5 - j * 0.001
                        break
                i += 1

            # 再用类jaccard公式
            i = 0  # 属性的下标
            for _property in propertyNameList:
                single_word_list = list(_property)
                c = 0
                for word in single_word_list:
                    if word in predicate:
                        c += 1
                pivot_prob[i] += 0.15 * c
                i += 1

            # 根据实体数量  顺便计算空着的单元格
            linking = [0] * self.colNumber
            count = 0  # 总实体链接数
            for i in range(self.rowNumber):
                if i == 0:
                    continue
                for j in range(self.colNumber):
                    if len(self.cell[i][j].href) != 0:
                        linking[j] += 1
                        count += 1
                    if self.cell[i][j].content.isspace() or self.cell[i][j].content == '':
                        pivot_prob[j] -= 0.1
            count *= 4  # 稀释实体链接所加的可能性
            for i in range(self.colNumber):
                if count != 0:
                    linking[i] /= count
                pivot_prob[i] += linking[i]

            # 某些列天生作为主列的可能性就很低
            relationshipList = FileIO.readJson(path2)
            i = 0
            for _property in propertyNameList:
                if _property in relationshipList:
                    pivot_prob[i] -= 0.2
                i += 1

        max = pivot_prob[0]
        idx = 0
        for i in range(self.colNumber - 1):
            if pivot_prob[i + 1] == max:
                idx = -1
            if pivot_prob[i + 1] > max:
                max = pivot_prob[i + 1]
                idx = i + 1

        if idx != -1 and pivot_prob[idx] > 0.26:
            self.personNameIndex = idx
            return True;
        return False;

    def isEntityRelationshipTable(self) -> bool:
        """
        判断是否为实体关系表
        :return:是则返回True，否则返回False
        """

        #  最粗暴的第一轮筛选
        if self.rowNumber == 1 or self.colNumber == 1:
            return False
        if self.rowNumber > 2 and self.colNumber > 2:
            return False
        if self.rowNumber > 2:
            return False

        # 属性行绝不可能有数字
        pattern = re.compile(r'^((\d)+)(\.)?(-[01]?\d+)?(-?\d{2})?年?(\d{1,2}月?)?(\d{2}日?)?')
        for property_ in self.propertyNameList:
            if re.match(pattern, property_):  # 数字(含浮点数)或者为年月的 就不链接了
                return False;

        # 凭借一种经验 第一列(行)不可能有数字
        for item in self.cell[0]:
            if re.match(pattern, item.content):
                return False;
        col = self.getColAt(0)
        for item in col:
            if re.match(pattern, item.content):
                return False;

        for row in self.cell:
            for col in row:
                temp_content = col.content
                pattern = re.compile(r'[A-Za-z0-9]+')
                temp_content = re.sub(pattern, '', temp_content)
                if len(temp_content) > 20 and not col.multi:  # 除数字英文外若字数太多的话也不太可能
                    return False

        return True

    # 表里如果有关系这个属性之类的 则将这一属性和下标位置返回(可能不止一个)
    def __getPropertyRelationshipList(self):
        """
        获取属性关系列表，并且把与人物有关的属性由高到低排序
        :return:属性关系列表
        """
        path = r'configuration/relationshipConf/propertyRelationship.json'
        propertyRelationshipList = FileIO.readJson(path)
        propertyList = self.getPropertyList(isPropertyName=True)
        indexAndNameList = []
        c1 = 0
        for propertyName in propertyList:
            if c1 == self.personNameIndex:
                continue
            c2 = 0
            for name_in_file in propertyRelationshipList:
                if name_in_file in propertyName:
                    indexAndNameList.append((c2, propertyName))
                    break
                c2 += 1
            c1 += 1
        sortIndexList = sorted(indexAndNameList, key=lambda indexAndNum: indexAndNum[0])  # 根据序号排序
        sortIndexList = [indexAndName[1] for indexAndName in sortIndexList]
        return sortIndexList

    def extractPropertyRelationship(self):
        """
        从当前表格中抽取属性关系
        :return: 属性关系列表
        """
        def listFindPosition(AList: list, waitFind: str):
            for i in range(len(AList)):
                if waitFind in AList[i]:
                    return i
            return -1

        relationship = []
        if not self.prefix:
            return relationship
        propertyNameList = self.getPropertyList(isPropertyName=True)
        if len(propertyNameList) == 0:
            return relationship
        propertyRelationshipList = self.__getPropertyRelationshipList()  # 属性关系列表，例如[关系,辈分]
        if len(propertyRelationshipList) == 0:  # 如果不存在属性关系，则返回空
            return relationship
        if len(propertyRelationshipList) >= 1:  # 如果存在多个属性关系，则删除其余低级的属性关系
            for i in range(1, len(propertyRelationshipList)):
                self.deleteOneCol(listFindPosition(propertyNameList, propertyRelationshipList[i]))
            propertyNameList = self.getPropertyList(isPropertyName=True)  # 删除属性列之后更新一下属性列表
        personNameList = self.getPersonColList(getName=True)  # 获取主列纯净版
        index = listFindPosition(propertyNameList, propertyRelationshipList[0])
        relationshipList = [str(relationship.content) for relationship in self.getColAt(index)]  # 获得关系名列表
        self.deleteOneCol(index)  # 删除关系名列表
        propertyLineNum = self.discriminatePropertyLineNum(self.getUnfoldDirection())
        prefix = self.prefix
        for i in range(propertyLineNum, self.rowNumber):
            # 构建三元组
            if i < len(relationshipList) and i < len(personNameList):
                single_relation = relationshipList[i]
                pattern = re.compile('[,，、]')
                single_relation = re.split(pattern, single_relation)[0]
                relationship.append([prefix, single_relation, personNameList[i]])
        return relationship

    # YSXX 艺术形象
    def extractYSXXRelationship(self):
        path1 = r'configuration/relationshipConf/YSXX/subject.json'
        path2 = r'configuration/relationshipConf/YSXX/object.json'
        path3 = r'configuration/relationshipConf/YSXX/predicateExtend.json'
        relationship = []
        candidate_subject = FileIO.readJson(path1)
        candidate_object = FileIO.readJson(path2)
        propertyNameList = self.getPropertyList(isPropertyName=True)  # 获取属性行

        # 先找主语 和 宾语的列
        subject_idx = 0
        object_idx = 0
        for candidate in candidate_subject:
            i = 0
            end_flag = False
            for property_ in propertyNameList:
                if candidate == property_:
                    subject_idx = i
                    end_flag = True
                    break
                i += 1
            if end_flag:
                break
        for candidate in candidate_object:
            i = 0
            end_flag = False
            for property_ in propertyNameList:
                if candidate in property_:
                    object_idx = i
                    end_flag = True
                    break
                i += 1
            if end_flag:
                break

        if subject_idx + object_idx == 0:
            return []

        subject_list = self.getColAt(subject_idx)
        object_list = self.getColAt(object_idx)

        # 处理(丰富)一下谓语
        temp_predicate = '作品{相关人物' + ':[' + self.prefix + ']}'
        predicate_list = [temp_predicate] * (len(subject_list) - 1)

        propertyLineNum = self.discriminatePropertyLineNum(self.getUnfoldDirection())
        extendedPropertyList = FileIO.readJson(path3)
        virtual_name_list = [''] * len(extendedPropertyList)
        idx = 0
        for i in range(len(extendedPropertyList)):
            if extendedPropertyList[i] == '':
                idx = i + 1
            else:
                virtual_name_list[i] = extendedPropertyList[idx]
        j = 0
        for property_ in propertyNameList:
            k = 0
            for exProperty in extendedPropertyList:
                if exProperty in property_ and exProperty != '':  # 是可以作为属性的谓语
                    ex_col = self.getColAt(j)
                    for i in range(propertyLineNum, self.rowNumber):
                        t = ex_col[i].content
                        if t.isspace() or t == '':
                            continue
                        single_predicate = predicate_list[i - propertyLineNum].strip('}')
                        single_predicate += ';' + virtual_name_list[k] + ":['" + t + "']}"
                        predicate_list[i - propertyLineNum] = single_predicate
                k += 1
            j += 1

        for i in range(propertyLineNum, len(subject_list)):
            subject = subject_list[i].content
            if subject == '' or subject.isspace():  # 饰演者中有空格
                continue
            subject = subject.split('、')[0]
            single_relation = [subject, predicate_list[i - 1]]
            object_ = object_list[i].content
            if '<<' in object_ or '《' in object_:
                pattern = re.compile(r'[(<<)《].+[(>>)》]')
                span_ = re.search(pattern, object_).regs[0]
                object_ = object_[span_[0] : span_[1]]
            single_relation.append(object_)
            relationship.append(single_relation)

        return relationship

    # 标题关系表
    def extractCaptionRelationship(self):
        """
        从表格中抽取标题关系
        :return:标题关系列表
        """

        path = r'configuration/relationshipConf/captionExtend.json'

        relationship = []
        subject = self.prefix
        predicate = self.name
        if predicate == '未命名表格':
            return relationship

        propertyNameList = self.getPropertyList(isPropertyName=True)  # 获取属性行
        column = self.getPersonColList(getName=False, removeHeader=True)  # 获取主列

        object = []
        linking_entity = []

        relationship.append(subject)

        # 丰富一下谓语
        predicateList = [predicate] * len(column)
        propertyLineNum = self.discriminatePropertyLineNum(self.getUnfoldDirection())
        extendedPropertyList = FileIO.readJson(path)
        virtual_name_list = [''] * len(extendedPropertyList)
        idx = 0
        for i in range(len(extendedPropertyList)):
            if extendedPropertyList[i] == '':
                idx = i + 1
            else:
                virtual_name_list[i] = extendedPropertyList[idx]
        j = 0
        first_flag = True
        for property in propertyNameList:
            k = 0
            for exProperty in extendedPropertyList:
                if exProperty in property and exProperty != '':  # 是可以作为属性的谓语
                    ex_col = self.getColAt(j)
                    for i in range(propertyLineNum, self.rowNumber):
                        t = ex_col[i].content
                        if t.isspace() or t == '':
                            continue
                        if first_flag:
                            single_predicate = predicate + "{" + virtual_name_list[k] + ":['" + t + "']}"
                        else:
                            single_predicate = predicateList[i - propertyLineNum].strip('}')
                            single_predicate += ';' + virtual_name_list[k] + ":['" + t + "']}"
                        predicateList[i - propertyLineNum] = single_predicate
                    if first_flag:
                        first_flag = False
                k += 1
            j += 1

        relationship.append(predicateList)

        # 删除括号
        for i in range(len(column)):
            single_object = re.sub(u"\[.*\]|\(.*\)|\<.*\>|（.*）|【.*】|\{.*\}|（.*\)|\(.*）", "", column[i].content)
            object.append(single_object)
            if column[i].href.__contains__(column[i].content):
                linking_entity.append(column[i].href[column[i].content])
            else:
                linking_entity.append('null')

        # 这步是去空格(但英文间的空格不去)
        idx_list = []
        for i in range(len(object)):
            if object[i].isspace():
                if i + 1 < len(object) and object[i + 1].isalpha():
                    continue
                else:
                    idx_list.append(i)
        for i in reversed(range(len(idx_list))):
            object = object[0: i] + object[i + 1:]

        relationship.append(object)
        relationship.append(linking_entity)

        return relationship

    def extractEntityRelationship(self):
        relationshsip = [self.prefix]
        _key = []
        _value = []
        _href = []

        key_list = self.cell[0]
        value_list = self.cell[1]

        if self.unfoldDirection == 'ENTITY':  # 因为每格都是键值对 所以特殊处理一下
            path = r'configuration/relationshipConf/entityProperty.json'
            entityPropertyList = FileIO.readJson(path)
            key_list = []
            value_list = []
            _dict = {}  # 用来辅助去除重复的
            pattern = re.compile('[:：]')
            # 使用高频词侦测法得到的表格的模式会比较多样

            Con  = self.cell[0][0].content.strip('\u3000\u0200\xa0')
            Con1 = self.cell[0][1].content.strip('\u3000\u0200\xa0')
            Con2 = self.cell[1][0].content.strip('\u3000\u0200\xa0')
            if re.search(pattern, Con):  # 当前格子有冒号
                if re.search(pattern, Con1) and re.search(pattern, Con2): # 每格都有冒号
                    for row in self.cell:
                        for col in row:
                            if _dict.__contains__(col.content):
                                continue
                            else:
                                if not (':' in col.content or '：' in col.content):  # 有不规整的情况 有的格子就没有引号
                                    continue
                                key_list.append(col)
                                value_list.append(col)
                                _dict[col.content] = 1

            L = [[0 for i in range(self.colNumber)] for j in range(self.rowNumber)]
            for i in range(self.rowNumber):
                for j in range(self.colNumber):
                    content = self.cell[i][j].content
                    for entity in entityPropertyList:
                        if entity in content:
                            L[i][j] = 1
                            break
            row_flag = False
            col_flag = False
            for i in range(self.rowNumber - 1):
                for j in range(self.colNumber - 1):
                    if L[i][j] and L[i][j + 1]:
                        row_flag = True
                    if L[i][j] and L[i + 1][j]:
                        col_flag = True

            if row_flag and not col_flag:
                for i in range(self.rowNumber):
                    for j in range(self.colNumber):
                        if i % 2 == 0:
                            key_list.append(self.cell[i][j])
                        else:
                            value_list.append(self.cell[i][j])
            elif col_flag and not row_flag:
                for j in range(self.colNumber):
                    for i in range(self.rowNumber):
                        if j % 2 == 0:
                            key_list.append(self.cell[i][j])
                        else:
                            value_list.append(self.cell[i][j])
            else:
                for i in range(self.rowNumber - 1):
                    for j in range(self.colNumber - 1):
                        if L[i][j]:
                            key_list.append(self.cell[i][j])
                            if L[i][j + 1] and not L[i + 1][j]:
                                value_list.append(self.cell[i + 1][j])
                            elif L[i + 1][j] and not L[i][j + 1]:
                                value_list.append(self.cell[i][j + 1])

        for j in range(len(value_list)):
            content = value_list[j].content
            value_split = [content]
            if value_list[j].multi:
                content = value_list[j].key_content
                value_split = content.split('_;_')  # 分割

            for value in value_split:
                if value.isspace() or value == '':
                    continue

                if '《' in value:
                    colon = [value]
                else:
                    dot_flag = False
                    for i in range(len(value)):  # 这步是暂时没想到好方法防止1.70米被拆成1
                        if value[i] == '.':
                            if i + 1 < len(value) and value[i + 1].isdigit():
                                dot_flag = True
                    if dot_flag:
                        pattern = re.compile('[，。,(（]')
                    else:
                        pattern = re.compile('[，。,.(（]')
                    _main = re.split(pattern, value)[0]  # 把逗号之类符号后面的东西干掉
                    colon = re.split('[:：]', _main)

                if len(colon) > 1:  # value格中存在冒号
                    if colon[-1].strip(' ') != '':  # 有些垃圾表格的单元格中有冒号但是冒号后面没值的
                        _value.append(colon[-1])
                        _key.append(colon[-2].replace(' ', '').replace('\u3000', ''))
                    else:
                        continue
                else:
                    if key_list[j].content.strip(':：') == '':  # 有的表格的键cell是空的
                        continue
                    temp_key = re.split('[:：]', key_list[j].content)  # 这步是针对六道毅词条中的一条表的操作
                    _key.append(temp_key[0].strip(':：').replace(' ', '').replace('\u3000', ''))
                    _value.append(colon[0])

                if value_list[j].href.__contains__(value_list[j].content) and not value_list[j].multi:
                    _href.append(value_list[j].href[value_list[j].content])
                else:
                    _href.append('null')

        relationshsip.append(_key)
        relationshsip.append(_value)
        relationshsip.append(_href)
        return relationshsip

    # 获奖关系表
    def extractRewardRelationship(self):
        """
        从获奖关系表格中抽取关系
        :return:标题关系列表
        """
        relationship = []
        pattern = re.compile('\d{4}(-\d+)*(-\d+)*')  # 匹配年月日
        path = r'configuration/relationshipConf/rewardExtended.json'
        extended_flag = False
        if self.prefix:
            if self.name:
                predicate = self.name
                extended_flag = False

            subject = self.prefix

            # 宾语的处理稍显复杂
            # 有的奖项莫名时间不一定在第一列(如: Red Velvet) 所以要查查看
            # 有的奖项干脆连时间都不放
            time_idx = 0
            for item in self.cell:
                if re.match(pattern, item[0].content):
                    break
                time_idx += 1
            rewards = []
            time_ = []
            if time_idx + 1 < len(self.cell):
                for item in self.cell[time_idx]:
                    time_.append(item.content)
                for item in self.cell[time_idx + 1]:
                    rewards.append(item.content)
            elif time_idx == len(self.cell):
                idx = 0
                for row in self.cell:
                    j = 0
                    for item in row:
                        if item.content != '' or not item.content.isspace():
                            idx = j
                            break
                    j += 1
                for item in self.cell[idx]:
                    rewards.append(item.content)
            else:
                for item in self.cell[time_idx]:
                    content = item.content
                    mid = re.search(pattern, content)
                    mid = mid.regs[0][1]
                    rewards.append(content[mid:])
                    time_.append(content[:mid])

            if self.name:
                extended_flag = True
                normal_title_list = FileIO.readJson(path)
                for nt in normal_title_list:
                    if nt in predicate:
                        extended_flag = False

                if '其他' in predicate:
                    extended_flag = False

                temp_predicate = predicate.split('/')[-1]  # 亚洲/亚太音乐奖
                for reward in rewards:
                    if temp_predicate in reward:
                        extended_flag = False
                        break

            for j in range(len(self.cell[-1])):
                one_relationship = [subject]
                content = rewards[j]
                predicate = '获奖'
                if '（获奖）' in content:
                    predicate = '获奖'
                elif '（提名）' in content:
                    predicate = '提名'

                one_relationship.append(predicate)

                content = content.strip('\xa0\u3000\u0020')
                pattern = re.compile('[\xa0\u3000]')  # 奖项名中或许会含有\u0020这种空格
                object = re.split(pattern, content)[0]
                object = re.sub('\u0020', '', object)  # 现在再把这\u00020消掉
                pattern = re.compile(u"\[.*\]|\(.*\)|\<.*\>|（.*）|【.*】|\{.*\}|（.*\)|\(.*）")
                object = re.sub(pattern, '', object)

                if extended_flag:
                    object = self.name + object
                one_relationship.append(object)
                if time_idx != len(self.cell):
                    one_relationship.append(time_[j])
                else:
                    one_relationship.append("null")
                relationship.append(one_relationship)

        return relationship

    def extractEntity(self, getEntityTriad=False):
        """
        从表格中抽取实体
        :return:实体列表
        """
        entity = []
        relationship = []
        if getEntityTriad:
            pass
        else:
            personNameList = self.getPersonColList(getName=False, removeHeader=True)  # 获取非纯净版主列 且去头
            if len(personNameList) == 0:
                return entity

            back_up_table = deepcopy(self)
            if back_up_table.__isNormal and back_up_table.__isCorrect:
                personNameIndex = back_up_table.__getPersonNameIndex()
                if personNameIndex != -1:
                    pivot_column = back_up_table.deleteOneCol(personNameIndex)
                else:
                    raise Exception("主列不存在")
                direction = back_up_table.getUnfoldDirection()
                lineNum = back_up_table.discriminatePropertyLineNum(direction)
                if lineNum >= 1:
                    heads = [str(head.content) for head in back_up_table.getRowAt(lineNum - 1)]
                    for i in range(lineNum, back_up_table.rowNumber):
                        for j in range(back_up_table.colNumber):
                            jsonStr = []
                            item = back_up_table.cell[i][j]
                            if str(item.content).isspace() or len(str(item.content)) == 0:
                                continue
                            jsonStr.extend(pivot_column[i])
                            jsonStr.append(heads[j])
                            jsonStr.append(str(item.content))
                            if item.href.__contains__(item.content):
                                jsonStr.append(item.href[item.content])
                            else:
                                jsonStr.append('null')
                            relationship.append(jsonStr)
                    return relationship
            else:
                raise Exception("该表格不规范，无法写成json串")

        return relationship

    def __getPersonNameIndex(self):
        """
        返回人名所在的列的索引
        :return:人名所在的列的索引
        """
        if self.personNameIndex != -1:
            return self.personNameIndex

    def clearTable(self):
        """
        清理表格，去除表格中无意义的序号，去除空行或者空列
        :return:无
        """
        propertyList = self.getPropertyList(isPropertyName=True)
        # 清除带有“序”的属性行
        clearSet = ["序号", "序"]
        indexes = [index for index, propertyName in enumerate(propertyList) if propertyName in clearSet]
        if indexes:
            if self.getUnfoldDirection() == "ROW":
                self.deleteOneCol(indexes[0])
            else:
                self.deleteOneRow(indexes[0])
            self.getAbsolutePosition()
        # 如果第一行内容为空，则删除第一行
        canContinue = True
        for item in self.getRowAt(0):
            if not ((item.content.isnumeric() and len(str(item.content)) == 1) or str(item.content).isspace()):
                canContinue = False  # 只要一格不为空
            if not canContinue:
                break
        if canContinue:
            self.deleteOneRow(0)
        # 如果最后一行是参考资料或者为空，删除这一行
        canContinue = True
        for item in self.getRowAt(self.rowNumber - 1):
            if "参考资料" in str(item.content):
                canContinue = True
                break
            if not str(item.content).isspace():  # 当前格子为空
                canContinue = False
                break

        if canContinue:
            self.deleteOneRow(self.rowNumber - 1)

        # 将表格中的单纯的符号单元转化为空格
        for i in range(self.rowNumber):
            for j in range(self.colNumber):
                if self.cell[i][j].getTableItemType() == "标点类型":
                    self.cell[i][j].content = ""

        # 有时候整列都直接为空 则将整列抬走
        for j in reversed(range(self.colNumber)):
            blank_count = 0
            for i in range(self.rowNumber):
                if self.cell[i][j].content == "" or self.cell[i][j].content.isspace():
                    blank_count += 1
            if blank_count / (self.rowNumber - 1) > 0.8:
                self.deleteOneCol(j)
                self.deleted = True

    # 关系表持久化1
    def persistence1(self, relationship: list, path_out):
        path_out = path_out + '_1' + '.csv'
        out = open(path_out, 'w', newline='', encoding='utf8')
        csv_write = csv.writer(out, dialect='excel')
        record = [];
        record.append(relationship[0])  # 主语

        record_count = 0
        for i in range(len(relationship[2])):
            record.append(relationship[1][i])  # 谓语
            record.append(relationship[2][i])
            if relationship[2][i].isspace() != True and relationship[2][i] != '':  # 要是不为空就写入
                csv_write.writerow(record)
                csv_write.writerow([])
                record_count += 1
            record.pop()
            record.pop()
        out.close()
        return record_count

    def persistence2(self, relationship: list, path_out):
        path_out = path_out + '_2' + '.csv'
        if not len(relationship):
            return
        out = open(path_out, 'w', newline='', encoding='utf8')
        csv_write = csv.writer(out, dialect='excel')
        record_count = 0
        for row in relationship:
            record = [];
            for col in row:
                record.append(col)
            csv_write.writerow(record)
            csv_write.writerow([])
            record_count += 1
        out.close()
        return record_count

    def persistence3(self, relationship: list, path_out):
        path_out = path_out + '_3' + '.csv'
        if not len(relationship):
            return
        out = open(path_out, 'w', newline='', encoding='utf8')
        csv_write = csv.writer(out, dialect='excel')
        record_count = 0
        for row in relationship:
            record = []
            record.extend(row)
            csv_write.writerow(record)
            csv_write.writerow([])
            record_count += 1
        out.close()
        return record_count

    # 实体关系表
    def persistence4(self, relationship: list, path_out):
        path_out = path_out + '_4' + '.csv'
        out = open(path_out, 'w', newline='', encoding='utf8')
        csv_write = csv.writer(out, dialect='excel')
        record_count = 0
        for i in range(len(relationship[1])):
            record = [];
            if len(relationship[1][i]) > 10 or len(relationship[2][i]) > 15:
                continue
            record.append(relationship[0])  # 主语
            record.append(relationship[1][i])  # 谓语
            record.append(relationship[2][i])  # 宾语
            csv_write.writerow(record)
            csv_write.writerow([])
            record_count += 1
        out.close()
        return record_count

    # 获奖记录
    def persistence5(self, relationship: list, path_out):
        path_out = path_out + '_5' + '.csv'
        out = open(path_out, 'w', newline='', encoding='utf8')
        csv_write = csv.writer(out, dialect='excel')
        record_count = 0
        for row in relationship:
            record = [];
            record.append(row[0])
            predicate = '获奖'
            if row[1] == '提名':
                predicate = '提名'

            # 需要把时间作为谓语的属性, 格式为 获奖{时间:['2009']}
            t = row[3]
            if t != 'null':
                predicate += '{时间:[\'' + t + '\']}'
            record.append(predicate)
            record.append(row[2])
            csv_write.writerow(record)
            csv_write.writerow([])
            record_count += 1
        out.close()
        return record_count


class TypeTree:
    """
    类型树类
    """

    def __init__(self):
        """
        初始化类型树
        """
        tree = Tree()
        tree.create_node(tag="类型", identifier="类型")
        tree.create_node(tag="超链接", identifier="超链接", parent="类型")
        tree.create_node(tag="图片", identifier="图片", parent="类型")
        tree.create_node(tag="字符和数字", identifier="字符和数字", parent="类型")
        tree.create_node(tag="其他类型", identifier="其他类型", parent="类型")
        tree.create_node(tag="标点类型", identifier="标点类型", parent="类型")
        tree.create_node(tag="字符类型", identifier="字符类型", parent="字符和数字")
        tree.create_node(tag="数字类型", identifier="数字类型", parent="字符和数字")
        tree.create_node(tag="中文", identifier="中文", parent="字符类型")
        tree.create_node(tag="英文", identifier="英文", parent="字符类型")
        tree.create_node(tag="<=0", identifier="<=0", parent="数字类型")
        tree.create_node(tag="0-1", identifier="0-1", parent="数字类型")
        tree.create_node(tag=">=1", identifier=">=1", parent="数字类型")
        tree.create_node(tag="大写", identifier="大写", parent="英文")
        tree.create_node(tag="小写", identifier="小写", parent="英文")
        tree.create_node(tag="大小写混合", identifier="大小写混合", parent="英文")
        # tree.show()
        self.tree = tree

    def getTypeCharacter(self, table: Table):
        """
        计算表格的行类型特征和列类型特征
        :param table:传入的表格
        :return:行类型特征rowTypeCharacter 和 列类型特征colTypeCharacter
        """
        row_bias = col_bias = 1
        if table.rowNumber < 3:
            row_bias = 0
        if table.colNumber < 3:
            col_bias = 0

        rowTypeCharacter = 0
        colTypeCharacter = 0
        rowTypeCharacterList = []
        colTypeCharacterList = []
        for i in range(row_bias, table.rowNumber - 1):
            row_at1 = table.getRowAt(i)
            row_at2 = table.getRowAt(table.rowNumber - 1)
            colTypeCharacterList.append(self.VType(row_at1, row_at2))
        if colTypeCharacterList:
            colTypeCharacter = np.mean(colTypeCharacterList)

        for j in range(col_bias, table.colNumber - 1):
            rowTypeCharacterList.append(self.VType(table.getColAt(j), table.getColAt(table.colNumber - 1)))
        if rowTypeCharacterList:
            rowTypeCharacter = np.mean(rowTypeCharacterList)
        sumNumber = rowTypeCharacter + colTypeCharacter
        if sumNumber == 0:
            return rowTypeCharacter, colTypeCharacter
        return rowTypeCharacter / sumNumber, colTypeCharacter / sumNumber

    def _VType(self, item1: TableItem, item2: TableItem) -> int:
        """
        计算两个表格单元之间的类型差异距离
        :param item1: 表格单元1
        :param item2: 表格单元2
        :return: 类型差异距离
        """
        distance = 0
        typeNode1 = item1.type_
        typeNode2 = item2.type_
        if typeNode1 is None or typeNode2 is None:
            raise Exception("当前类型为None，无法计算出类型之间的距离")
        level1 = self.tree.depth(typeNode1)
        level2 = self.tree.depth(typeNode2)
        if typeNode1 == typeNode2:
            return distance
        if level1 > level2:
            while level1 != level2:
                typeNode1 = self.tree.parent(typeNode1).identifier
                distance += 1
                level1 -= 1
        elif level2 > level1:
            while level1 != level2:
                typeNode2 = self.tree.parent(typeNode2).identifier
                distance += 1
                level2 -= 1
        if level1 == level2:
            while typeNode1 != typeNode2:
                typeNode1 = self.tree.parent(typeNode1).identifier
                typeNode2 = self.tree.parent(typeNode2).identifier
                distance += 2
        return distance

    def VType(self, v1: list, v2: list) -> float:
        """
        计算两个列表之间的类型差异
        :param v1:列表1
        :param v2:列表2
        :return:类型差异值
        """
        res = 0
        len1 = len(v1)
        len2 = len(v2)
        if len1 == 0 or len2 == 0:
            return res
        m = min(len1, len2)
        for i in range(m):
            res += self._VType(v1[i], v2[i])
        return res / m


def changeTig2Table(tag: Tag, caption='未命名表格', prefix=None) -> Table:
    """
    将tag标签转化为table数据结构
    :param prefix: 前缀
    :param caption:标题
    :param tag: 输入的标签
    :return: 返回表格
    """

    def changeTag2TableItem(tag: Tag, rowIndex: int, colIndex: int):
        """
        把标签转化为单元格
        :param tag: 带转化标签
        :param rowIndex: 单元格的行索引
        :param colIndex: 单元格的列索引
        :return: 单元格
        """
        rowspan = colspan = 1
        # 获取表格单元中的超链接
        href = {}
        aList = tag.find_all("a")
        count = 0;
        for a in aList:
            if a.has_attr("href"):
                count += 1;
                href[a.text] = r"https://baike.baidu.com" + a["href"]

        # 获取表格单元中的图片
        imgSrc = []
        imgList = tag.find_all("img")
        for img in imgList:
            tableItem = TableItem()
            tableItem.hasPic = True
            return tableItem  # 有图片的表格直接判死刑

        divFlag = False
        divList = tag.find_all('div')
        if len(divList) > 1:
            divFlag = True
            text = ''
            blank_count = 0
            for div in divList:
                if div.text.isspace() or div.text == '':
                    blank_count += 1
                    continue
                single_text = div.text + '_;_'
                text += single_text
            if blank_count == len(divList) - 1:
                divFlag = False
        else:
            text = tag.text

        text = re.sub('(\[)\d+(\])', '', text)  # 去除索引注释，例如 [12]
        content = text.replace("\xa0", "")  # 去掉空白符

        text2 = re.sub('(\[)\d+(\])', '', tag.text)  # 去除索引注释，例如 [12]
        content2 = text2.replace("\xa0", "")  # 去掉空白符

        # 获取表格的占据行列数
        if tag.has_attr("rowspan"):
            rowspan = int(tag['rowspan'])
        if tag.has_attr("colspan"):
            colspan = int(tag['colspan'])

        tagName = tag.name
        tableItem = TableItem(content2, rowIndex, rowspan, colIndex, colspan, href, imgSrc, tagName=tagName)

        pattern = re.compile('(\[)\d+(\])')  # 匹配索引注释的模式 如[12]
        if re.search(pattern, tag.text):  # 如果含有了注释
            tableItem.hasIndex = True
        if len(tag.find_all("b")) != 0:
            tableItem.particular['b'] = 1
        if divFlag:
            tableItem.key_content = content
            tableItem.multi = True

        return tableItem

    def finalDeal(table: Table, colLenList: list, rowNumber: int):
        """
        最终处理，该步骤将表格重新初始化，并且重新计算绝对位置，判断表格类型
        :param table: 表格名
        :param colLenList: 列长度列表
        :param rowNumber: 行数
        :return:
        """
        table.colNumber = max(colLenList)
        table.rowNumber = rowNumber
        table.getAbsolutePosition()
        table.initialNormal()  # 判断是否正常
        table.initialCorrect()  # 判断是否正确
        table.initialTableItemsType()  # 初始化表格单元的类型

    table = Table()  # 默认构造器(行数, 列数, 表格名字, 二维数组, 展开方向)
    table.cell = []
    colLenList = []
    table.name = str(caption)  # 命名
    table.prefix = prefix

    thead = tag.find("thead")
    tbody = tag.find("tbody")

    rowIndex = 0
    if thead:  # 如果存在<thead> 则th应该是存在的
        table.unfoldDirection = 'ROW'
        for row in thead.children:
            colIndex = 0
            colSize = 0
            innerList = []
            for colData in row.children:
                tableItem = changeTag2TableItem(colData, rowIndex, colIndex)
                colIndex += 1
                innerList.append(tableItem)
                colSize += tableItem.colspan
            colLenList.append(colSize)
            table.cell.append(innerList)
            rowIndex += 1

        if tbody:
            for row in tbody.children:
                if row.attrs.__contains__('class'):
                    if 'cellModule-pager' in row.attrs['class']:
                        continue
                colIndex = 0
                colSize = 0
                innerList = []
                for colData in row.children:
                    tableItem = changeTag2TableItem(colData, rowIndex, colIndex)
                    colIndex += 1
                    innerList.append(tableItem)
                    colSize += tableItem.colspan
                rowIndex += 1
                colLenList.append(colSize)
                table.cell.append(innerList)

        else:
            for rowData in tag.children:
                if rowData.name == 'thead':
                    continue
                if rowData.attrs.__contains__('class'):
                    if 'cellModule-pager' in rowData.attrs['class']:
                        continue
                colIndex = 0
                colSize = 0
                innerList = []
                for colData in rowData.children:
                    if isinstance(colData, NavigableString):
                        continue
                    tableItem = changeTag2TableItem(colData, rowIndex, colIndex)
                    colIndex += 1
                    colSize += tableItem.colspan
                    innerList.append(tableItem)
                colLenList.append(colSize)
                table.cell.append(innerList)
                rowIndex += 1
    elif tbody:  # 有些表会把属性也放在tbody里
        for rowData in tbody:
            colIndex = 0
            colSize = 0
            innerList = []
            for colData in rowData.children:
                if isinstance(colData, NavigableString):
                    continue
                tableItem = changeTag2TableItem(colData, rowIndex, colIndex)
                colIndex += 1
                colSize += tableItem.colspan
                innerList.append(tableItem)
            colLenList.append(colSize)
            table.cell.append(innerList)
            rowIndex += 1
    else:
        for rowData in tag.children:
            if rowData.attrs.__contains__('class'):
                if 'cellModule-pager' in rowData.attrs['class']:
                    continue
            colIndex = 0
            colSize = 0
            innerList = []
            for colData in rowData.children:
                if isinstance(colData, NavigableString):
                    continue
                tableItem = changeTag2TableItem(colData, rowIndex, colIndex)
                colIndex += 1
                colSize += tableItem.colspan
                innerList.append(tableItem)
            colLenList.append(colSize)
            table.cell.append(innerList)
            rowIndex += 1
    finalDeal(table, colLenList, rowIndex)
    return table
