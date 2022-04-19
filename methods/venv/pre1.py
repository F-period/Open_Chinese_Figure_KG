from bs4 import BeautifulSoup
from bs4.dammit import encoding_res
from bs4.element import Tag, NavigableString, Comment
from tableExtract.table import *
from tableExtract.TableItem import *
from copy import deepcopy
from queue import Queue
from urllib.parse import unquote
from pyhanlp import HanLP  # 使用前导入 HanLP工具
from FileIO import FileIO
from tableExtract.table import changeTig2Table, Table, TableItem

import json
import os
import random
import re
import time
import requests
import string
import jieba
# 这里是在正式抽取前对表格的一些预处理操作

# 用来将html页面中的空白符去掉 方便之后的操作
def htmlPreTreat(_html: str):
    """
    html预处理，包括去除注释、脚本、文章、代码，返回标准化的html和soup
    :param _html: 输入的html串
    :return: _html为返回的格式化的字符串，_soup为返回的soup
    """
    _html = _html.replace("\r", "").replace("\t", "").replace("\n", "")
    _soup = BeautifulSoup(_html, 'lxml')
    # 去除注释
    [comment.extract() for comment in _soup.findAll(text=lambda text: isinstance(text, Comment))]
    # 去除脚本
    [script.extract() for script in _soup.findAll('script')]
    [style.extract() for style in _soup.findAll('style')]
    # 去除文章
    [article.extract() for article in _soup.find_all('article')]
    # 去除代码
    [code.extract() for code in _soup.find_all('code')]
    # 格式化
    return _html, _soup


# 用来判断一个表格是否规范
def throughHeuristicRule(table: Tag):
    """
    判断是否通过启发式规则 具体讲解在ppt 11页
    :return:
    """

    def _Rule1(tag) -> bool:
        """
        1如果实体信息存在于有<TABLE>标签的表格之中，即<table></table>标签之中，
        那么通常此表格表现为多行多列(大于两行两列）的结构
        :param tag:标签
        :return:，满足该规则记为True，不满足该规则记为False;
        """
        #
        if len(tag.contents) >= 2:
            now = tag.next_element
            if now.name == "caption":  # 指页面中本身的标签名字就是<caption>
                now = now.nextSibling
            if len(now.contents) == 1:
                now = now.nextSibling
            if now == None:
                return False
            if len(now.contents) >= 2:
                return True
            if len(now.next.contents) >= 2:
                return True
        tag = tag.contents[0]

        if len(tag.contents) >= 2:
            now = tag.next_element
            if now.name == "caption":  # 指页面中本身的标签名字就是<caption>
                now = now.nextSibling
            if len(now.contents) == 1:
                now = now.nextSibling
            if now == None:
                return False
            if len(now.contents) >= 2:
                return True
            if len(now.next.contents) >= 2:
                return True

        else:
            return False
        return False

    def _Rule2(tag) -> bool:
        """
        2无论对于有<TABLE>标签的表格还是无<TABLE>标签的表格来说，
        其实体信息所在区域中不会包含大量的超链接、表单、图片、脚本或者嵌套的表格，记为b，不满足该规则记为b_;
        :return:
        """
        # 获取表格中所有超链接
        hrefs = [a['href'] for a in tag.find_all('a', href=True)]
        # 获取子表单
        tables = tag.find_all('table')
        sub_table = []
        for table in tables:
            if type(table) == Tag:
                if type(table.descendants) == Tag:
                    sub_table.append(table.descendants.find_all('table'))
        scripts = tag.find_all('script')
        img = tag.find_all('img')
        thead = tag.find("thead")
        tbody = tag.find("tbody")
        if thead and tbody:
            rows = len(thead.contents) + len(tbody.contents)
            cols = len(thead.next.contents)  # thead的next是tr或th 所以就是统计第一行有多少个td
        else:
            if tbody:
                tag = tbody
            ul = tag.find('ul')
            if ul:
                rows = len(ul.contents) + 1
                cols = len(ul.contents) - 1
            else:
                rows = len(tag.contents)
                now = tag.contents[0]
                if now.name == "caption":
                    now = now.nextSibling
                cols = len(now.contents)

        if len(hrefs) > rows * cols * 2 or len(sub_table) > 3 or len(scripts) > 1 or len(img) >= 1:
            return False
        else:
            return True

    def _Rule3(tag) -> bool:
        """
        3属性名一般出现在前几行或前几列，记为c，不满足该规则记为c_;
        :param tag:
        :return:
        """
        tagContents = tag.contents
        segment = HanLP.newSegment()
        if len(tagContents) >= 2:
            # 判断前2行、前2列是否有属性
            for tagContent in tagContents[0:2]:
                if tagContent.name == "caption":
                    continue
                contentList = tagContent.contents
                for content in contentList:
                    results = list(segment.seg(content.text))
                    natureList = [str(result.nature) for result in results]
                    if natureList.count("n") > 0:
                        return True
                    # for result in results:
                    #     if str(result.nature) not in tableExtractor.attrTypeSet:
                    #     return False

                    # return True
            return False
        tag = tag.contents[0]
        tagContents = tag.contents
        segment = HanLP.newSegment()
        if len(tagContents) >= 2:
            # 判断前2行、前2列是否有属性
            for tagContent in tagContents[0:2]:
                if tagContent.name == "caption":
                    continue
                contentList = tagContent.contents
                for content in contentList:
                    print(tag)
                    results = list(segment.seg(content.text))
                    natureList = [str(result.nature) for result in results]
                    if natureList.count("n") > 0:
                        return True
            return False
        return False


    if _Rule1(table):
        if _Rule2(table):
            if _Rule3(table):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def extractListTable(tag: Tag, name, url) -> list:
    """
    从Tag中抽取列表
    :param tag:带抽取标签
    :return: 表格的list
    """
    tableList = []
    titleList = tag.find_all(class_='normal title-td')
    ulList = tag.find_all("ul")
    if len(titleList) == len(ulList):
        for i in range(len(titleList)):
            titleTag = titleList[i]
            dataTag = ulList[i]
            caption = titleTag.text
            liList = dataTag.contents
            cellList = []
            colSizeList = []
            rowNumber = len(liList)
            for rowIndex in range(rowNumber):
                li = liList[rowIndex]
                spans = li.find_all("span")
                colIndex = 0
                innerList = []
                for span in spans:
                    if span.text == '▪':
                        continue
                    else:
                        text = span.text.strip()
                        href = {}
                        aList = span.find_all("a")
                        if aList:
                            for a_node in aList:
                                if a_node.has_attr("href"):
                                    href[a_node.text] = a_node['href']

                        imgSrc = []
                        imgList = span.find_all("img")
                        for img in imgList:
                            if img.has_attr("src"):
                                imgSrc.append(img["src"])
                        newTableItem = TableItem(text, rowIndex, 1, colIndex, 1, href, imgSrc,
                                                 tagName=span.name)
                        colIndex += 1
                        innerList.append(newTableItem)
                    colSizeList.append(colIndex)
                cellList.append(innerList)
            if len(colSizeList) == 0:
                print("获奖关系表有异常", end=" ")
                print(name, end=' ')
                print(url)
                continue
            newTable = Table(rowNumber, max(colSizeList), caption, table=cellList)
            newTable.tableType = '获奖关系表'
            newTable.hrefMap = url
            newTable.prefix = name
            tableList.append(newTable)
    return tableList


# 得到一个表格的标题
def getCaption(_tag: Tag, previou_caption):
    """
    提取标签中的表格标题 和 标题前缀
    :param _tag:输入的标签
    """
    _caption = "未命名表格"
    temp = _tag.find(name="caption")  # 去除标题，标题不计入行列之中
    if temp:  # 直接在表格tag之内有caption
        _caption = temp.text
        [caption.extract() for caption in _tag.find_all(name="caption")]
    else:  # 大多数情况都不会直接有caption的 只能自己去找
        # 被同一个大tag包裹下的前一个tag
        # previous_element 的话是前一个tag下的最后一个element(可能是字符串NavigableString)
        _previous = _tag.previous_sibling  # 表格中没有caption 只有往前找一次
        if _previous:
            title = _previous.find(attrs={"class": re.compile(r"^.*title.*$")}) # 找当前tag下有的含title的class
            if title:  # 可信度比较高
                if len(title.contents) == 2:
                    _caption = title.contents[1]
                elif len(title.contents) == 1:
                    if isinstance(title.contents[0], NavigableString):
                        _caption = title.contents[0]
                    else:
                        _caption = title.contents[0].text
                    return str(_caption), True
            else:  # 不太可靠的
                if len(_previous.contents) == 1:
                    if isinstance(_previous.contents[0], NavigableString):
                        quasi_caption = _previous.contents[0]
                        pattern = re.compile(r'[a-zA-Z0-9.-]')
                        temp = re.sub(pattern, '', quasi_caption)
                        if 0 < len(temp) < 10:  # 标题除去数字英文之后不应过长
                            _caption = quasi_caption
                    elif _previous.contents[0].name == 'b':
                        quasi_caption = _previous.contents[0].text
                        pattern = re.compile(r'[a-zA-Z0-9.-]')
                        temp = re.sub(pattern, '', quasi_caption)
                        if 0 < len(temp) < 10:  # 标题除去数字英文之后不应过长
                            _caption = quasi_caption

            _previous = _previous.previous_sibling  # 为了以防万一识别错误了 再往前面找一次

            if _previous.name == 'table' and previou_caption != '':
                return previou_caption, True

            if _previous:
                title = _previous.find(attrs={"class": re.compile(r"^.*title.*$")})
                if title:
                    if len(title.contents) == 2:
                        _caption = title.contents[1]
                    elif len(title.contents) == 1:
                        if isinstance(title.contents[0], NavigableString):
                            _caption = title.contents[0]
                        else:
                            _caption = title.contents[0].text

    _exception = '注'  # 有些表格会有一行字 以"注: xxx" 开头

    if _caption == "未命名表格" or _exception in _caption:  # 有的表格布局比较奇怪 标题和表格中间莫名有若干不可见的留白tag
        _previous = _tag
        for i in range(0, 7):  # 比较耗时的向前的7次探寻
            if _previous.previous_sibling != None:
                _previous = _previous.previous_sibling
            else:
                _previous = _previous.previous_element
            if _previous.attrs.__contains__('class'):
                if _previous.attrs['class'][0] == 'para-title':
                    _caption = _previous.contents[0].contents[1]
                    return str(_caption), True
            elif _previous.name == 'h3' or _previous.name == 'h2':
                _caption = _previous.text
                return str(_caption), True

    else:
        return str(_caption), True
    return str(_caption), False


def getTable(_html: str, url):
    """
    表格定位，从html中抽取表格，并返回一个list，该list中每个元素都是一个Table对象
    :param _html:待抽取的html串
    :return:一个list，该list中每个元素都是一个Table对象
    """
    flag = 0
    _tableList = []  # 用来存储最后要返回的table
    _html = _html.decode()
    _html, _soup = htmlPreTreat(_html)   # 把html页面中的空白符去掉 更容易之后的找caption
    if _soup.find('title') is None:
        return _tableList, flag
    else:
        name = _soup.find('title').string.strip('_百度百科')  # 百科的题目--瓦尔特P38手枪 （军事武器枪械）
    tagTable = _soup.find_all(name="table")  # 找到所有的Table标签

    # tagTable就是当前页面的所有table表格, tag就是其中一张表格
    bias = 0
    previous_caption = ''
    for tag in tagTable:
        flag += 1
        if throughHeuristicRule(tag):  # 看是否能通过启发式规则
            caption, certain_flag = getCaption(tag, previous_caption)  # 获取表格的标题
            if '专辑曲目（' in caption:  # 有一群表格的标题是这个 然后很不规整
                continue
            if certain_flag:  # 如果当前所获得的标题非常可靠 那么有些时候也可能是下一个表格的标题
                previous_caption = caption
            else:
                previous_caption = ''
            if caption == '获奖记录':
                tableList = extractListTable(tag, name, url)
                if len(tableList) > 0:
                    _tableList.extend(tableList)  # 用extend是因为某些明星获奖记录有若干张
                bias = len(tableList)
                continue

            occupation_tag = None
            occupation_list = []
            candi_tags = _soup.find_all('dt', class_ = ['basicInfo-item', 'name'])
            pattern = re.compile('[\xa0\u3000\u0020]')
            for candi in candi_tags:
                if re.sub(pattern, '', candi.text) == '职业':
                    occupation_tag = candi
                    break
            if occupation_tag:
                occupation = occupation_tag.next_sibling.text
                pattern = re.compile('[,，、 ]')
                occupation_list = re.split(pattern, occupation)

            aTable = changeTig2Table(tag, caption)  # 将tag转化为Table对象
            aTable.hrefMap = url
            aTable.prefix = name
            if len(occupation_list):
                aTable.occupation = occupation_list

            # 这里代码结构没写好 只能放在这里将有图片的web表格丢弃了
            pic_flag = False
            for row in aTable.cell:
                for item in row:
                    if item.hasPic:
                        pic_flag = True
            if pic_flag:
                flag -= 1
                continue

            _tableList.append(aTable)
    return _tableList, flag + bias


def dealWithTableList(table):
    """
    :param _tableList: Table的列表，该列表中每个元素都是一个Table对象
    :return:
    """
    table = table.extendTable()  # 表格规整 主要就是把rowspan等处理一下 normal和correct的判断也在这里
    flag = False

    if table.tableType == '获奖关系表':
        table = table.flip()  # 转置
        return table, True

    if table.isNormal() and table.isCorrect():  # 判断表格是否正常且正确
        if table.getUnfoldDirection() == "COL":  # 把表格全部变成横向展开的
            table = table.flip()  # 转置
        table.clearTable()  # 清理表格数据
        flag = True

    if flag:
        if table.isStatisticTable():  # 是不是数据型表格
            flag = False
        else:  # 看属性行有无空值 在一定程度上能挡掉二维表格(二维表格的第一行第一列一般为空)
            for property_ in table.propertyNameList:
                if property_.isspace() or property_ == '':
                    flag = False
                    break

    if flag:
        table.clearPunctuation()

    return table, flag


