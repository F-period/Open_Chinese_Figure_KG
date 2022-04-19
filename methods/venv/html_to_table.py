# coding=utf-8
#@Time : 2021/9/15 22:21
import json
import time
import random
import csv
import os
import pickle
from bs4 import BeautifulSoup
from preprocess import *
import zipfile
from tableExtract.table import *
from candidate import *
from disambiguation import *
from pre1 import *
import time

# 这里是先把页面中的表格本地化

if __name__ == '__main__':

    html_path_in = r'/data2/百科人物原始页面/personPageWithTables'
    path_list = os.listdir(html_path_in)  # 读取输入文件夹
    path_out = r'/data1/baikepedia/data/tables1'
    counter = 0
    table_bound = 0
    table_retract = 0
    counter1 = 0
    counter2 = 0
    group = 3
    interval = 150
    start_flag = 0
    count = 0
    for suffix_path in path_list:
        count += 1
        print(count)
        if suffix_path != '27904484.html' and not start_flag:  # 测试用
            continue
        start_flag = 1
        cha = suffix_path.strip('.html')
        # if len(str(cha)) >= 8 :
        #     continue
        print(cha)
        # if len(str(cha)) <= 8 and count <= 310 and cha[0] == '2':
        #     if group != 0:
        #         group -= 1
        #         count += 1
        #     elif group == 0 and interval != 0:
        #         interval -= 1
        #         continue
        #     elif group == 0 and interval == 0:
        #         if random.randint(0, 9) > 4:
        #             group = 3
        #         else:
        #             interval = 150
        #         continue
        # else:
        #     continue
        url = html_path_in + r'/' + suffix_path
        with open(url, 'rb') as file:
            _tableList = []
            try:
                _tableList, has_table = getTable(file.read(), url)  # hasTable是当前页面表格数的上界
                table_bound += has_table
                table_retract += len(_tableList)
            except Exception as e:
                print("有异常", e, suffix_path)

            table_number_suffix = 0  # 用于命名 表当前是页面中第几个表格

            for _table in _tableList:
                table_number_suffix += 1
                temp_url = url.split('/')[-1]
                table_number = temp_url.strip('.html') + '_' + str(table_number_suffix)
                table_out = path_out + r'/' + table_number + '.pickle'
                file2 = open(table_out, 'wb')
                pickle.dump(_table, file2, pickle.HIGHEST_PROTOCOL)
                file2.close()
    print('完成', count)

    # path = r'/data2/百科人物原始页面/百科人物原始页面.zip'
    # path_out = r'/data1/baikepedia/data/tables2'
    # azip = zipfile.ZipFile(path)
    # has_table_number_html = 0
    # has_table_number = 0
    # recog_html = 0
    # recog_table = 0
    # print('开始')
    # exception_path = r'/data1/baikepedia/data/exception.log'
    # counter1 = 0
    # group = 3
    # interval = 500
    # start_flag = 0
    # for file in azip.namelist():
        # cha = file.split('/')[-1]
        # if len(cha) != 0:
        #     cha = cha[0]
        # else:
        #     continue
        # if cha == '1' and counter1 < 200:
        #     if group != 0:
        #         group -= 1
        #     if group == 0 and interval != 0:
        #         interval -= 1
        #         continue
        #     if group == 0 and interval == 0:
        #         if random.randint(0, 9) > 3:
        #             group = 5
        #         else:
        #             interval = 500
        #         continue
        # else:
        #     continue

    #     print(file)
    #     if start_flag:
    #         break
    #     if file != '百科人物原始页面/10611865.html' and not start_flag:
    #         continue
    #     start_flag = 1
    #
    #     url = file.split(r'/')[-1]  # 标记表格属于哪个html文件
    #     _html = azip.read(file)
    #     _tableList = []
    #     has_table = 0
    #     try:
    #         _tableList, has_table = getTable(_html, url)  # 得到某html中的所有表格
    #     except Exception as e:
    #         exception_log = open(exception_path, 'a', encoding='utf8')
    #         exception_log.writelines(e.args)
    #         exception_log.writelines('\n')
    #         exception_log.writelines(file)
    #         exception_log.writelines('\n')
    #         exception_log.writelines("实际含table标签的网页数")
    #         exception_log.writelines(str(has_table_number_html))
    #         exception_log.writelines('识别出来有html的网页数')
    #         exception_log.writelines(str(recog_html))
    #         exception_log.writelines("当前网页含有表格数: ")
    #         exception_log.writelines(str(has_table_number))
    #         exception_log.writelines("当前识别表格数: ")
    #         exception_log.writelines(str(recog_table))
    #         exception_log.writelines('\n')
    #         exception_log.close()
    #         continue
    #     if has_table != 0:
    #         has_table_number_html += 1  # 实际含table标签的网页数
    #
    #     has_table_number += has_table
    #     # print("当前网页含有表格数: ")
    #     # print(has_table_number)
    #
    #     if len(_tableList) == 0:
    #         continue
    #     recog_html += 1  # 识别出来有html的网页数
    #     recog_table += len(_tableList)
    #     # print("当前识别表格数: ")
    #     # print(recog_table)
    #     counter1 += 1
    #     table_number_suffix = 0  # 用于命名 表当前是页面中第几个表格
    #     count = 0
    #     for _table in _tableList:
    #
    #         count += 1
    #         table_number_suffix += 1
    #         # if count <= 38:
    #         #     continue
    #         table_number = url.strip('.html') + '_' + str(table_number_suffix)
    #         table_out = path_out + r'/' + table_number + '.pickle'
    #         file2 = open(table_out, 'wb')
    #         pickle.dump(_table, file2, pickle.HIGHEST_PROTOCOL)
    #         file2.close()
    # print("实际含table标签的网页数")
    # print(has_table_number_html)
    # print('识别出来有html的网页数')
    # print(recog_html)
    # print("当前网页含有表格数: ")
    # print(has_table_number)
    # print("当前识别表格数: ")
    # print(recog_table)