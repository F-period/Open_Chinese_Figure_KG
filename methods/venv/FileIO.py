import csv
import json
import pickle
import sys

from bs4 import Tag


class FileIO:
    def __init__(self):
        """
        初始化函数，无作用
        """
        pass

    @staticmethod
    def writePkl(filepath: str, data, mode='wb+') -> bool:
        """
        将数据写入到 pkl文件
        :param filepath: 写入的文件的绝对路径
        :param data: 待写入的数据，可以是任何非递归的数据结构
        :param mode: 写入的模式，默认为"wb+"，即完全覆盖已有文件
        :return: 写入成功则返回True，否则返回False
        """
        try:
            with open(filepath, mode) as f:
                pickle.dump(data, f)
                return True
        except Exception as e:
            print(f"写入pkl文件<{filepath}>失败", e)
            return False

    @staticmethod
    def readPkl(filepath: str, mode='rb+'):
        """
        从pkl文件中读取数据
        :param filepath: pkl文件绝对路径
        :param mode: 读取模式，默认为“rb+”，即读取
        :return: 返回读取到的数据
        """
        try:
            with open(filepath, mode=mode) as f:
                return pickle.load(f)
        except Exception as e:
            print(f"读取pkl文件<{filepath}>失败", e)

    @staticmethod
    def writeList2Pkl(filepath: str, dataList: list):
        """
        将数据列表分段写入到pkl文件中，该方法写入的pkl文件，仅能通过readPkl2List读取
        :param filepath: pkl文件的绝对路径
        :param dataList: 待写入的数据列表。
        :return: 无
        """
        for data in dataList:
            try:
                with open(filepath, 'ab') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print("列表写入pkl文件失败", e)

    @staticmethod
    def readPkl2List(filepath: str) -> list:
        """
        读取pkl文件到数据列表
        :param filepath: pkl文件的绝对路径
        :return: 读取的数据列表
        """
        resList = []
        try:
            with open(filepath, 'rb') as f:
                while True:
                    try:
                        data = pickle.load(f)
                        resList.append(data)
                    except EOFError:
                        break
        except Exception as e:
            print("读取pkl文件到列表失败", e)
        finally:
            return resList

    @staticmethod
    def writeTag2Html(filepath: str, tag: Tag):
        """
        将beautifulsoup中的Tag类标签，整个写入到html文件之中
        :param filepath: html文件的绝对路径
        :param tag: 待写入的标签
        :return: 无
        """
        with open(
                filepath,
                mode="w+", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Title</title>
            </head>
            <body>
            <table data-sort="sortDisabled" log-set-param="table_view">""")
            f.write(str(tag))
            f.write("""</table>
            </body>
            </html>
                            """)

    @staticmethod
    def readHtmlFormFile(filepath: str) -> str:
        """
        从html文件中读取html串
        :param filepath: html文件的绝对路径
        :return: html串
        """
        with open(filepath, mode='r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def write2Json(data, filepath: str, mode="w+", changeLine=False):
        """
        将数据写入到Json文件中
        :param data: 数据
        :param filepath: json文件的绝度路径
        :param mode: 写模式
        :param changeLine:是否在每个两个数据之中空一行，是则True，否则False
        :return: 无
        """
        with open(filepath, mode, encoding="utf-8", ) as f:
            json.dump(data, f, ensure_ascii=False)
            if changeLine:
                f.write("\n")

    @staticmethod
    def readJson(filepath: str, mode="r+"):
        """
        从json文件从读取json串，并转化为python动态数据
        :param filepath: json文件所在绝对路径
        :param mode: 读模式
        :return: 转化后的python动态数据
        """
        with open(filepath, mode, encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def writeTriad2csv(filepath: str, TriadList: list, mode="a"):
        """
        将三元组写入到csv文件中
        :param filepath: csv文件的绝对路径
        :param TriadList: 三元组列表，例如[['小王'，'爸爸'，'老王'],['小白','哥哥','大白']]
        :param mode: 写模式
        :return: 无
        """
        with open(filepath, mode=mode, encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(TriadList)
