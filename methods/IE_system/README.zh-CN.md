Language : [🇨🇳](./README.zh-CN.md) ｜[🇺🇸](./README.md)

# IE_system

本项目尝试对某个领域中的一些文本（新闻、文章等）中持续抽取三元组，从而不断产出结构化的数据，用于知识图谱的构建等下游任务。（理想状态😀）

## 目录

- [系统工作流介绍](#系统工作流介绍)
- [项目意义和目标](#项目意义和目标)
- 

## 系统工作流介绍

这个系统包含三个部分。

第一，从文本中提取三元组

1. 使用 Spacy 工具对输入的文本进行依存分析。
2. 设计规则提取句子中的主语、谓语、宾语成分作为三元组。

第二、筛选三元组并且推荐给用户进行标注

1. 对三元组进行降噪。
2. 使用知识表示学习的方法对三元组打分，例如，TransE方法。更多的方法请参考：<https://github.com/thunlp/OpenKE>。
3. 推荐得分较高的三元组给用户标注。
4. 用户标注三元组，标注内容有两个：
   1. 对实体标注实体类型
   2. 对关系标注关系类型
5. 输出标注过的实体和关系，并且输出一套统计信息。

## 项目意义和目标

本项目连接了三个任务：

1. 实体和关系的发现
2. 主动学习（还不确定是不是主动学习）
3. 增量学习（也称作持续学习）

## 系统使用

本系统依赖的主要库如下：

```
  - numpy=1.20.2
  - numpy-base=1.20.2
  - python=3.7.0
  - pytorch=1.6.0
  - torchtext=0.6.0
  - spacy==3.0.6
  - tokenizers==0.10.3
  - transformers==4.5.1
  - zh-core-web-trf==3.0.0
```

### 1 安装Spacy

[spaCy · Industrial-strength Natural Language Processing in Python](https://spacy.io/)

这一个工业级的NLP工具，可以进行分词、词性标注、依存分析等NLP基础任务。就目前NLP的发展，一般认为这三个问题已经解决，所以本项目直接使用Spacy工具进行这三个任务的处理。

Spacy工具安装命令：

```
conda install -c conda-forge spacy
conda install -c conda-forge cupy
```

Spacy的中文模型下载：

https://github.com/explosion/spacy-models/releases/download/zh_core_web_trf-3.0.0/zh_core_web_trf-3.0.0-py3-none-any.whl

Spacy的中文模型安装：

```
pip install zh_core_web_trf-3.0.0-py3-none-any.whl
```

更详细的安装指导可以见此文档：[Install spaCy · spaCy Usage Documentation](https://spacy.io/usage)

测试中文模型是否安装成功：

```
import spacy as sp
sp.require_gpu()
nlp = sp.load("zh_core_web_trf")
```

可能的报错:

1. 如果无法加载中文模型的参数，那么令pytorch的版本为1.6.0

   ```
   conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
   ```

### 2  执行文件

直接运行`test_oie.py`即可。





