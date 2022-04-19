# 模型训练

## 模型结构

我们使用BERT作为编码器来提取文本中的特征，而BERT的预训练权重使用哈工大发布的[RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm)。因为BERT最后一层的特征与预训练的任务关联比较强，我们取出倒数第二、三层的输出，拼接后作为下一步的输入来使用。接下来，我们对输出的特征做max、min和avg的pooling，并且拼接在一起。根据是否存在额外的实体详细描述的文本，模型的实现有略微的不同，但是pooling的范围只是实体本身文本的范围(包含开头的[CLS]和后面紧接的[SEP])。最后，pooling的特征经过一个两层MLP，输出最终的7类(药物、疾病、症状、检查科目、细菌、病毒、医学专科)的logits，logits经过sigmoid后能得到概率，如果所有类别的概率都小于0.5，那么我们输出NoneType分类。详细的结构请参考我们提供的训练代码。

## 训练数据

我们使用了如下几种类型的训练数据：
* 原始的训练实体+类型one-hot标签：其中文本输入格式为`[CLS] + 实体文本 + [SEP]`
* 原始的训练实体+bing10+类型one-hot标签：bing10是使用训练数据中的实体，去 [必应(bing)](https://cn.bing.com/) 查询所得的top10 snippet结果，作为实体的文字描述。此时每一个实体都会被扩充成约10个条目，每个条目文本输入形式为`[CLS] + 实体文本 + [SEP] + bing某条snippet + [SEP]`。在做预测时，使用与实体相关对应的所有条目预测结果之间的平均投票作为最终的预测。
* 原始的训练实体+c1b1+类型one-hot标签：c1b1是使用训练数据中的实体，去 [中国医药信息查询平台(CMIP)](https://www.dayi.org.cn/) 的top 1查询结果加上bing的top 1结果，如果CMIP没法精确匹配，则使用bing的top 2结果。此时每一个实体都会被扩充成2个条目，每个条目文本输入形式为`[CLS] + 实体文本 + [SEP] + CMIP或bing + [SEP]`。在做预测时，同样使用所有预测结果的投票作为最终的预测。

我们对训练数据按照8:1:1的比例划分为T+V1+V2，采用下面模式来做验证。
* 使用T训练，同时使用V1和V2验证，取两个模型：V1验证F1最高+V2验证F1最高
* 使用T+V2训练，使用V2验证，取一个模型：V2验证F1最高
* 使用T+V1训练，使用V2验证，取一个模型：V1验证F1最高

训练数据共9个文件，分别为：
* **train_val1_2.txt**
    * 原始实体，训练T -> 验证V1 + V2
* **train_val1.txt**
    * 原始实体，训练T + V2 -> 验证V1
* **train_val2.txt**
    * 原始实体，训练T + V1 -> 验证V2
* **train_bing10_val1_2.txt**
    * 原始实体+bing10，训练T -> 验证V1 + V2
* **train_bing10_val1.txt**
    * 原始实体+bing10，训练T + V2 -> 验证V1
* **train_bing10_val2.txt**
    * 原始实体+bing10，训练T + V1 -> 验证V2
* **train_c1b1_val1_2.txt**
    * 原始实体+c1b1，训练T -> 验证V1 + V2
* **train_c1b1_val1.txt**
    * 原始实体+c1b1，训练T + V2 -> 验证V1
* **train_c1b1_val2.txt**
    * 原始实体+c1b1，训练T + V1 -> 验证V2

训练数据的下载链接为[百度网盘](https://pan.baidu.com/s/1663KVascFPD-wkklUOph0Q)，提取码：7js2。

## 训练细节

我们使用python3的pytorch进行实现，使用fastNLP的一些框架代码和huggingface的transformers中的BERT实现。模型训练BERT fine-tune的学习率取1e-5，MLP层的学习率取1e-3，使用Tesla V100进行训练，batch大小为32(物理4*更新间隔8)，根据不同的设置，使用Adam或者SGD优化器，训练10~100个epoch。

## 模型

我们一共生成了14个模型进行投票，每个模型的使用的训练参数如下：
* **model_1_epoch_6, model_1_epoch_26**
    * `python bc.py --train-path=./train_data/train_val1_2.txt --epochs=50 --optim=adam --run-tag=1 --device=cuda:0`
* **model_2_epoch_42**
    * `python bc.py --train-path=./train_data/train_val1.txt --epochs=50 --optim=adam --run-tag=2 --device=cuda:0`
* **model_3_epoch_17**
    * `python bc.py --train-path=./train_data/train_val2.txt --epochs=50 --optim=adam --run-tag=3 --device=cuda:0`
* **model_4_epoch_1, model_4_epoch_2**
    * `python bcc.py --train-path=./train_data/train_bing10_val1_2.txt --epochs=10 --optim=adam --run-tag=4 --device=cuda:0`
* **model_5_epoch_4**
    * `python bcc.py --train-path=./train_data/train_bing10_val1.txt --epochs=10 --optim=adam --run-tag=5 --device=cuda:0`
* **model_6_epoch_2**
    * `python bcc.py --train-path=./train_data/train_bing10_val2.txt --epochs=10 --optim=adam --run-tag=6 --device=cuda:0`
* **model_7_epoch_13, model_7_epoch_17**
    * `python bcc.py --train-path=./train_data/train_bing10_val1_2.txt --epochs=20 --optim=sgd --run-tag=7 --device=cuda:0`
* **model_8_epoch_74, model_8_epoch_89**
    * `python bcc.py --train-path=./train_data/train_c1b1_val1_2.txt --epochs=100 --optim=sgd --run-tag=8 --device=cuda:0`
* **model_9_epoch_73**
    * `python bcc.py --train-path=./train_data/train_c1b1_val1.txt --epochs=100 --optim=sgd --run-tag=9 --device=cuda:0`
* **model_10_epoch_54**
    * `python bcc.py --train-path=./train_data/train_c1b1_val2.txt --epochs=100 --optim=sgd --run-tag=10 --device=cuda:0`

根据不同的硬件配置、随机数设定和浮点误差，实际跑出的epoch选择可能和我们跑完的结果会有一些差别。由于模型数据非常大(~18G)，我们给出下载链接如下，[百度网盘](https://pan.baidu.com/s/1h62XTMT23zkg14H5rSxqbA)，提取码：nw1a。

# 模型预测

## 测试数据

和训练数据一样，我们提供了测试的数据如下：
* **entity_test.txt**
    * 原始测试数据
* **test_with_bing10.txt**
    * 原始测试数据+bing10
* **test_with_c1b1.txt**
    * 原始测试数据+c1b1

测试数据的下载链接为：[百度网盘](https://pan.baidu.com/s/1PtkSUx4dr6m4aWM6tGXjEA)，提取码：wm80。

## 模型预测方法

模型预测的结果是14个模型预测的平均投票。我们提供了在测试集合上已经预测完毕的logits放在pred_data目录下。当然logits文件也可以用代码生成，下载提供的模型后，所有logits文件的生成方式如下：

* **logits_test_m1_e6.txt**
    * `python bc.py --train=2 --test-path=./test_data/entity_test.txt --model-path=./models/model_1_epoch_6 --test-id=test_m1_e6 --device=cuda:0`
* **logits_test_m1_e26.txt**
    * `python bc.py --train=2 --test-path=./test_data/entity_test.txt --model-path=./models/model_1_epoch_26 --test-id=test_m1_e26 --device=cuda:0`
* **logits_test_m2_e42.txt**
    * `python bc.py --train=2 --test-path=./test_data/entity_test.txt --model-path=./models/model_2_epoch_42 --test-id=test_m2_e42 --device=cuda:0`
* **logits_test_m3_e17.txt**
    * `python bc.py --train=2 --test-path=./test_data/entity_test.txt --model-path=./models/model_3_epoch_17 --test-id=test_m3_e17 --device=cuda:0`
* **logits_test_m4_e1.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_bing10.txt --model-path=./models/model_4_epoch_1 --test-id=test_m4_e1 --device=cuda:0`
* **logits_test_m4_e2.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_bing10.txt --model-path=./models/model_4_epoch_2 --test-id=test_m4_e2 --device=cuda:0`
* **logits_test_m5_e4.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_bing10.txt --model-path=./models/model_5_epoch_4 --test-id=test_m5_e4 --device=cuda:0`
* **logits_test_m6_e2.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_bing10.txt --model-path=./models/model_6_epoch_2 --test-id=test_m6_e2 --device=cuda:0`
* **logits_test_m7_e13.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_bing10.txt --model-path=./models/model_7_epoch_13 --test-id=test_m7_e13 --device=cuda:0`
* **logits_test_m7_e17.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_bing10.txt --model-path=./models/model_7_epoch_17 --test-id=test_m7_e17 --device=cuda:0`
* **logits_test_m8_e74.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_c1b1.txt --model-path=./models/model_8_epoch_74 --test-id=test_m8_e74 --device=cuda:0`
* **logits_test_m8_e89.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_c1b1.txt --model-path=./models/model_8_epoch_89 --test-id=test_m8_e89 --device=cuda:0`
* **logits_test_m9_e73.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_c1b1.txt --model-path=./models/model_9_epoch_73 --test-id=test_m9_e73 --device=cuda:0`
* **logits_test_m10_e54.txt**
    * `python bcc.py --train=2 --test-path=./test_data/test_with_c1b1.txt --model-path=./models/model_10_epoch_54 --test-id=test_m10_e54 --device=cuda:0`

要生成模型预测结果，请确保pred_data文件夹存在上述logits文件，然后使用如下命令：
```
python model_pred.py
```
可以生成模型预测结果`results_test_model_vote.txt`。

# 最终预测

## 预测方法

首先，我们使用规则进行匹配，无法匹配到的，我们使用模型来做预测。接下来，我们使用[LSTM-CRF-medical代码库](https://github.com/yixiu00001/LSTM-CRF-medical)中的疾病和症状对预测进行修正，其中github_disease.txt和github_symptom.txt分别从[disease汇总全部数据.xlsx](https://github.com/yixiu00001/LSTM-CRF-medical/blob/master/datasets/disease%E6%B1%87%E6%80%BB%E5%85%A8%E9%83%A8%E6%95%B0%E6%8D%AE.xlsx)和[symptom症状全数据.xlsx](https://github.com/yixiu00001/LSTM-CRF-medical/blob/master/datasets/symptom%E7%97%87%E7%8A%B6%E5%85%A8%E6%95%B0%E6%8D%AE.xlsx)中抽取得到。最后通过后处理规则修正，得到最终结果

用如下命令：
```
python pre_rule_pred.py # 生成基于规则的结果
python final_pred.py # 生成模型和规则融合的结果
```
可以生成最终的预测结果`result.txt`。