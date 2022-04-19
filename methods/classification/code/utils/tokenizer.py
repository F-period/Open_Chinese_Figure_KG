from transformers import BertModel, BertTokenizer
from fastNLP import Vocabulary, DataSet, Const
from fastNLP.io import DataBundle

# 【训练模型的辅助模块】
# 【实现bert的tokenizer】

class MyBertTokenizer:
    def __init__(self, target_vocab_list, bert_model_name, target_vocab_name='target'):
        # self.bert_model = BertModel.from_pretrained(bert_model_name)
        # 【从bert_modle_name里获取一个基于WordPiece实现的bert分词器】
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        # 【指明分类模型和分句符号】
        self.cls = '[CLS]'
        self.sep = '[SEP]'

        # 【设置目标词汇集】
        self.target_vocab = Vocabulary(padding=None, unknown=None)
        self.target_vocab.add_word_lst(target_vocab_list)
        self.target_size = len(self.target_vocab)

        # 【data bundle:一个类似字典的数据存储单元】
        # 【根据目标词汇集进行构造】
        self.data_bundle = DataBundle()
        self.data_bundle.set_vocab(self.target_vocab, target_vocab_name)

    # 【通过填充&截断进行尺寸的归一化】
    # 【返回归一化之后的结果和填充值】
    def padAndTruncate(self, arr, padding, truncate_mode='tail', truncate=510):
        paddings = []
        new_arr = arr[:]
        if truncate > len(arr):
            # 【如果原数据较小，则填充】
            paddings = [padding] * (truncate - len(arr))
        # 【如果原数据较大，进行截断操作，从头或从尾】
        elif truncate_mode == 'tail':
            new_arr = new_arr[:truncate]
        else:
            new_arr = new_arr[-truncate:]

        assert len(new_arr)+len(paddings) == 510
        return new_arr, paddings

    # 【将目标单词转换为one_hot表示形式】
    def _get_target_one_hot(self, id):
        result = [0] * self.target_size
        if id >= 0 and id < self.target_size:
            result[id] = 1
        return result

    # 【对数据进行编码】
    def _encode(self, data, target=None, truncate_mode='tail'):
        # 【对传输进来的数据进行编码处理】
        assert truncate_mode == 'tail' or truncate_mode == 'head'
        processed_data, data_padding = self.padAndTruncate(data, 0, truncate_mode)
        processed_data = [self.cls]+processed_data+[self.sep]
        # 【将数据转换为id】
        if truncate_mode=='tail':
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(processed_data) + data_padding
        else:
            input_ids = data_padding + self.bert_tokenizer.convert_tokens_to_ids(processed_data)
        # 【确保数据处理的正确】
        assert len(input_ids) == 512

        # 【如果是目标，转化为one_hot表示形式】
        if target:
            target_ids = None
            target_index = self.target_vocab.to_index(target) if target in self.target_vocab else -1
            target_ids = self._get_target_one_hot(target_index)

        if target:
            return input_ids, target_ids
        else:
            return input_ids, None

    # 【对数据(训练集？)进行处理】
    def dataProcessor(self, data_list, name):
        data = {Const.RAW_WORD: [], Const.CHAR_INPUT: [],
                Const.TARGET: [], Const.INPUT_LEN: []}
        # 【对data_list逐词进行编码】
        for words, target in data_list:
            input_ids, target_ids = self._encode(words, target)
            data[Const.RAW_WORD].append(''.join(words))
            data[Const.CHAR_INPUT].append(input_ids)
            data[Const.TARGET].append(target_ids)
            data[Const.INPUT_LEN].append(len(input_ids)-1)
        # 【将处理结果加入Bundle】
        self.data_bundle.set_dataset(DataSet(data), name=name)

    # 【对测试数据进行处理】
    def testDataProcessor(self, data_list, name):
        data = {'sample_id': [],Const.RAW_WORD: [], Const.CHAR_INPUT: [], Const.INPUT_LEN: []}
        for idx, words in enumerate(data_list):
            input_ids, _ = self._encode(words)
            data['sample_id'].append(idx)
            data[Const.RAW_WORD].append(''.join(words))
            data[Const.CHAR_INPUT].append(input_ids)
            data[Const.INPUT_LEN].append(len(input_ids)-1)
        # 【将处理结果加入Bundle】
        self.data_bundle.set_dataset(DataSet(data), name=name)

    # 【获取处理并存储在bundle里的数据】
    def getDataSet(self):
        self.data_bundle.set_input(Const.CHAR_INPUT, Const.TARGET)
        self.data_bundle.set_target(Const.RAW_WORD, Const.TARGET, Const.INPUT_LEN)

        return self.data_bundle

# 【与上述结构相似，相比于上述的：多出数据内容】
class MyBertContentTokenizer:
    def __init__(self, target_vocab_list, bert_model_name, target_vocab_name='target'):
        # self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.cls = '[CLS]'
        self.sep = '[SEP]'

        self.target_vocab = Vocabulary(padding=None, unknown=None)
        self.target_vocab.add_word_lst(target_vocab_list)
        self.target_size = len(self.target_vocab)

        self.data_bundle = DataBundle()
        self.data_bundle.set_vocab(self.target_vocab, target_vocab_name)

    def padAndTruncate(self, item, content, padding, truncate=509):
        paddings = []
        arr1 = list(item)
        arr2 = list(content)

        new_arr1 = []
        new_arr2 = []
        # 【截断和填充时要增进考虑内容项】
        if truncate <= len(arr1):
            new_arr1 = arr1[:truncate]
        elif truncate <= len(arr1)+len(arr2):
            new_arr1 = arr1[:]
            new_arr2 = arr2[:truncate-len(arr1)]
        else: # truncate > len(arr1)+len(arr2)
            new_arr1 = arr1[:]
            new_arr2 = arr2[:]
            paddings = [padding] * (truncate - len(arr1) - len(arr2))

        return new_arr1, new_arr2, paddings

    def _get_target_one_hot(self, id):
        result = [0] * self.target_size
        if id >= 0 and id < self.target_size:
            result[id] = 1

        return result

    def _encode(self, item, content, target=None,):
        p1, p2, padding = self.padAndTruncate(item, content, 0)

        processed_data = [self.cls] + p1 + [self.sep] + p2 + [self.sep]
        seq_len = len(p1) + 2
        item_mask = ([1] * seq_len) + ([0] * (512 - seq_len))
        assert len(item_mask) == 512

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(processed_data) + padding
        if len(input_ids) != 512:
            print('catch!')
        assert len(input_ids) == 512

        target_ids = None
        if target:
            target_index = self.target_vocab.to_index(target) if target in self.target_vocab else -1
            target_ids = self._get_target_one_hot(target_index)

        return input_ids, item_mask, target_ids

    def dataProcessor(self, data_list, name):
        # 【读入数据时的处理，对每一个data存成下列格式，注意content】
        data = {
            'item': [],
            'content': [],
            'chars': [],
            'item_mask': [],
            'seq_len': [],
            'target': [],
        }
        for item, content, target in data_list:
            input_ids, item_mask, target_ids = self._encode(item, content, target)
            data['item'].append(item)
            data['content'].append(content)
            data['chars'].append(input_ids)
            data['item_mask'].append(item_mask)
            data['seq_len'].append(512)
            data['target'].append(target_ids)
        self.data_bundle.set_dataset(DataSet(data), name=name)

    def testDataProcessor(self, data_list, name):
        data = {
            'item': [],
            'content': [],
            'chars': [],
            'item_mask': [],
            'seq_len': [],
        }
        for item, content in data_list:
            input_ids, item_mask, _ = self._encode(item, content, None)
            data['item'].append(item)
            data['content'].append(content)
            data['chars'].append(input_ids)
            data['item_mask'].append(item_mask)
            data['seq_len'].append(512)
        self.data_bundle.set_dataset(DataSet(data), name=name)

    def getDataSet(self):
        self.data_bundle.set_input('chars', 'item_mask', 'target')
        self.data_bundle.set_target('item', 'content', 'seq_len', 'target')

        return self.data_bundle