from fastNLP.modules import ConditionalRandomField, allowed_transitions
from transformers import BertModel, BertConfig
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

# 【训练模型的辅助模块】

# 【bert实体分类模型，扩展自pytorch中的模型】
class BertEntityTypeClassifier(nn.Module):
    def __init__(self, target_vocab, bert_model_name, hidden_dim, fc_dropout, use_ll_cls= True, bert_st=None, bert_ed=None):
        """
        :param tag_vocab: fastNLP Vocabulary
        """
        super().__init__()
        # 【进行相应参数的初始化：隐藏层、目标词汇集】
        self.hidden_dim = hidden_dim
        self.target_vocab = target_vocab
        self.bert_config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)

        # 【定义开始层以及结束层】
        if bert_st is None:
            bert_st = -1 # default last layer
            bert_ed = None
        elif bert_ed is not None and bert_ed <= bert_st:
            bert_ed = None # default only fetch 1 layer

        # 【计算bert层数】
        self.bert_layers = 1
        if bert_ed is not None:
            self.bert_layers = bert_ed - bert_st
        elif bert_st < 0:
            self.bert_layers = - bert_st
        else:
            self.bert_layers = self.bert_config.num_hidden_layers - bert_st

        self.bert_h_st = bert_st
        self.bert_h_ed = bert_ed

        # 【这是否是一个分类模型】
        self.use_cls = use_ll_cls

        # 【初始化bert的编码器】
        self.bert_encoder = BertModel.from_pretrained(bert_model_name, config=self.bert_config)
        # 【根据层级计算模型的特征大小】
        self.bert_feature_size = self.bert_config.hidden_size * self.bert_layers
        # 【全连接层】
        if self.use_cls:
            # max, min and avg pooling, plus last layer cls
            self.fc_feature_size = 3 * self.bert_feature_size + self.bert_config.hidden_size
        else:
            self.fc_feature_size = 3 * self.bert_feature_size
        self.mid_fc = nn.Linear(self.fc_feature_size, hidden_dim)
        self.mid_fc_a = nn.GELU()
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(hidden_dim, len(target_vocab))
        self.out_fc_a = nn.Sigmoid()

    def _predict_with_logits(self, logits):
        return (nn.Sigmoid()(logits)>0.5).int()

    # 【forward轮】
    def _forward(self, input_ids, target_ids):
        # TODO: 几层 dropout
        mask = input_ids.ne(0).float() # batch x seq_len
        bert_result = self.bert_encoder(input_ids, attention_mask=mask)
        bert_hidden_states = bert_result[2]
        bert_feature = torch.cat(bert_hidden_states[self.bert_h_st:self.bert_h_ed], dim=-1) # batch * seq_len * bert_feature_size

        # do pooling
        bert_feature_tr = bert_feature.transpose(1, 2) # batch * bert_feature_size * seq_len
        feature_level_mask = mask.unsqueeze(1).repeat(1, self.bert_feature_size, 1)

        # masked max pool
        feature_for_max_pool = bert_feature_tr * feature_level_mask
        feature_for_max_pool[feature_level_mask.eq(0)] = float('-10000.0') # set a reasonable small value for max
        feature_masked_maxpool, _ = feature_for_max_pool.max(-1) # batch * bert_feature_size

        # masked min pool
        feature_for_min_pool = bert_feature_tr * feature_level_mask
        feature_for_min_pool[feature_level_mask.eq(0)] = float('10000.0')  # set a reasonable large value for min
        feature_masked_minpool, _ = feature_for_min_pool.min(-1)  # batch * bert_feature_size

        # masked mean pool
        feature_for_mean_pool = bert_feature_tr * feature_level_mask
        feature_masked_meanpool = feature_for_mean_pool.sum(-1) / feature_level_mask.sum(-1) # batch * bert_feature_size

        feature_list = []
        if self.use_cls:
            # last layer [CLS] output
            feature_from_last_cls = bert_result[0][:, 0, :].squeeze(1) # batch * bert_hidden_size

            feature_list = [feature_masked_maxpool, feature_masked_minpool, feature_masked_meanpool, feature_from_last_cls]
        else:
            feature_list = [feature_masked_maxpool, feature_masked_minpool, feature_masked_meanpool]

        feature = torch.cat(feature_list, dim=-1)
        fc1_o = self.mid_fc_a(self.mid_fc(feature))
        fc2_i = self.fc_dropout(fc1_o)
        logits = self.out_fc(fc2_i)
        #p = self.out_fc_a(logits) # sigmoid multi classification

        if target_ids is None:
            # TODO: 是否加入mask
            return {'pred': self._predict_with_logits(logits), 'logits': logits}
        else:
            loss = nn.BCEWithLogitsLoss(reduction='none')(logits, target_ids.float())
            return {'loss': loss}

    # 【调用forward-->见上定义】
    def forward(self, chars, target):
        return self._forward(chars, target)

    # 【调用predict-->见上定义】
    def predict(self, chars):
        return self._forward(chars, None)

class BertEntityTypeWithContentClassifier(nn.Module):
    def __init__(self, target_vocab, bert_model_name, hidden_dim, fc_dropout, use_ll_cls= True, bert_st=None, bert_ed=None):
        """
        :param tag_vocab: fastNLP Vocabulary
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_vocab = target_vocab

        self.bert_config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)

        if bert_st is None:
            bert_st = -1 # default last layer
            bert_ed = None
        elif bert_ed is not None and bert_ed <= bert_st:
            bert_ed = None # default only fetch 1 layer

        self.bert_layers = 1
        if bert_ed is not None:
            self.bert_layers = bert_ed - bert_st
        elif bert_st < 0:
            self.bert_layers = - bert_st
        else:
            self.bert_layers = self.bert_config.num_hidden_layers - bert_st

        self.bert_h_st = bert_st
        self.bert_h_ed = bert_ed

        self.use_cls = use_ll_cls

        self.bert_encoder = BertModel.from_pretrained(bert_model_name, config=self.bert_config)
        self.bert_feature_size = self.bert_config.hidden_size * self.bert_layers
        if self.use_cls:
            # max, min and avg pooling, plus last layer cls
            self.fc_feature_size = 3 * self.bert_feature_size + self.bert_config.hidden_size
        else:
            # max, min and avg pooling, plus last layer cls
            self.fc_feature_size = 3 * self.bert_feature_size
        self.mid_fc = nn.Linear(self.fc_feature_size, hidden_dim)
        self.mid_fc_a = nn.GELU()
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(hidden_dim, len(target_vocab))
        self.out_fc_a = nn.Sigmoid()

    # 【借助logits进行预测】
    def _predict_with_logits(self, logits):
        return (nn.Sigmoid()(logits)>0.5).int()

    # 【forward轮】
    def _forward(self, input_ids, item_mask, target_ids):
        att_mask = input_ids.ne(0).float() # batch x seq_len
        bert_result = self.bert_encoder(input_ids, attention_mask=att_mask)
        bert_hidden_states = bert_result[2]
        bert_feature = torch.cat(bert_hidden_states[self.bert_h_st:self.bert_h_ed], dim=-1) # batch * seq_len * bert_feature_size

        # 【计算三种mask】
        mask = item_mask.float()
        # do pooling
        bert_feature_tr = bert_feature.transpose(1, 2) # batch * bert_feature_size * seq_len
        feature_level_mask = mask.unsqueeze(1).repeat(1, self.bert_feature_size, 1)

        # masked max pool
        feature_for_max_pool = bert_feature_tr * feature_level_mask
        feature_for_max_pool[feature_level_mask.eq(0)] = float('-10000.0') # set a reasonable small value for max
        feature_masked_maxpool, _ = feature_for_max_pool.max(-1) # batch * bert_feature_size

        # masked min pool
        feature_for_min_pool = bert_feature_tr * feature_level_mask
        feature_for_min_pool[feature_level_mask.eq(0)] = float('10000.0')  # set a reasonable large value for min
        feature_masked_minpool, _ = feature_for_min_pool.min(-1)  # batch * bert_feature_size

        # masked mean pool
        feature_for_mean_pool = bert_feature_tr * feature_level_mask
        feature_masked_meanpool = feature_for_mean_pool.sum(-1) / feature_level_mask.sum(-1) # batch * bert_feature_size

        feature_list = []
        if self.use_cls:
            # last layer [CLS] output
            # 【是否最后要结合[cls]层的特征】
            feature_from_last_cls = bert_result[0][:, 0, :].squeeze(1) # batch * bert_hidden_size

            feature_list = [feature_masked_maxpool, feature_masked_minpool, feature_masked_meanpool, feature_from_last_cls]
        else:
            feature_list = [feature_masked_maxpool, feature_masked_minpool, feature_masked_meanpool]

        feature = torch.cat(feature_list, dim=-1)
        fc1_o = self.mid_fc_a(self.mid_fc(feature))
        fc2_i = self.fc_dropout(fc1_o)
        logits = self.out_fc(fc2_i)
        #p = self.out_fc_a(logits) # sigmoid multi classification

        if target_ids is None:
            # TODO: 是否加入mask
            # 【如果并非目标，返回预测结果及其logits】
            return {'pred': self._predict_with_logits(logits), 'logits': logits}
        else:
            loss = nn.BCEWithLogitsLoss(reduction='none')(logits, target_ids.float())
            # 【否则返回这一轮的损失】
            return {'loss': loss}

    # 【调用forward-->见上定义】
    def forward(self, chars, item_mask, target):
        return self._forward(chars, item_mask, target)

    # 【调用predict-->见上定义】
    def predict(self, chars, item_mask):
        return self._forward(chars, item_mask, None)