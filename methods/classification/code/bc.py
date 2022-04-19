import os, math, random, argparse
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.autograd as autograd
from transformers import BertModel, BertTokenizer
from fastNLP import Vocabulary, DataSet, Const
from fastNLP import Trainer, Tester
from fastNLP import SpanFPreRecMetric, ClassifyFPreRecMetric, MetricBase, GradientClipCallback, WarmupCallback, Callback
from fastNLP.core.losses import LossInForward
from fastNLP.modules.decoder import ConditionalRandomField
from fastNLP.io import DataBundle
from utils.preprocess import MyDataLoader
from utils.models import BertEntityTypeClassifier
from utils.tokenizer import MyBertTokenizer

# 【模型代码】

# 【获取模型的预测结果】
def get_pred(target_vocab, preds, raw_words):
    # 【转换成相应列表】
    pred_l = preds.tolist()
    raw_words = list(raw_words)
    # 【转换成相应列表】
    assert len(pred_l) == len(raw_words)
    result = []
    # 【组合成预测结果:标号到单词】
    for pred, word in zip(pred_l, raw_words):
        types = []
        for j, pred_flag in enumerate(pred):
            if pred_flag != 0:
                types.append(target_vocab.idx2word[j])
        if types:
            result.append((word, ';'.join(types)))
        else:
            result.append((word, 'NoneType'))

    return result

# 【获取预测结果和对于的logitss】
def get_pred_and_logits(target_vocab, preds, logits, raw_words):
    # 【转换成相应列表】
    pred_l = preds.tolist()
    logits_l = logits.tolist()
    logits_idx_l = logits.argmax(-1).tolist()
    is_none_l = (1-preds.sum(-1)).tolist()
    raw_words = list(raw_words)

    # 【确认维度是一致的】
    assert len(pred_l) == len(raw_words)
    result = []
    # 【组合成预测结果:标号到单词】
    for is_none, best_id, word in zip(is_none_l, logits_idx_l, raw_words):
        if is_none != 0:
            result.append((word, 'NoneType'))
        else:
            result.append((word, target_vocab.idx2word[best_id]))

    # 【组合成logits】
    result_logits = []
    for word, logits in zip(raw_words, logits_l):
        result_logits.append([word] + logits)

    return result, result_logits

# 【PR矩阵：用于模型评价的自定义类，有精确率和召回率决定】
class MyPRMetrics(ClassifyFPreRecMetric):
    def __init__(self, target_vocab):
        # 【确定目标词汇集】
        self.target_vocab = target_vocab
        super().__init__()

        self.ground_truth = set()
        self.prediction = set()

    # 【评价计算结果】
    # raw_word：原始单词，pred预测结果，target目标结果
    def evaluate(self, raw_words, pred, target, seq_len):

        batch_size = pred.size(0)
        for i in range(batch_size):
            cur_words = raw_words[i]
            s = 0
            # 【计算prediction】
            for j, pred_flag in enumerate(pred[i]):
                s += pred_flag.item()
                if pred_flag.item() != 0:
                    self.prediction.add(f"{cur_words}_{self.target_vocab.idx2word[j]}")
            if s == 0:
                self.prediction.add(f"{cur_words}_NoneType")

            s = 0
            # 【计算ground_truth】
            for j, label_flag in enumerate(target[i]):
                s += label_flag.item()
                if label_flag.item() != 0:
                    self.ground_truth.add(f"{cur_words}_{self.target_vocab.idx2word[j]}")
            if s == 0:
                self.ground_truth.add(f"{cur_words}_NoneType")

    # 【获取类中的内容】
    def get_metric(self, reset=True):
        tp_set = self.ground_truth.intersection(self.prediction)
        tp = len(tp_set)
        p = len(self.prediction)
        t = len(self.ground_truth)
        print(tp, p, t)
        epsilon = 1e-13
        prec = tp / (p + epsilon)
        rec = tp / (t + epsilon)
        f = 2 * prec * rec / (prec + rec + epsilon)
        # 【默认情况下清空矩阵】
        if reset:
            self.ground_truth = set()
            self.prediction = set()
        # 【返回F-measur、精确率、召回率三种指标】
        return {'f': f, 'pre': prec, 'rec': rec}

# 【损失评价，采用BCELoss函数】
class BCELossMetrics(MetricBase):
    def __init__(self, target_vocab):
        super().__init__()

        self.val_loss = 0
        self.val_size = 0
        self.multi = len(target_vocab)

    # 【评价BCE损失，调用pytorch里的想要相应函数】
    def evaluate(self, logits, target):
        with torch.no_grad():
            loss = nn.BCEWithLogitsLoss(reduction='none')(logits, target.float())
            self.val_loss += torch.sum(loss).float().item()
            self.val_size += (loss.view(-1)).size(0) * self.multi

    # 【获取类中的内容】
    def get_metric(self, reset=True):
        val_loss = self.val_loss / self.val_size if self.val_size != 0 else 0
        # 【默认情况下清空改矩阵】
        if reset:
            self.val_loss = 0
            self.val_size = 0
        # 【返回损失】
        return {'loss': val_loss}

# 【为各个学习模型的随机指定统一的值，防止结果的不确定性】
def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 【处理器：拓展自fastNLP的回调处理】
class MyMetricsHandler(Callback):
    def __init__(self, model, extra_test_data, metrics, batch_size, tag = ''):
        super().__init__()
        self.metrics = []
        self.cur_stat = None
        self.file_path = f"./saved/run_log_{tag}_{str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S.%f'))}.txt"
        self.field_list = ['epoch', 'step', 'train_loss', 'val_loss', 'test_loss', 'val_f', 'test_f', 'val_pre', 'val_rec', 'test_pre', 'test_rec']
        if extra_test_data is None:
            self.extra_tester = None
        else:
            # 【如果有额外的测试集，根据它构建Tester】
            self.extra_tester = Tester(model=model,
                                     data=extra_test_data,
                                     metrics=metrics,
                                     batch_size=batch_size,
                                     device=None,
                                     verbose=0,
                                     use_tqdm=False)

    # 【获取某一域中的值】
    def _get_field_str_value(self, d, metric):
        return str(d.get(metric, ''))

    # 【字典更新】
    def _update_dict_with_prefix(self, d, data, prefix):
        for k, v in data.items():
            d[f"{prefix}_{k}"] = v

    # 【backward操作开始之前，存储当前状态】
    def on_backward_begin(self, loss):
        # 【如果到了需要更新的时候】
        if self.step % self.update_every == 0:
            self.cur_stat = {'epoch': self.epoch, 'step': self.step, 'train_loss': loss.item()}
        else:
            self.cur_stat = None

    # 【当一个数据batch结束之后】
    def on_batch_end(self):
        if ((self._trainer.validate_every > 0 and self.step % self._trainer.validate_every == 0) or
            (self._trainer.validate_every < 0 and self.step % len(self._trainer.data_iterator) == 0)) \
                and self._trainer.dev_data is not None:
            # since it need to do validation, metrics updates are postponed to validation step
            pass
        else:
            # 【如果有当前状态的：存储下来，否则等验证之后】
            if self.cur_stat is not None:
                self.metrics.append(self.cur_stat)
                self.cur_stat = {}

    # 【验证结束之后：对当前的状态也进行对于的更新】
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        # 【状态更新】
        if self.cur_stat is None:
            self.cur_stat = {'epoch': self.epoch, 'step': self.step}

        # 【字典更新】
        for k, v_dict in eval_result.items():
            self._update_dict_with_prefix(self.cur_stat, v_dict, 'val')

        # 【如果有额外的测试集，还要根据它再跟想一次k、v】
        if self.extra_tester is not None:
            test_res = self.extra_tester.test()
            for k, v_dict in test_res.items():
                self._update_dict_with_prefix(self.cur_stat, v_dict, 'test')
            print(f"-----test-----\n{self.extra_tester._format_eval_results(test_res)}\n----------")

        # 【存储验证完毕的状态并清空】
        self.metrics.append(self.cur_stat)
        self.cur_stat = {}

    # 【训练结束之后的操作】
    def on_train_end(self):
        with open(self.file_path, 'w', encoding='utf-8') as output:
            output.write('\t'.join(self.field_list))
            output.write('\n')
            # 【将每一轮中记录的状态信息都写入到输出文件中】
            for item in self.metrics:
                output.write('\t'.join(self._get_field_str_value(item, f) for f in self.field_list))
                output.write('\n')

# 【基于FastNLP的Trainer扩展的一个训练器】
class MyCustomizedTrainer(Trainer):
    def __init__(self, train_data, model, optimizer=None, loss=None,
                 batch_size=32, sampler=None, drop_last=False, update_every=1,
                 num_workers=0, n_epochs=10, print_every=5,
                 dev_data=None, metrics=None, metric_key=None,
                 validate_every=-1, save_path=None, use_tqdm=True, device=None,
                 callbacks=None, check_code_level=0, tag='', **kwargs):
        super().__init__(train_data, model, optimizer, loss,
                 batch_size, sampler, drop_last, update_every,
                 num_workers, n_epochs, print_every,
                 dev_data, metrics, metric_key,
                 validate_every, save_path, use_tqdm, device,
                 callbacks, check_code_level, **kwargs)
        # 【新增的机制：tag】
        self.tag = tag

    # 【根据输入的k,v值获取训练器中的计算结果】
    def _fetch_metric_value(self, result):
        for k, v_dict in result.items():
            if self.metric_key in v_dict:
                return str(v_dict[self.metric_key])

        return 0

    # 【通过callback进行验证】
    def _do_validation(self, epoch, step):
        self.callback_manager.on_valid_begin()
        res = self.tester.test()

        is_better_eval = False
        # save all epoch for later reference
        # 【将每一论的结果全部存储下来】
        if self.save_path is not None:
            path = os.path.join(self.save_path, "_".join(
                [self.tag, 'epoch', str(epoch)]))
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            torch.save(model.state_dict(), path)
        if self._better_eval_result(res):
            # if self.save_path is not None:
            #     self._save_model(self.model,
            #                      "best_" + "_".join([self.model.__class__.__name__, self.metric_key, self._fetch_metric_value(res), self.start_time]))
            # elif self._load_best_model:
            #     self._best_model_states = {name: param.cpu().clone() for name, param in self.model.state_dict().items()}
            self.best_dev_perf = res
            self.best_dev_epoch = epoch
            self.best_dev_step = step
            is_better_eval = True
        # get validation results; adjust optimizer
        # 【通过回调函数，每一轮进行一个调节】
        self.callback_manager.on_valid_end(res, self.metric_key, self.optimizer, is_better_eval)
        # 【最后返回结果】
        return res

if __name__ == "__main__":
    # 【设置模型设备及参数】
    # 【argparse:对命令行参数进行解析】
    parser = argparse.ArgumentParser(description='model parameters')
    parser.add_argument('--device', type=str, default='cpu', help='cpu, cuda:0, cuda:1...')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--update-every', type=int, default=8)
    parser.add_argument('--bert-model-name', type=str, default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument('--bert-lr', type=float, default=3e-5)
    parser.add_argument('--mlp-lr', type=float, default=1e-3)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--train-path', type=str, default='')
    parser.add_argument('--test-path', type=str, default='')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--run-tag', type=str, default='default')
    parser.add_argument('--test-id', type=str, default='')
    args = parser.parse_args()

    # 【设置启动种子】
    setup_seed(666)
    #torch.backends.cudnn.enabled = False
    #torch.autograd.set_detect_anomaly(True)

    # 【设置参数】
    hidden_dim = 1024
    dropout_rate = 0.2

    # 【输入输出路径】
    train_path = args.train_path
    test_path = args.test_path

    # 【标签列表】
    label_list = ['药物', '疾病', '症状', '检查科目', '细菌', '病毒', '医学专科']
    # 【自定义的bert分词器---见tokenizer.py】
    bert_tokenizer = MyBertTokenizer(label_list, args.bert_model_name)

    # 【bert分类模型】
    model = BertEntityTypeClassifier(
        bert_tokenizer.target_vocab,
        args.bert_model_name,
        hidden_dim,
        dropout_rate,
        use_ll_cls=False,
        bert_st=-3,
        bert_ed=-1
    )

    # 【如果输入的训练数据】
    if args.train==1:
        print(f"train file: {train_path}")
        # 【这里bc.py使用的训练数据路径：entity_test.txt】
        train_data, val_data, test_data = MyDataLoader.getTrainData(train_path)

        # random.shuffle(total_data)
        # 【将训练数据、值、(如果有)测试数据 加入bert的分词机器】
        bert_tokenizer.dataProcessor(train_data, 'train')
        bert_tokenizer.dataProcessor(val_data, 'dev')
        if test_data:
            bert_tokenizer.dataProcessor(test_data, 'test')


        # 【获取训练集】
        data_set = bert_tokenizer.getDataSet()

        print(f"train_set: {len(train_data)}\tdev_set: {len(val_data)}\ttest_set: {len(test_data)}")

        # 【初始化矩阵，分别是模型结果评价和损失评估->详见上面的类定义】
        metrics = [
            MyPRMetrics(data_set.get_vocab('target')),
            BCELossMetrics(data_set.get_vocab('target'))
        ]

        # 【设置bert模型参数】
        bert_param_ids = list(map(id, model.bert_encoder.parameters()))
        rest_params = filter(lambda x: id(x) not in bert_param_ids, model.parameters())

        hierarchical_lr = [{'params': model.bert_encoder.parameters(), 'lr': args.bert_lr},
                           {'params': rest_params, 'lr': args.mlp_lr, 'weight_decay': 0.01},
                           ]

        # 【优化器采用SGD或者Adam】
        if args.optim == 'sgd':
            optimizer = optim.SGD(hierarchical_lr, momentum=0.9)
        else:
            optimizer = optim.Adam(hierarchical_lr, amsgrad=False)

        tag = f"model_{args.run_tag}"
        check_level = 0

        # 【定义回调步骤所作的操作-->fastNLP在定义的callback的机制】
        callbacks = [
            # 【每次backward前，将parameter的gradient clip到某个范围】
            GradientClipCallback(clip_type='norm', clip_value=5.0), # clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
            #WarmupCallback(0.1, schedule='linear'),
            # 【自定义的backward处理-->见上面的类定义】
            MyMetricsHandler(model, data_set.get_dataset('test') if test_data else None, metrics, args.batch_size, tag)
        ]
        print(data_set.get_dataset('train'))

        # 【获取一个Trainer-->见上面的类定义】
        trainer = MyCustomizedTrainer(data_set.get_dataset('train'), model, optimizer, batch_size=args.batch_size, update_every=args.update_every,
                          n_epochs=args.epochs, dev_data=data_set.get_dataset('dev'), metrics=metrics,
                          dev_batch_size=args.batch_size, callbacks=callbacks, device=args.device, test_use_tqdm=False,
                          use_tqdm=False, print_every=100, metric_key='f',check_code_level=check_level,
                          save_path=f'./saved/{tag}', tag=tag)
        trainer.train(load_best_model=False)

    # 【如果输入的是测试数据】
    elif args.train == 2:
        print(f"test file: {test_path}")
        # 【读入测试数据】
        test_data = MyDataLoader.getTestData(test_path)
        bert_tokenizer.testDataProcessor(test_data, 'test')
        data_set = bert_tokenizer.getDataSet()
        test_set = data_set.get_dataset('test')

        # 【对模型进行评价】
        model.load_state_dict(torch.load(args.model_path))
        model.to(torch.device(args.device))
        model.eval()

        # 【输出测试的结果和logits】
        output_results = f"./results_{args.test_id}.txt"
        output_logits = f"./logits_{args.test_id}.txt"
        print(f"output results: {output_results} logits: {output_logits}")

        num_data = len(test_set)
        i = 0
        pred = []
        logits = []
        batch_size = args.batch_size
        # 【每多少轮输出一次】
        print_every = 500
        next_print = print_every
        while i < num_data:
            if i >= next_print:
                print(i)
                next_print += print_every
            # 【以batch为单位进行训练，这里防止越界】
            j = min(i + batch_size, num_data)
            batch_x = torch.tensor(test_set[i:j][Const.CHAR_INPUT]).view(j - i, -1).to(args.device)
            # 【调用模型获取预测结果】
            pred_result = model.predict(batch_x)
            batch_pred = pred_result['pred'].view(j - i, -1)
            batch_logits = pred_result['logits'].view(j - i, -1)
            # 【根据当前轮的结果更新pred和logits】
            cur_pred, cur_logits = get_pred_and_logits(data_set.get_vocab('target'),
                                                       batch_pred,
                                                       batch_logits,
                                                       test_set[i:j][Const.RAW_WORD])
            pred.extend(cur_pred)
            logits.extend(cur_logits)
            i = j

        # 【写入预测结果】
        with open(output_results, 'w', encoding='utf-8') as predict_file:
            for k, v in pred:
                predict_file.write(f"{k}\t{v}\n")

        # 【写入logits结果】
        with open(output_logits, 'w', encoding='utf-8') as logits_file:
            for item in logits:
                logits_file.write('\t'.join([str(term) for term in item]))
                logits_file.write('\n')
