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
from utils.preprocess import MyDataContentLoader
from utils.models import BertEntityTypeWithContentClassifier
from utils.tokenizer import MyBertContentTokenizer

# 【同样是用于生成模型logits的代码】
# 【和bc里很多地方是差不多一致的】
# 【这个文件里仅标出不一致的地方】

def get_pred(target_vocab, preds, raw_words):
    pred_l = preds.tolist()
    raw_words = list(raw_words)
    assert len(pred_l) == len(raw_words)
    result = []
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

def get_pred_and_logits(target_vocab, preds, logits, raw_words):
    pred_l = preds.tolist()
    logits_l = logits.tolist()
    logits_idx_l = logits.argmax(-1).tolist()
    is_none_l = (1-preds.sum(-1)).tolist()
    raw_words = list(raw_words)

    assert len(pred_l) == len(raw_words)
    result = []
    for is_none, best_id, word in zip(is_none_l, logits_idx_l, raw_words):
        if is_none != 0:
            result.append((word, 'NoneType'))
        else:
            result.append((word, target_vocab.idx2word[best_id]))

    result_logits = []
    for word, logits in zip(raw_words, logits_l):
        result_logits.append([word] + logits)

    return result, result_logits

# 【获取具体的某一个预测结果】
def get_actual_single_pred(pred_logits_list, target_vocab):
    # 【对概率做sigmoid操作】
    avg = torch.Tensor([0.0] * len(target_vocab))
    f_sigmoid = nn.Sigmoid()
    for logits in pred_logits_list:
        avg += f_sigmoid(logits)
    avg /= float(len(pred_logits_list))

    mx, idx = avg.max(-1)
    mx = mx.item()
    idx = idx.item()

    # 【选取预测概率大于0.5的类型，否则返回NoneType】
    if mx >= 0.5:
        return target_vocab.idx2word[idx]
    else:
        return 'NoneType'

# 【将batch的结果存储成k,v字典】
def save_batch_result(pred_dict, batch_item, batch_logits):
    batch_logits_list = batch_logits
    for k, v in zip(batch_item, batch_logits_list):
        if k in pred_dict:
            pred_dict[k].append(v.tolist())
        else:
            pred_dict[k] = [v.tolist()]

# 【获取单轮的预测结果和logits】
def get_single_pred_and_logits(pred_logits_list, target_vocab):
    avg = torch.Tensor([0.0] * len(target_vocab))
    f_sigmoid = nn.Sigmoid()
    for logits in pred_logits_list:
        avg += f_sigmoid(torch.Tensor(logits))
    avg /= float(len(pred_logits_list))

    mx, idx = avg.max(-1)
    mx = mx.item()
    idx = idx.item()

    avg_logits = torch.log(avg / (1.0 - avg))
    avg_logits[avg >= 1.0] = 10000.0
    avg_logits[avg < 0.0] = -10000.0

    if mx >= 0.5:
        return target_vocab.idx2word[idx], avg_logits
    else:
        return 'NoneType', avg_logits

# 【获取最后一轮的预测结果和logits】
def get_final_pred_logits(pred_dict, target_vocab):
    pred = []
    logits = []
    for k, v in pred_dict.items():
        p, l = get_single_pred_and_logits(v, target_vocab)
        pred.append((k, p))
        logits.append([k] + l.tolist())

    return pred, logits

class MyEntityLevelPRMetrics(ClassifyFPreRecMetric):
    def __init__(self, target_vocab):
        self.target_vocab = target_vocab
        super().__init__()

        self.ground_truth = set()
        self.pred_table = {}

    def evaluate(self, item, content, pred, logits, seq_len, target):
        batch_size = pred.size(0)
        for i in range(batch_size):
            cur_words = item[i]
            if cur_words in self.pred_table:
                self.pred_table[cur_words].append(logits[i].cpu())
            else:
                self.pred_table[cur_words] = [logits[i].cpu()]

            s = 0
            for j, label_flag in enumerate(target[i]):
                s += label_flag.item()
                if label_flag.item() != 0:
                    self.ground_truth.add(f"{cur_words}_{self.target_vocab.idx2word[j]}")
            if s == 0:
                self.ground_truth.add(f"{cur_words}_NoneType")

    def get_metric(self, reset=True):
        tp = 0
        for item, logits_list in self.pred_table.items():
            pred_tag = get_actual_single_pred(logits_list, self.target_vocab)

            cur_pair = f"{item}_{pred_tag}"

            if cur_pair in self.ground_truth:
                tp += 1

        p = len(self.pred_table)
        t = len(self.ground_truth)
        print(tp, p, t)
        epsilon = 1e-13
        prec = tp / (p + epsilon)
        rec = tp / (t + epsilon)
        f = 2 * prec * rec / (prec + rec + epsilon)
        if reset:
            self.ground_truth = set()
            self.pred_table = {}
        return {'f': f, 'pre': prec, 'rec': rec}

class BCELossMetrics(MetricBase):
    def __init__(self, target_vocab):
        super().__init__()

        self.val_loss = 0
        self.val_size = 0
        self.multi = len(target_vocab)

    def evaluate(self, logits, target):
        with torch.no_grad():
            loss = nn.BCEWithLogitsLoss(reduction='none')(logits, target.float())
            self.val_loss += torch.sum(loss).float().item()
            self.val_size += (loss.view(-1)).size(0) * self.multi

    def get_metric(self, reset=True):
        val_loss = self.val_loss / self.val_size if self.val_size != 0 else 0
        if reset:
            self.val_loss = 0
            self.val_size = 0
        return {'loss': val_loss}

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
            self.extra_tester = Tester(model=model,
                                     data=extra_test_data,
                                     metrics=metrics,
                                     batch_size=batch_size,
                                     device=None,  # 由上面的部分处理device
                                     verbose=0,
                                     use_tqdm=False)

    def _get_field_str_value(self, d, metric):
        return str(d.get(metric, ''))

    def _update_dict_with_prefix(self, d, data, prefix):
        for k, v in data.items():
            d[f"{prefix}_{k}"] = v

    def on_backward_begin(self, loss):
        if self.step % self.update_every == 0:
            self.cur_stat = {'epoch': self.epoch, 'step': self.step, 'train_loss': loss.item()}
        else:
            self.cur_stat = None

    def on_batch_end(self):
        if ((self._trainer.validate_every > 0 and self.step % self._trainer.validate_every == 0) or
            (self._trainer.validate_every < 0 and self.step % len(self._trainer.data_iterator) == 0)) \
                and self._trainer.dev_data is not None:
            # since it need to do validation, metrics updates are postponed to validation step
            pass
        else:
            if self.cur_stat is not None:
                self.metrics.append(self.cur_stat)
                self.cur_stat = {}

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if self.cur_stat is None:
            self.cur_stat = {'epoch': self.epoch, 'step': self.step}

        for k, v_dict in eval_result.items():
            self._update_dict_with_prefix(self.cur_stat, v_dict, 'val')

        if self.extra_tester is not None:
            test_res = self.extra_tester.test()
            for k, v_dict in test_res.items():
                self._update_dict_with_prefix(self.cur_stat, v_dict, 'test')
            print(f"-----test-----\n{self.extra_tester._format_eval_results(test_res)}\n----------")

        self.metrics.append(self.cur_stat)
        self.cur_stat = {}

    def on_train_end(self):
        with open(self.file_path, 'w', encoding='utf-8') as output:
            output.write('\t'.join(self.field_list))
            output.write('\n')
            for item in self.metrics:
                output.write('\t'.join(self._get_field_str_value(item, f) for f in self.field_list))
                output.write('\n')

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
        self.tag = tag

    def _fetch_metric_value(self, result):
        for k, v_dict in result.items():
            if self.metric_key in v_dict:
                return str(v_dict[self.metric_key])

        return 0

    def _do_validation(self, epoch, step):
        self.callback_manager.on_valid_begin()
        res = self.tester.test()

        is_better_eval = False
        # save all epoch for later reference
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
        self.callback_manager.on_valid_end(res, self.metric_key, self.optimizer, is_better_eval)
        return res

if __name__ == "__main__":

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

    setup_seed(666)
    #torch.backends.cudnn.enabled = False
    #torch.autograd.set_detect_anomaly(True)

    hidden_dim = 1024
    dropout_rate = 0.2

    train_path = args.train_path
    test_path = args.test_path

    label_list = ['药物', '疾病', '症状', '检查科目', '细菌', '病毒', '医学专科']
    # 【这里采用了不同的bert分词器】
    bert_tokenizer = MyBertContentTokenizer(label_list, args.bert_model_name)

    model = BertEntityTypeWithContentClassifier(
        bert_tokenizer.target_vocab,
        args.bert_model_name,
        hidden_dim,
        dropout_rate,
        use_ll_cls=False,
        bert_st=-3,
        bert_ed=-1
    )

    if args.train==1:
        print(f"train file: {train_path}")
        train_data, val_data, test_data = MyDataContentLoader.getTrainData(train_path)

        # random.shuffle(total_data)
        bert_tokenizer.dataProcessor(train_data, 'train')
        bert_tokenizer.dataProcessor(val_data, 'dev')
        if test_data:
            bert_tokenizer.dataProcessor(test_data, 'test')

        data_set = bert_tokenizer.getDataSet()

        print(f"train_set: {len(train_data)}\tdev_set: {len(val_data)}\ttest_set: {len(test_data)}")

        metrics = [
            MyEntityLevelPRMetrics(data_set.get_vocab('target')),
            BCELossMetrics(data_set.get_vocab('target'))
        ]

        bert_param_ids = list(map(id, model.bert_encoder.parameters()))
        rest_params = filter(lambda x: id(x) not in bert_param_ids, model.parameters())

        hierarchical_lr = [{'params': model.bert_encoder.parameters(), 'lr': args.bert_lr},
                           {'params': rest_params, 'lr': args.mlp_lr},
                           ]

        if args.optim == 'sgd':
            optimizer = optim.SGD(hierarchical_lr, momentum=0.9)
        else:
            optimizer = optim.Adam(hierarchical_lr, amsgrad=False)

        tag = f"model_{args.run_tag}"
        check_level = 0

        callbacks = [
            GradientClipCallback(clip_type='norm', clip_value=5.0), # clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
            #WarmupCallback(0.1, schedule='linear'),
            MyMetricsHandler(model, data_set.get_dataset('test') if test_data else None, metrics, args.batch_size, tag)
        ]
        print(data_set.get_dataset('train'))

        trainer = MyCustomizedTrainer(data_set.get_dataset('train'), model, optimizer, batch_size=args.batch_size, update_every=args.update_every,
                          n_epochs=args.epochs, dev_data=data_set.get_dataset('dev'), metrics=metrics,
                          dev_batch_size=args.batch_size, callbacks=callbacks, device=args.device, test_use_tqdm=False,
                          use_tqdm=False, print_every=100, metric_key='f',check_code_level=check_level,
                          save_path=f'./saved/{tag}', tag=tag)
        trainer.train(load_best_model=False)

    elif args.train == 2:
        print(f"test file: {test_path}")
        test_data, test_items = MyDataContentLoader.getTestData(test_path)
        bert_tokenizer.testDataProcessor(test_data, 'test')
        data_set = bert_tokenizer.getDataSet()
        test_set = data_set.get_dataset('test')

        model.load_state_dict(torch.load(args.model_path, map_location={'cuda:1': args.device} ))
        model.to(torch.device(args.device))
        model.eval()

        output_results = f"./results_{args.test_id}.txt"
        output_logits = f"./logits_{args.test_id}.txt"
        print(f"output results: {output_results} logits: {output_logits}")

        num_data = len(test_set)
        print(f"in total {len(test_items)} items with {num_data} lines")
        i = 0
        batch_size = args.batch_size
        print_every = 500
        next_print = print_every

        pred_dict = {}
        while i < num_data:
            if i >= next_print:
                print(i)
                next_print += print_every
            j = min(i + batch_size, num_data)
            batch_item = test_set[i:j]['item']
            batch_x = torch.tensor(test_set[i:j]['chars']).view(j - i, -1).to(args.device)
            batch_mask = torch.tensor(test_set[i:j]['item_mask']).view(j - i, -1).to(args.device)

            pred_result = model.predict(batch_x, batch_mask)

            # batch_pred = pred_result['pred'].view(j - i, -1)
            batch_logits = pred_result['logits'].view(j - i, -1)

            save_batch_result(pred_dict, batch_item, batch_logits)

            i = j

        # merge results
        pred, logits = get_final_pred_logits(pred_dict, data_set.get_vocab('target'))

        with open(output_results, 'w', encoding='utf-8') as predict_file:
            for k, v in pred:
                predict_file.write(f"{k}\t{v}\n")

        with open(output_logits, 'w', encoding='utf-8') as logits_file:
            for item in logits:
                logits_file.write('\t'.join([str(term) for term in item]))
                logits_file.write('\n')
