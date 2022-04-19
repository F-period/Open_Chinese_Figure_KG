from abc import ABC

from .utils.add_modifier import add_modified_words
from .utils.anaphor_resolution import anaphor_resolution
from .utils.get_ISA import get_ISA_triple
from ..base.module import Module
import os
import spacy as sp
from spacy import displacy
from spacy.symbols import nsubj, VERB, dobj, conj


class OIE(Module, ABC):
    """
    This class try to do open information extraction from raw text.
    """

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        sp.require_gpu()

        self.nlp = sp.load("zh_core_web_trf")

    def extract(self, sample, is_add_modifier=False, is_anaphor_resolution=False, is_extract_ISA=False):
        # nlp.add_pipe("merge_entities")
        doc = self.nlp(sample["text"])
        quick_look = list()
        triple_list = list()
        for possible_subject in doc:
            # 第一类：抽取三元组关系
            # 如果一个词是名词性主语，同时head为动词，则可以抽出一个三元组
            if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
                for possible_object in possible_subject.head.children:
                    if possible_object.dep == dobj:
                        triple = [possible_subject, possible_subject.head, possible_object]

                        if is_anaphor_resolution:
                            # 保存前进行指代消解
                            if possible_subject.text in ['他', '她', '它', '他们', '她们', '它们', '这', '那']:
                                subject = anaphor_resolution(possible_subject.head)
                                if subject is not None:
                                    triple = [subject, possible_subject.head, possible_object]
                                else:
                                    triple = [possible_subject, possible_subject.head, possible_object]
                            elif possible_object.text in ['他', '她', '它', '他们', '她们', '它们', '这', '那']:
                                object = anaphor_resolution(possible_object.head)
                                if object is not None:
                                    triple = [possible_subject, possible_subject.head, object]
                                else:
                                    triple = [possible_subject, possible_subject.head, possible_object]
                            else:
                                triple = [possible_subject, possible_subject.head, possible_object]

                        head_idx = possible_subject.idx
                        tail_idx = possible_object.idx
                        if is_add_modifier:
                            # 为主语词加上相连的复合词与形容词修饰语
                            modifier_subject, head_idx = add_modified_words(possible_subject)
                            # 为宾语添加复合词与形容词修饰语
                            modifier_object, tail_idx = add_modified_words(possible_object)
                            triple[0] = modifier_subject
                            triple[2] = modifier_object

                        quick_look.append([str(i) for i in triple])

                        tmp = {"head": triple[0],
                               "head_span": (head_idx, head_idx + len(triple[0])),
                               "relation": triple[1],
                               "relation_span": (
                                   triple[1].idx, triple[1].idx + len(triple[1])),
                               "tail": triple[2],
                               "tail_span": (tail_idx, tail_idx + len(triple[2])),
                               }
                        triple_list.append(tmp)

            # 第二类：提取ISA关系
            if is_extract_ISA:
                if possible_subject.dep_ == 'appos' and possible_subject.head.pos_ == 'PROPN':
                    quick_look.append(get_ISA_triple(possible_subject))
        result = dict(id=sample["id"], sysId=sample["sysId"], text=sample["text"], quick_look=quick_look,
                      triple_list=triple_list)
        # result = quick_look
        return result, quick_look
