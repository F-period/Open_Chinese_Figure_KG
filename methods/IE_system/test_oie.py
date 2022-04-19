import os

from tqdm import tqdm

from system.krl import KRL
from system.utils.format import format_data
from system.oie import OIE


# extract one file
def execute_file(input_fp, output_fp):
    oie = OIE()
    oie.extract_file(input_fp, output_fp)


# extract one sentence
def execute_sentence():
    oie = OIE()
    # test one data
    line = {"text": "印度空军参谋长阿尔琼也提防巴空军的“决定性行动”，并且他致电帕赞科特基地司令苏里上校"}
    line = {"text": "中美两国的人民反对大规模的杀伤性的武器"}
    line = {"id": "6",
            "sysId": "eb88374b30fda925b399e787a927327c",
            "text": "乔治·塞菲里斯，生于小亚细亚的斯弥尔纳城，父亲是雅典大学教授，国际法专家。",
            "event_list": [
                {"event_type": "举办类", "trigger": "举行", "trigger_start_index": "38", "trigger_end_index": "40",
                 "trigger_entity_type": "NONE", "arguments": [
                    {"role": "会议", "argument": "抗议集会", "argument_start_index": "40", "argument_end_index": "44",
                     "argument_entity_type": "Meeting"},
                    {"role": "地点", "argument": "普天间基地", "argument_start_index": "31", "argument_end_index": "36",
                     "argument_entity_type": "ZBGC"},
                    {"role": "时间", "argument": "13日", "argument_start_index": "0", "argument_end_index": "3",
                     "argument_entity_type": "Time"},
                    {"role": "主体", "argument": "冲绳和平运动中心", "argument_start_index": "4", "argument_end_index": "12",
                     "argument_entity_type": "Org"}]}]}

    sample = line['text']
    result, quick_look = oie.extract(sample, True, True, True)
    print(quick_look)
    # s += len(result)
    # opobj.write(str(result) + "\n")
    # opobj2.write(str(quick_look) + "\n")
    # print(s)
    # opobj.close()
    # opobj2.close()


def clean_triples(train_fp, output_fp, is_train: bool):
    krl = KRL()
    model_type = 'TransE'

    if is_train:
        model_type = 'TransE'
        krl.train(train_fp, model_type=model_type, dev_path=train_fp, save_path='./krl_{}_saves'.format(model_type))
    else:
        krl.load(save_path='./krl_{}_saves'.format(model_type), model_type=model_type)


if __name__ == "__main__":
    # 1 extract the triples
    # eg:{"id": "870", "sysId": "3669195fb557cea411d166d353cc194d",
    # "text": "目前，黎以临时边界“蓝线”沿线，特别是靠近叙利亚戈兰高地的地段局势紧张，黎以军队和联合国驻黎巴嫩南部临时部队(联黎部队)都处于高度戒备状态，以应对以色列空袭叙利亚可能引发的军事冲突。",
    # "event_list": [{"event_type": "军事冲突类", "trigger": "空袭", "trigger_start_index": "76", "trigger_end_index": "78", "trigger_entity_type": "$element$", "arguments": [{"role": "主体", "argument": "以色列", "argument_start_index": "73", "argument_end_index": "76", "argument_entity_type": "Country"}, {"role": "目标", "argument": "叙利亚", "argument_start_index": "78", "argument_end_index": "81", "argument_entity_type": "Country"}]}]}
    # -> [['南部临时部队(联黎部队)', '处于', '高度戒备状态'], ['以色列', '空袭', '叙利亚']]

    input_file_path = 'data/all_data.json'
    triples_file_path = 'result/1_after_extract.txt'
    # execute_file(input_file_path, triples_file_path)

    # 2 clean the triples
    # transform the data format
    # [['南部临时部队(联黎部队)', '处于', '高度戒备状态'], ['以色列', '空袭', '叙利亚']] ->
    # 南部临时部队(联黎部队), 处于, 高度戒备状态
    # 以色列, 空袭, 叙利亚
    formatted_fp = 'result/1_after_extract_formatted.txt'
    format_data(triples_file_path, formatted_fp)

    # using Knowledge Relation Learning (KRL) to score the triples
    cleared_file_path = 'result/2_cleared_extract.txt'
    clean_triples(train_fp=formatted_fp, output_fp=cleared_file_path, is_train=True)
