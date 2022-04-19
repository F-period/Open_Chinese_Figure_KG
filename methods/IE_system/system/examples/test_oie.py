from ..oie import OIE

oie = OIE()
# test one data
line = {"text": "印度空军参谋长阿尔琼也提防巴空军的“决定性行动”，并且他致电帕赞科特基地司令苏里上校"}
line = {"text": "中美两国的人民反对大规模的杀伤性的武器"}
line = {"id": "6", "sysId": "eb88374b30fda925b399e787a927327c", "text": "13日，冲绳和平运动中心组织了2800名冲绳县民，到驻冲绳美军普天间基地周边举行抗议集会。", "event_list": [{"event_type": "举办类", "trigger": "举行", "trigger_start_index": "38", "trigger_end_index": "40", "trigger_entity_type": "NONE", "arguments": [{"role": "会议", "argument": "抗议集会", "argument_start_index": "40", "argument_end_index": "44", "argument_entity_type": "Meeting"}, {"role": "地点", "argument": "普天间基地", "argument_start_index": "31", "argument_end_index": "36", "argument_entity_type": "ZBGC"}, {"role": "时间", "argument": "13日", "argument_start_index": "0", "argument_end_index": "3", "argument_entity_type": "Time"}, {"role": "主体", "argument": "冲绳和平运动中心", "argument_start_index": "4", "argument_end_index": "12", "argument_entity_type": "Org"}]}]}
sample = line
result, quick_look = oie.extract(sample, True, True, True)
print(result)
# s += len(result)
# opobj.write(str(result) + "\n")
# opobj2.write(str(quick_look) + "\n")
# print(s)
# opobj.close()
# opobj2.close()