def format_data(input_fp, output_fp):
    """
    Transform the data format,
    eg:
    [['南部临时部队(联黎部队)', '处于', '高度戒备状态'], ['以色列', '空袭', '叙利亚']] ->
    南部临时部队(联黎部队), 处于, 高度戒备状态
    以色列, 空袭, 叙利亚
    """

    with open(output_fp, 'w', encoding='utf8') as output_f:
        with open(input_fp, 'r', encoding='utf-8') as input_f:
            for line in input_f.readlines():
                for each in eval(line):
                    output_f.write(",".join(each) + '\n')


if __name__ == "__main__":
    format_data()
