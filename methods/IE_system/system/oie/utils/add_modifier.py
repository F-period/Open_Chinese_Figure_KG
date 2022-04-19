# 主语、宾语添加修饰成分
def add_modified_words(token):
    extra_words = {}
    subject_site = token.i
    index = token.idx
    childs = [token.children]
    while childs:
        buf = []
        for child in childs:
            for word in child:
                if word.dep_ in ['acl', 'nummod', 'appos']: continue
                buf.append(word.children)
                # 直接加上所有子句
                # if word.dep_ == 'compound:nn' or word.dep_ == 'amod':
                if word.text != '也':
                    extra_words[word.i] = word
        childs = buf
    # 筛除出直接相连的词，得到最终的主语短语
    l, r = subject_site - 1, subject_site + 1
    # res_sub or res_obj
    res_ = token.text
    while True:
        if l in extra_words:
            res_ = extra_words[l].text + res_
            index = extra_words[l].idx
            l -= 1
        else:
            break
    while True:
        if r in extra_words:
            res_ = res_ + extra_words[r].text
            r += 1
        else:
            break
    return res_, index  # 返回带修饰的词 和 词的首位置
