def get_ISA_triple(token):
    # 提取ISA关系
    site = token.i
    extra_words = {}
    childs = [token.children]
    while childs:
        buf = []
        for child in childs:
            for word in child:
                # if word.dep_ in ['nummod']: continue
                buf.append(word.children)
                # 直接加上所有子句
                # if word.dep_ == 'compound:nn' or word.dep_ == 'amod':
                if word.text != '也':
                    extra_words[word.i] = word
        childs = buf
    l, r = site - 1, site + 1
    res_text = token.text
    while True:
        if l in extra_words:
            res_text = extra_words[l].text + res_text
            l -= 1
        else:
            break
    tmp2 = [token.head.text, '是', res_text]
    return tmp2
