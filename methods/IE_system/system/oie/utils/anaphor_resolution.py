
# 指代消解
def anaphor_resolution(verb):
    for child in verb.children:
        # if child.dep == conj or verb.dep_ == "nmod:prep":
        if child.dep_ == "nmod:topic":
            return child