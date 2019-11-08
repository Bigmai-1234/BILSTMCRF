#将人工标注转化为BMEWO标注
def BMEWO(sent):
    w_t_dict = {'t':'_TIME','nr':'_PERSON','ns':'_LOCATION','nt':'_ORGANIZATION'}
    charLst = []
    tagLSt = []

    sent_split_lst = sent.strip().split("  ")
    for w in sent_split_lst:
        if "/" in w:
            ww = w.split("/")[0]
            wtag = w.split("/")[1]
            ww2char = [c for c in ww]
            charLst.extend(ww2char)
            #获得词的长度
            wlen = len(ww)
            #初始化标注
            #其他标注成'O'
            BMEWO_tag = ['O' for _ in ww]

            #如果是['t','nr','ns','nt'] 标注
            if wtag in ['t','nr','ns','nt']:
                BMEWO_tag[0] = 'B' + w_t_dict[wtag]
                BMEWO_tag[-1] = 'E' + w_t_dict[wtag]
                BMEWO_tag[1:-1] = ['M' + w_t_dict[wtag] for _ in BMEWO_tag[1:-1]]
            tagLSt.extend(BMEWO_tag)
        else:
            continue
    return charLst,tagLSt