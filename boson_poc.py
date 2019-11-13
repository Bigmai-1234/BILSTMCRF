#处理boson数据，将其与人民日报数据整合训练
"""主要提取：
time
company_name
location
org_name
person_name

"""
import re
#去除字符


#全角变半角
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

#抽取文章标注，返回文章 {原文标注：(实体类型，实体内容)}
def entityExtract(text):
    w_t_dict = {'time':'_TIME','person_name':'_PERSON','location':'_LOCATION','org_name':'_ORGANIZATION','company_name':'_ORGANIZATION','product_name':'O'}
    extDict = {}
    extRes = re.findall(r'{{[a-z].*?}}',text)
    for ext in set(extRes):
        entityName = ext[2:-2].split(":")[0]
        entityName = w_t_dict[entityName]
        entityText = ext[2:-2].split(":")[1]
        extDict[ext] = (entityName,entityText)
    return extDict


def fileRead(PATH):
    with open(PATH, encoding="utf-8") as f:
        return f.read()
def cleanSent(sent):
    sent = re.sub(r"\\n","",sent)
    sent = re.sub(r"\\", "", sent)
    sent = re.sub(r"\n", "", sent)
    sent = re.sub(r"\xa0", "", sent)
    sent = re.sub(r" ", "", sent)
    sent = re.sub(r"\)", "", sent)
    sent = re.sub(r"\(", "", sent)
    sent = re.sub(r"\+", "", sent)
    sent = re.sub(r"\t", "", sent)
    return sent

def deal_index_tag(sent,extDict):
    sent_ = sent+"".strip()
    index_tag = {}
    pcs = re.findall(r'{{[a-z].*?}}', sent_)
    if pcs:
        for pc in pcs:
            pctext = extDict[pc][1]
            if extDict[pc][0] == "O":
                pctag = ["O" for _ in pctext]
            else:
                pctag = ["M" + extDict[pc][0] for _ in pctext]
                pctag[0] = "B" + extDict[pc][0]
                pctag[-1] = "E" + extDict[pc][0]

            # 替换标注
            sent_ = re.sub(pc, pctext, sent_)
            index_tag[pctext] = pctag

    else:
        return sent,{}
    return sent_,index_tag

def deal(text,extDict):
    char = []
    tag = []
    sents = text.split("。")
    for sent in sents:
        sent = cleanSent(sent)
        if len(sent) > 4:
            #如果原文有标注
            sent_tmp,index_tag = deal_index_tag(sent,extDict)
            sentLst = [c for c in sent_tmp]
            tagLst = ["O" for _ in sent_tmp]


            if index_tag:
                #实体和标注，key 为实体
                for key in index_tag.keys():
                    find = re.finditer(key,sent_tmp)
                    #若找到实体
                    for f in find:
                        indx = f.span()
                        tagLst[indx[0]:indx[1]] = index_tag[key]



            assert len(sentLst) == len(tagLst)
            char.append(sentLst)
            tag.append(tagLst)

    return char,tag


if __name__ == '__main__':
    PATH = r"D:\DEV\businessInfomationProject\ChineseNER\data\BosonNLP.txt"
    import pandas as pd
    df = pd.DataFrame()
    text = fileRead(PATH)
    text = strQ2B(text)
    text = cleanSent(text)
    extDict = entityExtract(text)
    charLst,tagLst = deal(text,extDict)

    df['char'] = charLst
    df['tag'] = tagLst

    df.to_csv(r"D:\DEV\businessInfomationProject\ChineseNER\data\boson4train.csv")








