

"""
该脚本将人民日报数据 语料合并
主要合并 t ,nr , ns, nt
1、将分开的姓和名合并
2、将[]中的大粒度词合并
3、将时间合并
4、将全角字符统一转为半角字符
"""

RMRB_DATA_PATH = r"D:\DEV\businessInfomationProject\ChineseNER\data\rmrb.txt"
import re
import pandas as pd
from rmrb_BMEWO import BMEWO



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

def deal_(text_line):
    # 合并中括号里面的内容
    blacketLst = re.findall("\[.*?\]", text_line)
    if blacketLst:
        for b in blacketLst:
            b_trans = "".join([i.split("/")[0] for i in b[1:-1].split("  ")]) + "/"
            text_line = text_line.replace(b, b_trans)
    # 将姓名空格去除
    text_line = re.sub("(\w+)/nr  (\w+)/nr", r"\1\2/nr", text_line)
    # 将时间空格去除
    text_line = re.sub("(\w+)/t  (\w+)/t", r"\1\2/t", text_line)
    text_line = re.sub("(\w+)/t  (\w+)/t  (\w+)/t", r"\1\2\3/t", text_line)
    text_line = re.sub("(\w+)/t  (\w+)/t  (\w+)/t  (\w+)/t", r"\1\2\3\4/t", text_line)

    #去除（/w ）/w
    text_line = re.sub("（/w  ","",text_line)
    text_line = re.sub("  ）/w", "", text_line)
    return text_line


file =  open(RMRB_DATA_PATH,'r',encoding="utf8")

char4train = []
tag4train = []
res_df = pd.DataFrame()

try:
    while True:
        text_line_ = file.readline()
        if text_line_:
            text_line_ = text_line_[23:]
            #分句
            text_line_cut_Lst = text_line_.split("。/w  ")

            for text_line in text_line_cut_Lst:
                #去除最后的标点符号
                text_line = text_line.strip()
                if text_line[-2:] == "/w":
                    text_line = text_line[:-3]
                if len(text_line) > 20 :
                    text_line = deal_(text_line)
                    print(text_line)
                    text_line = strQ2B(text_line)

                    charLst, tagLSt = BMEWO(text_line)
                    char4train.append(charLst)
                    tag4train.append(tagLSt)

        else:
            break
finally:
    file.close()

res_df['char'] = char4train
res_df['tag'] = tag4train

res_df.to_csv(r"D:\DEV\businessInfomationProject\ChineseNER\data\rmrb4train.csv",encoding='gbk')

























