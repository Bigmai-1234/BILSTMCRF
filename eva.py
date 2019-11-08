import bilstmCRF.util as util
import os
import re
import torch
from bilstmCRF.bistmCRF import BiLSTM_CRF
import codecs

EMBEDDING_DIM = util.EMBEDDING_DIM
HIDDEN_DIM = util.HIDDEN_DIM
TRAIN_DATA_PATH = util.TRAIN_DATA_PATH
MODEL_PATH = util.MODEL_PATH
NEWS_PATH = util.NEWS_PATH


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

def cut_sentence(data):
    sentLst = data.split("。")
    return sentLst

_, word_to_ix, tag_to_ix = util.data_prepare(TRAIN_DATA_PATH)
OrderedDict = torch.load(os.path.join(os.getcwd(),"model"))
len_ = len(OrderedDict["word_embeds.weight"])

model = BiLSTM_CRF(len_, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


ix_to_tag = {v: k for k, v in tag_to_ix.items()}
# test_sent = '有什么是周敏和王维去了北京和毛里求斯才知道的事情'
# test_id = [word_to_ix[i] if i in word_to_ix.keys() else 2 for i in test_sent]
# test_res_ = torch.tensor(test_id,dtype=torch.long)
# eval_res = [ix_to_tag[t] for t in model(test_res_)[1]]
# #['O', 'O', 'O', 'O', 'B_PERSON', 'E_PERSON', 'O', 'B_PERSON', 'E_PERSON', 'O', 'O', 'B_LOCATION', 'E_LOCATION', 'O', 'B_LOCATION', 'M_LOCATION', 'M_LOCATION', 'E_LOCATION', 'O', 'O', 'O', 'O', 'O', 'O']



for parent, dirnames, filenames in os.walk(NEWS_PATH):
    for filename in filenames:
        print("transforming.....", filename)
        file_path_in = os.path.join(parent, filename)
        file_path_out = os.path.join(parent ,r"..\\NERout\\ner_{}".format(filename))
        test_d = codecs.open(file_path_in, 'r', 'utf-8').read()
        test_d = strQ2B(test_d)
        test_d = re.sub(r"\n\n","",test_d)
        test_d_Lst = cut_sentence(test_d)
        result = set()

        for sent in test_d_Lst:
            if len(sent) < 4: continue
            test_id = [word_to_ix[i] if i in word_to_ix.keys() else 2 for i in sent]
            test_res_ = torch.tensor(test_id, dtype=torch.long)
            eval_res = [ix_to_tag[t] for t in model(test_res_)[1]]
            tag2word_res = util.tag2word(sent,eval_res)
            for s in tag2word_res:
                result.add(s)

        with codecs.open(file_path_out, 'w+', 'utf-8') as surveyp:
            surveyp.write(",\n".join(result))

