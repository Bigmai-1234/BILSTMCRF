import torch
import pandas as pd
import os
import re

START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK_TAG = "<UNK>"

EPOCHES = 300
EMBEDDING_DIM = 500
HIDDEN_DIM = 500
BATCH_SIZE = 64
lr = 0.0001

TRAIN_DATA_PATH = r"D:\DEV\businessInfomationProject\ChineseNER\data\rmrb4train.csv"
MODEL_PATH = os.path.join(os.getcwd(), "model")


NEWS_PATH = r"D:\DEV\businessInfomationProject\TextRank4ZH_v2\news\newsin"


def data_prepare(TRAIN_DATA_PATH):
    # training data
    df = pd.read_csv(TRAIN_DATA_PATH, encoding='gbk', index_col=0)
    training_data = list(zip(df.char, df.tag))

    word_to_ix = {}
    for sentence, _ in training_data:
        for word in eval(sentence):
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {START_TAG: 0, STOP_TAG: 1, UNK_TAG: 2, 'O': 3}
    tag_to_ix_tmp = dict(
        zip([a + b for a in ['B', 'M', 'E'] for b in ['_TIME', '_PERSON', '_LOCATION', '_ORGANIZATION']], range(4, 16)))
    tag_to_ix.update(tag_to_ix_tmp)

    return training_data,word_to_ix,tag_to_ix

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w not in to_ix.keys():
            w = UNK_TAG
        idxs.append(to_ix[w])
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# for eval
def tag2word(sent,tagLst):
    res = []
    for i,t in enumerate(tagLst):
        if t != 'O':
            tagLst[i] = sent[i] + tagLst[i]
        else:
            tagLst[i] = '|'
    tmp = "".join(tagLst)
    tmp = set(tmp.split("|"))

    for w in tmp:
        if w:
            w_w = re.findall("[0-9]?[\u4e00-\u9fa5]+",w)
            w_w = "".join(w_w)
            w_t = w.split("_")[-1]
        else:
            continue
        res.append(w_w + " " + w_t)

    return res
