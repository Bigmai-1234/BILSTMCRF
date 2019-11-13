import pandas as pd
import os

PATH  = r"D:\DEV\businessInfomationProject\ChineseNER\data"

rmrb  = pd.read_csv(os.path.join(PATH,'rmrb4train.csv'),encoding='gbk',index_col=0)
boson = pd.read_csv(os.path.join(PATH,'boson4train.csv'),index_col=0)

df = pd.concat([rmrb,boson])
df = df.sample(frac=1)

df.to_csv(r"D:\DEV\businessInfomationProject\ChineseNER\data\rmrb_boson_train.csv")