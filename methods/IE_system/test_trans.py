from system.krl import KRL
from tqdm import tqdm

train_path = 'train.csv'
dev_path = 'val.csv'
model_type = 'TransE'

krl = KRL()
# krl.train(train_path, model_type=model_type, dev_path=train_path, save_path='./krl_{}_saves'.format(model_type))
# #
krl.load(save_path='./krl_{}_saves'.format(model_type), model_type=model_type)
# # krl.test(train_path)

import pandas as pd
import numpy as np

data = pd.read_csv('all.csv')
data = np.array(data).tolist()
pred = []
for i in tqdm(data):
    try:
        score = krl.predict(head=i[0], rel=i[1], tail=i[2])
    except:
        score = -1
    i.append(score)
    pred.append(i)

pd.DataFrame(pred).to_csv('pred_all.csv', header=None, index=None)
