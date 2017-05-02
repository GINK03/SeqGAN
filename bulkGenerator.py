
from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.core import Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization as BN
from keras.callbacks import LearningRateScheduler as LRS
import numpy as np
import random
import sys
import glob
import pickle
import re
import datetime

def sample(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

""" 一括で評価に使うデータセットを生成する """
def bulkGenerate():
  now   = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
  S_NUM = 1000
  with open("./tmp/step4.pkl", "rb") as f:
    char_vec = pickle.loads(f.read())

  with open("./tmp/char_index.pkl", "rb") as f:
    char_index = pickle.loads(f.read())
    index_char = { index:char for char, index in char_index.items() }
   
  model_name = sorted(glob.glob("models/*.model"))[-1]

  model = load_model(model_name)
  
  negatives = []
  for it in range(S_NUM):
    sent  = ["*"]*5
    diversity = 1.0
    next_char = None
    negative  = []
    for i in range(100):
      if next_char is None or next_char != "*":
        xs         = list(map(lambda x:char_vec[x], sent))
        preds      = model.predict(np.array([xs]), verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char  = index_char[next_index]
        sent.append(next_char)
        sent = sent[1:] 
        sys.stdout.write(next_char)
        sys.stdout.flush()
        negative.append(next_char)
        if next_char == "*":
          ...
    print()
    negatives.append( ("0", "".join(negative).replace("*", "")) )

  positives = []
  ps = list(filter(lambda x:x!="", open("./tmp/step2", "r").read().split("\n")))
  random.shuffle(ps)
  for p in ps:
    if "#" in p: continue
    if len(p.split()) < 50: continue
    if len(positives) > S_NUM:
      continue
    if len(re.findall(r"[a-z]", p)) > 20:
      continue
    p = p.replace("*", "").replace(" ", "")
    positives.append( ("1", p) )
    #print(p)
      
  target = negatives
  target.extend(positives)
  random.shuffle(target)
  with open("./tmp/sample_{now}.txt".format(now=now), "w") as f:
    for t in target:
      print("__SEP__".join(list(t)))
      f.write( "__SEP__".join(list(t)) + "\n" )


if __name__ == '__main__':
  bulkGenerate()
  
