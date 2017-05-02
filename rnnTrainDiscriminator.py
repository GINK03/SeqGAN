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


def build_model():
  model = Sequential()
  model.add(GRU(128, return_sequences=False, input_shape=(128, 512)))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  optimizer = Adam()
  model.compile(loss='binary_crossentropy', optimizer=optimizer) 
  return model

def train():
  with open("./tmp/step4.pkl", "rb") as f:
    char_vec = pickle.loads(f.read())

  with open("./tmp/char_index.pkl", "rb") as f:
    char_index = pickle.loads(f.read())
    index_char = { index:char for char, index in char_index.items() }

  Xs, Ys = [], []
  print("start to load data to memory...")
  target_file = sorted(glob.glob("tmp/sample_*.txt"))[-1]
  with open(target_file, "r") as f:
    for fi, line in enumerate(f):
      if fi%1000 == 0:
        print("now iter ", file=sys.stderr)
      line    = line.strip()
      print(line)
      y, sent = line.split("__SEP__")
      ys      = float(y)
      chars   = list(sent)
      
      """ 128字にカット """
      chars   = chars[:128]

      """ 128字に満たないのはパッディングする """
      b = ["*"]*(128 - len(chars) )
      b += chars
      chars = b
      print(len(chars))
      print(chars)
      try:
        xs = list( map(lambda x:char_vec[x], chars)  )
      except KeyError as e:
        continue

      Xs.append(xs)
      Ys.append(ys)
  print("finished heaping memory...")
  Xs, Ys = np.array(Xs), np.array(Ys)
  print("try to create model...")
  model = build_model()
  model.fit(Xs, Ys, batch_size=64, epochs=20)
  model.save("{target_file}.disc.model".format(target_file=target_file))


  """ 判別値を出力する """
  preds = []
  with open(target_file, "r") as f:
    for fi, line in enumerate(f):
      line    = line.strip()
      y, sent = line.split("__SEP__")
      ys      = float(y)
      chars   = list(sent)
      
      if ys != 0.0:
        continue

      """ 128字にカット """
      chars   = chars[:128]

      """ 128字に満たないのはパッディングする """
      b = ["*"]*(128 - len(chars) )
      b += chars
      chars = b
      
      xs = list( map(lambda x:char_vec[x], chars)  )

      pred = model.predict(np.array([xs]))[0].tolist()[0]
      print(pred)
      preds.append(pred)

  """ 報酬を計算してファイルに出力 """
  maxProb = max(preds)
  preds   = list(map(lambda x:x/maxProb, preds))
  with open(target_file, "r") as f, open("tmp/reinforce.txt", "w") as t:
    for fi, line in enumerate(f):
      line    = line.strip()
      y, sent = line.split("__SEP__")
      ys      = float(y)
      chars   = list(sent)
      
      if ys != 0.0:
        continue
      """ 128字にカット """
      chars   = chars[:128]
      score   = preds.pop(0)
      save = "%s__SEP__%s"%( str(score), " ".join(chars) )
      t.write(save + "\n")



train()
