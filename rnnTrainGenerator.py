
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

def sample(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

with open("./tmp/step4.pkl", "rb") as f:
  char_vec = pickle.loads(f.read())

with open("./tmp/char_index.pkl", "rb") as f:
  char_index = pickle.loads(f.read())
  index_char = { index:char for char, index in char_index.items() }

if '--init' in sys.argv:
  model = load_model("models/000000029.model")
else: 
  model = load_model("reinforceModels/reinfoce.model")


""" バリアブルラーニングレートを設定して、動的に報酬を反映する """
with open("tmp/reinforce.txt", "r") as f:
  for fi, line in enumerate(f):
    print(line)
    line = line.strip()
    score, raw = line.split("__SEP__")

    """ padding """
    raw        = ["*"]*5 + raw.split() + ["*"]
    score      = float(score)
    chars      = raw

    """ windowを作って、学習する """
    xs = []
    ys = []
    for i in range(0, len(chars) - 5 ):
      xs.append( list(map(lambda x:char_vec[x], chars[i:i+5])) )
      y = [0.]*len(char_index)
      y[char_index[chars[i+5]]] = 1.0
      ys.append( y )
      print( chars[i:i+5], chars[i+5] ) 

    model.optimizer.lr = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=score/1000.0))

    model.fit(xs, ys, batch_size=64, epochs=5)
    model.save("reinforceModels/reinfoce.model")

