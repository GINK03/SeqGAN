from __future__ import print_function
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, SimpleRNN, Input, RepeatVector
from keras.layers.core import Dropout, Lambda
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.engine.topology import Layer
from keras import backend as K
from keras.callbacks import LearningRateScheduler as LRS
import numpy as np
import random
import sys
import glob
import pickle
import re

WIDTH       = 2029
MAXLEN      = 31
INPUTLEN    = 20
ACTIVATOR   = 'relu'
DO          = Dropout(0.1)
inputs      = Input( shape=(INPUTLEN, ) ) 
repeat      = RepeatVector(31)(inputs)
generated   = Bi( GRU(512, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )( repeat )
generated   = TD( Dense(2049, kernel_initializer='lecun_uniform', activation='sigmoid') )( generated )
generated   = Lambda( lambda x:x*2.0 )(generated)
generator   = Model( inputs, generated )

generator.compile( optimizer=Adam(), loss='categorical_crossentropy' )

def train():
  for name in glob.glob('utils/dataset_*.pkl'):
    try:
      dataset = pickle.loads( open(name, 'rb').read()  )
    except EOFError as e:
      continue
    xs, ys  = dataset
    # - print( xs.shape )
    # - print( ys.shape )
    generator.fit( xs, ys, epochs=100)


def main():
  if '--train' in sys.argv:
     train()
  if '--eval' in sys.argv:
     eval()
if __name__ == '__main__':
  main()
