from __future__ import print_function
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, SimpleRNN, Input, RepeatVector
from keras.layers.core import Dropout, Lambda, Flatten, Reshape
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
INPUTLEN    = 1000
inputs      = Input( shape=(INPUTLEN, ) ) 
#enc         = Dense(2024, activation='linear')( inputs )
#enc         = Dense(1024, activation='tanh')( enc )
repeat      = RepeatVector(31)( inputs )
generated   = Bi( GRU(256, return_sequences=True) )( repeat )
generated   = TD( Dense(2049, activation='relu') )( generated )
generated   = TD( Dense(2049, activation='softmax') )( generated )
generator   = Model( inputs, generated )

#generated   = Lambda( lambda x:x*2.0 )(generated)

adhoc       = Model( inputs, generated )
adhoc.compile( optimizer=Adam(), loss='categorical_crossentropy' )

def train():
  for ge, name in enumerate( glob.glob('utils/dataset_*.pkl')[:1] ):
    dataset = pickle.loads( open(name, 'rb').read()  )
    xs, ys  = dataset
    for e in range(10):
      adhoc.fit( xs, ys, epochs=50 )
      adhoc.save_weights('models/adhoc_seed_%09d.h5'%e)
# - 偽のデータセットを作成する
def gen():
  term_index = pickle.loads( open('utils/term_index.pkl', 'rb').read() ) 
  index_term = { index:term for term, index in term_index.items() }
  adhoc.load_weights( 'models/adhoc_seed_000000003.h5' )
  xs = np.random.randn(10000, 1000) 
  res = adhoc.predict(xs)
  print( res.shape )
  with open('negative.data.txt', 'w') as f: 
    for e, rs in enumerate( res.tolist() ):
      print('0.0 ')
      for i, r in enumerate( rs ):   
        #print( index_term[np.argmax(r)], end=" " )
        f.write('%s '%index_term[np.argmax(r)] )
      print('\n')

def main():
  if '--train' in sys.argv:
     train()
  if '--gen' in sys.argv:
     gen()
if __name__ == '__main__':
  main()
