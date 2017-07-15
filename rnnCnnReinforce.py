
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

# import local file

import rnnPretrain
import rnnTrainDiscriminator

discriminator = rnnTrainDiscriminator.discriminator
generator     = rnnPretrain.generator

discriminator.load_weights('models/disc_000000009.h5')
generator.load_weights('models/adhoc_seed_000000006.h5')

def train():
  for i in range(200):
    # sampling
    xss = np.random.randn(100,1000)
    yss = generator.predict( xss )
    Ys  = []
    for ys in yss.tolist():
      ymax = [np.argmax(y)  for y in ys] 
      Ys.append( ymax )
      #print( ymax )

    rs = discriminator.predict( np.array( Ys ) )
    rs = [ r[0] for r in rs.tolist() ]
    #print( rs )
    print('average', sum(rs)/len(rs) ) 
    # スコアを反映する
    xss = xss.tolist()

    scores = []
    for r, Y, xs in  zip(rs, Ys, xss):
      #print(r, Y )
      ys = []
      for yi in Y:
        bu     = [0.0]*2049
        bu[yi] = 1.0
        ys.append(bu)
      # スコアの反映  
      ys = np.array( ys ) * r
      scores.append( ys )
    generator.fit( xss, np.array(scores), epochs=3 )
    if i%5 == 0:
      generator.save_weights('models/reinforced_%09d.h5'%i)

def predict():
  term_index = pickle.loads( open('utils/term_index.pkl', 'rb').read() )
  index_term = { index:term for term, index in term_index.items() }
  generator.load_weights('models/reinforced_000000095.h5')
  xss = np.random.randn(100,1000)
  yss = generator.predict( xss )
  Ys  = []
  for ys in yss.tolist():
    ymax = [np.argmax(y)  for y in ys] 
    
    yms =  [ index_term[ym] for ym in ymax ]
    print( yms )

if __name__ == '__main__':
  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
