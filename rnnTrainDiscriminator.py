from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Input
from keras.layers.merge import Concatenate
from keras.layers import LSTM, GRU, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dropout, Reshape, Flatten
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization as BN
from keras.callbacks import LearningRateScheduler as LRS
from keras.layers.embeddings import Embedding
from keras.models import Model
import numpy as np
import random
import sys
import glob
import pickle
import re

SEQLEN          = 31
embedding_dim   = 256
vocabulary_size = 2049
num_filters     = 512
#filter_sizes    = [(3,256),(4,256),(5,256),(1,256),(2,256)]
filter_sizes    = [3,4,5,1,2]
drop            = 0.5
def build_model():
  
  inputs        = Input(shape=(SEQLEN,), dtype='int32')
  embedding     = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=SEQLEN)(inputs)
  reshape       = Reshape((SEQLEN,embedding_dim,1))(embedding)

  conv_0        = Convolution2D(512, (3, 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
  conv_1        = Convolution2D(512, (4, 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
  conv_2        = Convolution2D(512, (5, 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
  conv_3        = Convolution2D(512, (1, 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
  conv_4        = Convolution2D(512, (2, 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

  maxpool_0     = MaxPooling2D(pool_size=(SEQLEN - 3 + 1, 1),  strides=(1,1), padding='valid')(conv_0)
  maxpool_1     = MaxPooling2D(pool_size=(SEQLEN - 4 + 1, 1),  strides=(1,1), padding='valid')(conv_1)
  maxpool_2     = MaxPooling2D(pool_size=(SEQLEN - 5 + 1, 1),  strides=(1,1), padding='valid')(conv_2)
  maxpool_3     = MaxPooling2D(pool_size=(SEQLEN - 1 + -3, 1), strides=(1,1), padding='valid')(conv_3)
  maxpool_4     = MaxPooling2D(pool_size=(SEQLEN - 2 + -2, 1), strides=(1,1), padding='valid')(conv_4)

  merged_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
  flatten       = Flatten()( merged_tensor )
  dropout       = Dropout( drop )( flatten )
  output        = Dense( 1, activation='sigmoid' )( dropout )

  model         = Model(inputs, output)
  opts          = Adam()
  model.compile(optimizer=opts, loss='binary_crossentropy', metrics=['accuracy'])
  return model

discriminator = build_model()

def generate_mix():
  term_index = pickle.loads( open('utils/term_index.pkl', 'rb').read() )
  with open('mix.data.txt') as f:
    ys = []
    xs = []
    for line in f:
      if line == '' : 
        continue
      line = line.strip()
      ents = line.split()
      weight = float( ents.pop(0) )
      x = [ int(term_index[x]) for x in ents ]
      ys.append( weight )
      xs.append( x )

    ys = np.array( ys )
    xs = np.array( xs )
    for i in range(10):
      discriminator.fit( xs, ys, validation_split=0.05, epochs=5 )
      discriminator.save_weights('models/disc_%09d.h5'%i)

if __name__ == '__main__':
  if '--step1' in sys.argv:
    generate_mix()
