
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

WIDTH       = 2029
MAXLEN      = 31
INPUTLEN    = 20
ACTIVATOR   = 'relu'
inputs      = Input( shape=(INPUTLEN, ) ) 
repeat      = RepeatVector(31)(inputs)
generated   = Bi( GRU(512, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )( repeat )
generated   = TD( Dense(2049, kernel_initializer='lecun_uniform', activation='sigmoid') )( generated )
generated   = Lambda( lambda x:x*2.0 )(generated)
generator   = Model( inputs, generated )

generator.compile( optimizer=Adam(), loss='categorical_crossentropy' )


