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

def build_model(mode=None, maxlen=None, output_dim=None):
  print('Build model...')
  def _scheduler(epoch):
    global counter
    counter += 1
    rate = learning_rates[counter]
    #model.lr.set_value(rate)
    print(model.optimizer.lr)
    #print(counter, rate)
    return model.optimizer.lr
  change_lr = LRS(_scheduler)
  model = Sequential()
  model.add(LSTM(128*10, return_sequences=False, input_shape=(maxlen, 512)))
  model.add(Dense(output_dim))
  model.add(Activation('softmax'))
  optimizer = Adam()
  model.compile(loss='categorical_crossentropy', optimizer=optimizer) 
  return model, change_lr


def sample(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def top(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  return np.argmax(preds)

def train(mode):
  with open("./tmp/step4.pkl", "rb") as f:
    char_vec = pickle.loads(f.read())

  with open("./tmp/char_index.pkl", "rb") as f:
    char_index = pickle.loads(f.read())
    index_char = { index:char for char, index in char_index.items() }

  Xs, Ys = [], []
  print("start to load data to memory...")
  counter = 0
  with open("tmp/step2", "r") as f:
    for fi, line in enumerate(f):
      if counter > 40000:
        break
      if fi%1000 == 0:
        print("now iter {counter}".format(counter=counter), file=sys.stderr)
      line = line.strip()
      chars = line.split()
      """ ある程度の長文を対象にしたいので、50字未満はスキップする """
      if len(chars) < 70:
        continue
      """ また、長すぎる文章もスキップしたいので、80字より大きいはスキップする """
      if len(chars) > 100:
        continue
      #print(chars)
      for i in range(0, len(chars) - 6):
        window = chars[i:i+6]
        if window[0] == "*" and window[-1] == "*":
          continue
        if window[-2] == "*" and window[-1] == "*":
          continue

        """ 特定の確率でデータセットに入れない """
        if random.random() > 0.20: 
          continue
        #print(window)
        """ ここからベクトル化 """
        ys     = [0.]*len(char_index)
        ys[char_index[window.pop()] ] = 1.0
        xs     = list( map(lambda x:char_vec[x], window) )

        Xs.append(xs)
        Ys.append(ys)
      counter += 1
  print("finished heaping memory...")
  Xs, Ys = np.array(Xs), np.array(Ys)
  print("try to create model...")
  model, scheduler = build_model(mode=mode, maxlen=5, output_dim=len(char_index))
  for iteration in range(1, 10000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(Xs, Ys, batch_size=64, epochs=1)#, callbacks=[scheduler])
    MODEL_NAME = "./models/%09d.model"%(iteration)
    model.save(MODEL_NAME)
    if iteration%1==0:
      for diversity in [0.8, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)
        sent = ["*"]*5
        print('----- Generating with seed: "' + "".join(sent) + '"')
        sys.stdout.write("".join(sent))
        for i in range(100):
          try:
            xs         = list(map(lambda x:char_vec[x], sent))
            preds      = model.predict(np.array([xs]), verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char  = index_char[next_index]
            sent.append(next_char)
            sent = sent[1:] 
            sys.stdout.write(next_char)
            sys.stdout.flush()
          except KeyError as e:
            break
        print()

def eval():
  INPUT_NAME = "./source/bocchan.txt"
  MODEL_NAME = "./models/%s.model"%(INPUT_NAME.split('/').pop())

  #path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
  #text = open(path).read().lower()
  text = open(INPUT_NAME).read()
  print('corpus length:', len(text))

  chars = sorted(list(set(text)))
  print('total chars:', len(chars))
  char_indices = dict((c, i) for i, c in enumerate(chars))
  indices_char = dict((i, c) for i, c in enumerate(chars))
  maxlen = 40
  step = 3
  sentences = []
  next_chars = []
  for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
  model = load_model(MODEL_NAME)

  #for diversity in [0.2, 0.5, 1.0, 1.2]:
  for diversity in [1.0, 1.2]:
    print()
    print('----- diversity:', diversity)
    generated = ''
    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
    #for i in range(400):
    for i in range(200):
      x = np.zeros((1, maxlen, len(chars)))
      for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.
      preds = model.predict(x, verbose=0)[0]
      #next_index = sample(preds, diversity)
      next_index = dynast(preds, diversity)
      next_char = indices_char[next_index]
      generated += next_char
      sentence = sentence[1:] + next_char
      sys.stdout.write(next_char)
      sys.stdout.flush()
    print()


def main():
  if '--train' in sys.argv:
     train(mode="adam")
  if '--train_rms' in sys.argv:
     train(mode="rms")
  if '--eval' in sys.argv:
     eval()
if __name__ == '__main__':
  main()
