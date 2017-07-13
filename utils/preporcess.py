import glob
import os
import MeCab
import re
import pickle
import numpy as np
import sys
MAXLEN = 30
PADD   = '<PAD>'

def nucReadAsUTF8():
  m = MeCab.Tagger('-Owakati')
  corpus = []
  for name in glob.glob('nuc/*.txt'):
    for line in os.popen('nkf -w {}'.format(name)).read().split('\n'):
      if line == '' or '＠' in line:
        continue
      # remove header-tag like F001:.
      line = re.sub(r'.\d{1,}：', '', line)
      spss =  m.parse(line).strip().split()
      # 30以下の長さのみ今回は採用
      if  len(spss) > 30:
        continue
      corpus.append( spss )

  open('corpus.pkl', 'wb').write( pickle.dumps( corpus ) ) 

def regurationTermNumber():
  corpus = pickle.loads( open('corpus.pkl', 'rb').read() )
  term_freq = {}
  for text in corpus:
    for term in text:
      if term_freq.get(term) is None:
        term_freq[term] = 0
      term_freq[term] += 1

  use = set()
  for term, freq in sorted(term_freq.items(), key=lambda x:x[1]*-1)[:2048]:
    use.add( term )

  regurationCorpus = []
  for text in corpus:
    if set(text) - use != set() :
      continue
    regurationCorpus.append( text )
  open('regurationCorpus.pkl', 'wb').write( pickle.dumps( regurationCorpus ) ) 

def toIndexAndReguration():
  regurationCorpus = pickle.loads( open('regurationCorpus.pkl', 'rb').read()  )

  pads = []
  for text in regurationCorpus:
    base = [PADD for i in range(MAXLEN+1) ]
    for i, term in enumerate(text):
      base[i] = term
    pads.append( base )

  flatten = set()
  [ [flatten.add(x) for x in xs] for xs in pads]
  
  term_index = {}
  for term in flatten:
    if term_index.get(term) is None:
      term_index[term] = len( term_index )

  # to indexies
  indexed = [ [term_index[term] for term in text] for text in pads]
  
  open('indexed.pkl', 'wb').write( pickle.dumps(indexed) ) 
   
def makePair():
  indexed = pickle.loads( open('indexed.pkl', 'rb').read() )

  pairs = []
  for text in indexed:
    seed = np.random.randn(20)
    pairs.append( (seed, np.array(text) ) )
  
  xs = [ pair[0] for pair in pairs ]
  ys = [ pair[1] for pair in pairs ]
  open('dataset.pkl', 'wb').write( pickle.dumps( (xs, ys) ) )

    
  

if __name__ == '__main__':
  if '--step1' in sys.argv: 
    nucReadAsUTF8()

  if '--step2' in sys.argv:
    regurationTermNumber()

  if '--step3' in sys.argv:
    toIndexAndReguration()

  if '--step4' in sys.argv:
    makePair()
