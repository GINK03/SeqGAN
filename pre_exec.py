import glob
import json
import re
import sys
import os
import pickle

""" テキストの切り出し """
def step1():
  for name in glob.glob("mstdn.jp/*.json"):
    t = json.loads(open(name, "r").read())
    c = re.sub(r"<.*?>", "", t["content"])
    c = re.sub(r"@.*?\s", "", c)
    if "" == c:
      continue
    if "http" in c: 
      continue
    if "*" in c:
      continue
    print(c)

""" N文字パディングしてスペース区切り """
def step2():
  with open("tmp/step1", "r") as f:
    for line in f:
      line = line.strip()
      es   = list(line)
      base = ["*"]*5
      base.extend(es)
      base.extend(["*"]*3)
      print(" ".join(base) )

""" fasttextでベクトル化 """
def step3():
  os.system("./fasttext skipgram -input tmp/step2 -output tmp/step3 -dim 512 -minCount 1") 
  
""" fasttextのベクトルをpkl化する """
def step4():
  char_vec = {}
  with open("tmp/step3.vec", "r") as f:
    _ = next(f)
    for line in f:
      line = line.strip()
      es   = iter(line.split())
      char = next(es)
      vec  = list(map(float, es))
      char_vec[char] = vec

  open("tmp/step4.pkl", "wb").write(pickle.dumps(char_vec))

""" charのindexを作成する """
def step5(): 
  char_index = {}
  with open("tmp/step2", "r") as f:
    cs = list(f.read())
    for i, c in enumerate(set(cs)):
      char_index[c] = i
    print(char_index)
  open("tmp/char_index.pkl", "wb").write(pickle.dumps(char_index))

if __name__ == '__main__':
  if '--step1' in sys.argv:
    step1()

  if '--step2' in sys.argv:
    step2()

  if '--step3' in sys.argv:
    step3()

  if '--step4' in sys.argv:
    step4()
  
  if '--step5' in sys.argv:
    step5()
