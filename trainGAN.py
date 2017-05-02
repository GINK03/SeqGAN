import os
import sys

"""" 最初にシードモデルからサンプルを作成する """

os.system("python3 bulkGenerator.py --seed")


""" シードから分離面を構築する """
""" 構築した分離面に対して重み付けする """

os.system("python3 rnnTrainDiscriminator.py")


""" generatorを学習する """
os.system("python3 rnnTrainGenerator.py")

""" 最初のステップが完了したら、回すだけ """ 

for i in range(1):
  os.system("python3 bulkGenerator.py")
  os.system("python3 rnnTrainDiscriminator.py")
  os.system("python3 rnnTrainGenerator.py")
