import pandas as pd
import tensorflow as tf

train_path = "../dataset/train.csv"

data=pd.read_csv(train_path, index_col=0, parse_dates=True) #reading from train.csv

print(data)
