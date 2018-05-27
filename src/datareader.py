import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

train_path = "../dataset/train.csv"

df = pd.read_csv(train_path, index_col=0, parse_dates=True) #reading from train.csv

# Convert categorical variable into dummy/indicator variables
# See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html#pandas-get-dummies for more details
df = pd.get_dummies(df)

# Split data into test set and cross-validation set
df_train, df_test = train_test_split(df, test_size=0.2)

print(df_train)
