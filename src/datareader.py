import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

import os

#headers=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]

def prepare_data(data_file_path):
    
    df = pd.read_csv(data_file_path, index_col=0, parse_dates=True) #reading from train.csv

    df.replace('?', np.nan, inplace=True) #for replacing Nan value
    df.dropna(inplace=True)

    # Convert categorical variable into dummy/indicator variables
    # See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html#pandas-get-dummies for more details
    df = pd.get_dummies(df)

    # Split data into test set and cross-validation set
    df_train, df_cv = train_test_split(df, test_size=0.2)

    # Determine number of rows and columns in each data frame
    num_train_entries = df_train.shape[0]
    num_train_features = df_train.shape[1] - 1

    num_cv_entries = df_cv.shape[0]
    num_cv_features = df_cv.shape[1] - 1

    df_train.to_csv('train_temp.csv', index=False)
    df_cv.to_csv('cv_temp.csv', index=False)

    # Append in header row information about how many columns and rows
    # are in each file as Tensorfloww requires.
    open("data_train.csv", "w").write(str(num_train_entries) +
                                          "," + str(num_train_features) +
                                          "," + open("train_temp.csv").read())

    open("data_cv.csv", "w").write(str(num_cv_entries) +
                                         "," + str(num_cv_features) +
                                         "," + open("cv_temp.csv").read())

    # Remove temp files
    os.remove('train_temp.csv')
    os.remove('cv_temp.csv')





prepare_data("../dataset/train.csv")

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='data_train.csv',
    target_dtype=np.int,
    features_dtype=np.float,
    target_column=8)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='data_cv.csv',
    target_dtype=np.int,
    features_dtype=np.dtype(float),
    target_column=8)





