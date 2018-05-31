import pandas as pd
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

import os


def prepare_data(data_file_path, test_file_labels_list):

    df = pd.read_csv(data_file_path, index_col=0, parse_dates=True) #reading from train.csv

    df.replace('?', np.nan, inplace=True) #for replacing Nan value
    df.dropna(inplace=True)

    # Convert categorical variable into dummy/indicator variables
    # See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html#pandas-get-dummies for more details
    df = pd.get_dummies(df)
    print(list(df))
    df = df.drop(columns=['G_o', 'M_p', 'E_gg', 'D_l'])

    #Split data into test set and cross-validation set
    df_train, df_cv = train_test_split(df, test_size=0.2)
    print(list(df_train))

    #Determine number of rows and columns in each data frame
    num_train_entries = df.shape[0]
    num_train_features = df.shape[1] - 1

    num_cv_entries = df_cv.shape[0]
    num_cv_features = df_cv.shape[1] - 1

    df.to_csv('train_temp.csv', index=False)
    df_cv.to_csv('cv_temp.csv', index=False)

    #Append in header row information about how many columns and rows
    #are in each file as Tensorfloww requires.
    open("data_train.csv", "w").write(str(num_train_entries) +
                                          "," + str(num_train_features) +
                                          "," + open("train_temp.csv").read())

    open("data_cv.csv", "w").write(str(num_cv_entries) +
                                         "," + str(num_cv_features) +
                                         "," + open("cv_temp.csv").read())

    # Remove temp files
    os.remove('train_temp.csv')
    os.remove('cv_temp.csv')

def prepare_data_test(data_test_path):

    df_test = pd.read_csv(data_test_path, index_col=0, parse_dates=True) #reading from test.csv


    df_test.replace('?', np.nan, inplace=True) #for replacing Nan value
    df_test.dropna(inplace=True)

    # Convert categorical variable into dummy/indicator variables
    # See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html#pandas-get-dummies for more details
    df_test = pd.get_dummies(df_test)


    df_test.to_csv('data_test.csv', index=False)
    return(list(df_test))


# Get the cv inputs
def get_cv_inputs():
  x = tf.constant(cv_set.data)
  y = tf.constant(cv_set.target)

  return x, y

# Get the training inputs
def get_train_inputs():
  x = tf.constant(training_set.data)
  y = tf.constant(training_set.target)

  return x, y

# Get test inputs
def get_test_inputs():
  import csv
  with open('data_test.csv', 'r') as f:
      reader = csv.reader(f)
      my_list = list(reader)
  my_list.pop(0)
  return np.array(my_list, dtype=float)


#preparing  training, cross-validation and test data sets
labels_list = prepare_data_test("../dataset/test.csv")
prepare_data("../dataset/train.csv", labels_list)



# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='data_train.csv',
    target_dtype=np.int,
    features_dtype=np.float,
    target_column=8)

cv_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='data_cv.csv',
    target_dtype=np.int,
    features_dtype=np.float,
    target_column=8)




# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=42)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[20, 40, 20],
    n_classes=2)


# Fit model.
classifier.fit(input_fn=get_train_inputs, steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=get_cv_inputs,
                                     steps=1)['accuracy']

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

# NEEDS-ATTENTION
predictions = list(classifier.predict(input_fn=get_test_inputs))

print("New Samples, Class Predictions:{}\n".format(predictions))
