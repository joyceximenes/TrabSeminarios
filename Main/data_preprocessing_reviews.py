import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

random.seed(13)

def normalize(X_train, X_val, X_test):
  X_train_normalized = X_train.values.astype(float)
  min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
  x_scaled = min_max_scaler.fit_transform(X_train_normalized)

    # Run the normalizer on the dataframe
  x = pd.DataFrame(x_scaled, columns=X_train.columns)

  X_val_normalized = X_val.values.astype(float)
  min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
  x_val_scaled = min_max_scaler.fit_transform(X_val_normalized)

    # Run the normalizer on the dataframe
  x_v = pd.DataFrame(x_val_scaled, columns=X_val.columns)

  X_test_normalized = X_test.values.astype(float)
  min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
  x_test_scaled = min_max_scaler.fit_transform(X_test_normalized)

    # Run the normalizer on the dataframe
  x_t = pd.DataFrame(x_test_scaled, columns=X_test.columns)
  return x, x_v, x_t

def split(dataset, val_frac=0.10, test_frac=0.10):
  X = dataset.loc[:, dataset.columns != 'price']
  X = X.loc[:, X.columns != 'id']
  X = X.loc[:, X.columns != 'host_id']
  X = X.loc[:, X.columns != 'Unnamed: 0']
  # Remove a coluna 'comments' se ela existir no dataset
  if 'comments' in X.columns:
      X = X.drop(columns=['comments'])

  y = dataset['price']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(val_frac+test_frac), random_state=1)
  X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_frac/(val_frac+test_frac), random_state=1)

  return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
  dataset = pd.read_csv('../Data/data_cleaned.csv')

  X_train, y_train, X_val, y_val, X_test, y_test = split(dataset)

  X_train, X_val, X_test = normalize(X_train, X_val, X_test)
  
  # Remova "_comments" dos nomes dos arquivos de saída
  X_train.to_csv('../Data/data_cleaned_train_X.csv', header=True, index=False)
  y_train.to_csv('../Data/data_cleaned_train_y.csv', header=True, index=False)

  X_val.to_csv('../Data/data_cleaned_val_X.csv', header=True, index=False)
  y_val.to_csv('../Data/data_cleaned_val_y.csv', header=True, index=False)

  X_test.to_csv('../Data/data_cleaned_test_X.csv', header=True, index=False)
  y_test.to_csv('../Data/data_cleaned_test_y.csv', header=True, index=False)