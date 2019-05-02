#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# preprocessing template

# importing the libraries
import numpy as np
import tensorflow as tf
import re
import time
import pandas as pd

# Importing the dataset as list
lines = open('captured', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Importing the dataset as DataFrame
# need to convert dataframe to float64 for feature scalling  
dataset = pd.read_fwf('captured', sep=" ", header = None)

# adding group no
dataset["group_no"] = dataset.T.isnull().all().cumsum()

# dropping the first two column
to_drop = [0, 1]
dataset.drop(to_drop, inplace=True, axis=1)

# dropping null value columns to avoid errors 
dataset.dropna(inplace = True) 
  
# new data frame with split value columns 
new = dataset[2].str.split(":", n = 1, expand = True)
  
# making seperate first name column from new data frame 
dataset["Name"]= new[0] 
new["Name"]=dataset["Name"]
  
# making seperate last name column from new data frame 
dataset["Feature"]= new[1] 
new["Feature"]=dataset["Feature"]

# relocate columns in data set ///dont do that without looking arround
dataset = dataset[[2, 'Name', 'Feature', 'group_no']]
to_drop = [2]
dataset.drop(to_drop, inplace=True, axis=1)

# works don't delete
to_drop = [0, 1]
new.drop(to_drop, inplace=True, axis=1)

"""
# cleaning 
new = new[new.Name != '},']
"""

# Encoding categorical data/// working
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_new = LabelEncoder()
new.values[:, 0] = labelencoder_new.fit_transform(new.values[:, 0])

# add group_no to new[]
new['group_no'] = dataset["group_no"]

# cleaning 2
new0=new[new.Name == 25]
new1=new[new.Name == 42]
new0['Feature'] = new0['Feature'].map(lambda x: str(x)[:-1])
new0['Feature'] = new0['Feature'].map(lambda x: str(x)[:-1])
new0['Feature'] = new0['Feature'].map(lambda x: str(x)[1:])
new0['Feature'] = new0['Feature'].map(lambda x: str(x)[1:])
new1['Feature'] = new1['Feature'].str.replace(r'\D', '').astype(int)

new0.columns = ['Name','Type','group_no']
new1.columns = ['Name','Data_length','group_no']
new0=pd.get_dummies(new0, prefix=['Type'], columns=['Type'])

# combaine datas
frames0 = [new0, new1]
prototype0 = pd.concat(frames0)

prototype0=prototype0.sort_index(axis=0, level=None, ascending=True, inplace=False,
                                 kind='quicksort', na_position='last', sort_remaining=True,
                                 by=None)

#dummy encoding ///did work
prototype0=pd.get_dummies(prototype0, prefix=['Name'], columns=['Name'])

#dropping the first two column that point feature names
prototype0 = prototype0[['Name_25', 'Name_42', 'Data_length', 'Type_pcap_file', 'group_no']]
prototype0 = prototype0.drop('Name_25', 1)
prototype0 = prototype0.drop('Name_42', 1)

#nan=0 for calculating
prototype01=prototype0.fillna(0)

#calculating
aggregation_functions = {'Data_length': 'sum', 'Type_pcap_file': 'sum', 'group_no': 'first'}
prototype001 = prototype01.groupby(prototype01['group_no']).aggregate(aggregation_functions)
del prototype001['group_no']

"""
# combaine columns----prototype1----
prototype1 = prototype0['Data_length'].combine_first(prototype0['Type_pcap_file'])
prototype1 = pd.DataFrame(data=prototype1)
prototype1['group_no'] = prototype0["group_no"]
prototype1['Name_25'] = prototype0["Name_25"]
prototype1['Name_42'] = prototype0["Name_42"]
prototype1 = prototype1[['Name_25', 'Name_42', 'Data_length', 'group_no']]
prototype1.columns = ['Name_25', 'Name_42', 'Features', 'group_no']
"""

prototype001['Tunnelling']=0
X = prototype001.iloc[:, 0:2].values
y = prototype001.iloc[:, 2].values

# Encoding categorical data

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN

# importing the Keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# initialising the ANN
classifier = Sequential()

# adding the input layer and first hiden layer
classifier.add(Dense(activation="relu", input_dim=2, units=6, kernel_initializer="uniform"))

# adding the second hiden layer with Dropout
classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))
classifier.add(Dropout(p = 0.1))

# adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 200) 

# Predicting the Test set results
y_pred = classifier.predict(X_test)











# feature scalling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(prototype001)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(30, 54):
    X_train.append(training_set_scaled[i-30:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


"""
# defining a function for cleaning(not working for some reason) 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"{", "", text)
    text = re.sub(r"}", "", text)
    text = re.sub(r"[", "", text)
    text = re.sub(r"]", "", text)
    text = re.sub(r"   ", "", text)
    text = re.sub(r"  ", "", text)
    text = re.sub(r"        ", "", text)
    text = re.sub(r"            ", "", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text 
  
clean_lines = []
for line in lines:
    clean_lines.append(clean_text(lines))  
    

    
clean_linesX
for line in dataset:
    clean_linesX.append(clean_text(line))
"""

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
dataset_train, dataset_test= train_test_split(prototype001, test_size = 0.2, random_state = 0)



###Building the network###

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularistion
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



