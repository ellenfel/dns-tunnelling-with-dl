#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# preprocessing template

# importing the libraries
import numpy as np
import tensorflow as tf
import re
import time
import pandas as pd

"""
# Importing the dataset as list
lines = open('captured', encoding = 'utf-8', errors = 'ignore').read().split('\n')
"""

# Importing the datasets as DataFrame
# need to convert dataframe to float64 for feature scalling
dataset0 = pd.read_fwf('normal_40000', sep=" ", header = None) #tunelsiz
dataset1 = pd.read_fwf('tunnelling1', sep=" ", header = None) #tünelli

# adding group no
dataset0["group_no"] = dataset0.T.isnull().all().cumsum()
dataset1["group_no"] = dataset1.T.isnull().all().cumsum()

to_drop = [3]
dataset1.drop(to_drop, inplace=True, axis=1)

# dropping the first two column
to_drop = [0, 1]
dataset0.drop(to_drop, inplace=True, axis=1)
dataset1.drop(to_drop, inplace=True, axis=1)

# dropping null value columns to avoid errors 
dataset0.dropna(inplace = True) 
dataset1.dropna(inplace = True) 
  
# new data frame with split value columns 
new0 = dataset0[2].str.split(":", n = 1, expand = True)
new1 = dataset1[2].str.split(":", n = 1, expand = True)
  
# making seperate first name column from new data frame 
dataset0["Name"]= new0[0] 
new0["Name"]=dataset0["Name"]

dataset1["Name"]= new1[0] 
new1["Name"]=dataset1["Name"]
  
# making seperate last name column from new data frame 
dataset0["Feature"]= new0[1] 
new0["Feature"]=dataset0["Feature"]

dataset1["Feature"]= new1[1] 
new1["Feature"]=dataset1["Feature"]

# relocate columns in data set ///dont do that without looking arround
dataset0 = dataset0[[2, 'Name', 'Feature', 'group_no']]
to_drop = [2]
dataset0.drop(to_drop, inplace=True, axis=1)

dataset1 = dataset1[[2, 'Name', 'Feature', 'group_no']]
to_drop = [2]
dataset1.drop(to_drop, inplace=True, axis=1)

# works don't delete
to_drop = [0, 1]
new0.drop(to_drop, inplace=True, axis=1)

to_drop = [0, 1]
new1.drop(to_drop, inplace=True, axis=1)

"""
# cleaning 
new = new[new.Name != '},']
"""

# Encoding categorical data/// working
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_new0 = LabelEncoder()
new0.values[:, 0] = labelencoder_new0.fit_transform(new0.values[:, 0])

labelencoder_new1 = LabelEncoder()
new1.values[:, 0] = labelencoder_new1.fit_transform(new1.values[:, 0])

# add group_no to new[]
new0['group_no'] = dataset0["group_no"]
new1['group_no'] = dataset1["group_no"]

# cleaning 2
fea0=new0[new0.Name == 702]  # type(str)
#fea1=new0[new0.Name == 916]  # data_len(int)
fea2=new0[new0.Name == 1263] # ip_len(int)
fea3=new0[new0.Name == 1067] # frame_len(int) 
fea7=new0[new0.Name == 1059] # frame.cap_len(int)
#fea8=new0[new0.Name == 1240] # ip.hdr_len(int)
#fea9=new0[new0.Name == 1273] # ip.proto(int)
#fea10=new0[new0.Name == 1858] # tcp_len(int)
fea11=new0[new0.Name == 1253] #ip.flag.rb
fea12=new0[new0.Name == 1251] #ip.flag.df
fea13=new0[new0.Name == 1252] #ip.flag.mf
#fea14=new0[new0.Name == 1849] #tcp.flag.res
#fea15=new0[new0.Name == 1847] #tcp.flag.ns
#fea16=new0[new0.Name == 1843] #tcp.flag.cwr
#fea17=new0[new0.Name == 1844] #tcp.flag.ecn
#fea18=new0[new0.Name == 1855] #tcp.flag.urg
#fea19=new0[new0.Name == 1842] #tcp.flag.ack
#fea20=new0[new0.Name == 1848] #tcp.flag.push

fea0['Feature'] = fea0['Feature'].map(lambda x: str(x)[:-1])
fea0['Feature'] = fea0['Feature'].map(lambda x: str(x)[:-1])
fea0['Feature'] = fea0['Feature'].map(lambda x: str(x)[1:])
fea0['Feature'] = fea0['Feature'].map(lambda x: str(x)[1:])

#fea1['Feature'] = fea1['Feature'].str.replace(r'\D', '').astype(int)

fea2['Feature'] = fea2['Feature'].map(lambda x: str(x)[:-1])
fea2['Feature'] = fea2['Feature'].str.replace(r'\D', '').astype(int)

fea3['Feature'] = fea3['Feature'].map(lambda x: str(x)[:-1])
fea3['Feature'] = fea3['Feature'].str.replace(r'\D', '').astype(int)

fea7['Feature'] = fea7['Feature'].map(lambda x: str(x)[:-1])
fea7['Feature'] = fea7['Feature'].str.replace(r'\D', '').astype(int)

#
#fea8['Feature'] = fea8['Feature'].map(lambda x: str(x)[:-1])
#fea8['Feature'] = fea8['Feature'].str.replace(r'\D', '').astype(int)

#fea9['Feature'] = fea9['Feature'].map(lambda x: str(x)[:-1])
#fea9['Feature'] = fea9['Feature'].str.replace(r'\D', '').astype(int)

#fea10['Feature'] = fea10['Feature'].map(lambda x: str(x)[:-1])
#fea10['Feature'] = fea10['Feature'].str.replace(r'\D', '').astype(int)

fea11['Feature'] = fea11['Feature'].map(lambda x: str(x)[:-1])
fea11['Feature'] = fea11['Feature'].str.replace(r'\D', '').astype(int)

fea12['Feature'] = fea12['Feature'].map(lambda x: str(x)[:-1])
fea12['Feature'] = fea12['Feature'].str.replace(r'\D', '').astype(int)

fea13['Feature'] = fea13['Feature'].map(lambda x: str(x)[:-1])
fea13['Feature'] = fea13['Feature'].str.replace(r'\D', '').astype(int)

#fea14['Feature'] = fea14['Feature'].map(lambda x: str(x)[:-1])
#fea14['Feature'] = fea14['Feature'].str.replace(r'\D', '').astype(int)

#fea15['Feature'] = fea15['Feature'].map(lambda x: str(x)[:-1])
#fea15['Feature'] = fea15['Feature'].str.replace(r'\D', '').astype(int)

#fea16['Feature'] = fea16['Feature'].map(lambda x: str(x)[:-1])
#fea16['Feature'] = fea16['Feature'].str.replace(r'\D', '').astype(int)

#fea17['Feature'] = fea17['Feature'].map(lambda x: str(x)[:-1])
#fea17['Feature'] = fea17['Feature'].str.replace(r'\D', '').astype(int)

#fea18['Feature'] = fea18['Feature'].map(lambda x: str(x)[:-1])
#fea18['Feature'] = fea18['Feature'].str.replace(r'\D', '').astype(int)

#fea19['Feature'] = fea19['Feature'].map(lambda x: str(x)[:-1])
#fea19['Feature'] = fea19['Feature'].str.replace(r'\D', '').astype(int)

#fea20['Feature'] = fea20['Feature'].map(lambda x: str(x)[:-1])
#fea20['Feature'] = fea20['Feature'].str.replace(r'\D', '').astype(int)

fea4=new1[new1.Name == 7685] # type(str)
fea5=new1[new1.Name == 7839] # ip_len(int)
fea6=new1[new1.Name == 7798] # frame_len(int)
fea21=new1[new1.Name == 7790] # frame.cap_len(int)
fea22=new1[new1.Name == 7833] # ip.flag_rb
fea23=new1[new1.Name == 7831] # ip.flag.df
fea24=new1[new1.Name == 7832] # ip.flag.mf


fea4['Feature'] = fea4['Feature'].map(lambda x: str(x)[:-1])
fea4['Feature'] = fea4['Feature'].map(lambda x: str(x)[:-1])
fea4['Feature'] = fea4['Feature'].map(lambda x: str(x)[1:])
fea4['Feature'] = fea4['Feature'].map(lambda x: str(x)[1:])

fea5['Feature'] = fea5['Feature'].map(lambda x: str(x)[:-1])
fea5['Feature'] = fea5['Feature'].str.replace(r'\D', '').astype(int)

fea6['Feature'] = fea6['Feature'].map(lambda x: str(x)[:-1])
fea6['Feature'] = fea6['Feature'].str.replace(r'\D', '').astype(int)

fea21['Feature'] = fea21['Feature'].map(lambda x: str(x)[:-1])
fea21['Feature'] = fea21['Feature'].str.replace(r'\D', '').astype(int)

fea22['Feature'] = fea22['Feature'].map(lambda x: str(x)[:-1])
fea22['Feature'] = fea22['Feature'].str.replace(r'\D', '').astype(int)

fea23['Feature'] = fea23['Feature'].map(lambda x: str(x)[:-1])
fea23['Feature'] = fea23['Feature'].str.replace(r'\D', '').astype(int)

fea24['Feature'] = fea24['Feature'].map(lambda x: str(x)[:-1])
fea24['Feature'] = fea24['Feature'].str.replace(r'\D', '').astype(int)


fea0.columns = ['Name','Type','group_no']
fea2.columns = ['Name','Ip_len','group_no']
fea3.columns = ['Name','Frame_len','group_no']
fea7.columns = ['Name','Frame.cap_len','group_no']
fea11.columns = ['Name','Ip.flag.rb','group_no']
fea12.columns = ['Name','Ip.flag.df','group_no']
fea13.columns = ['Name','Ip.flag.mf','group_no']
fea0=pd.get_dummies(fea0, prefix=['Type'], columns=['Type'])

fea4.columns = ['Name','Type','group_no']
fea5.columns = ['Name','Ip_len','group_no']
fea6.columns = ['Name','Frame_len','group_no']
fea21.columns = ['Name','Frame.cap_len','group_no']
fea22.columns = ['Name','Ip.flag.rb','group_no']
fea23.columns = ['Name','Ip.flag.df','group_no']
fea24.columns = ['Name','Ip.flag.mf','group_no']
fea4=pd.get_dummies(fea4, prefix=['Type'], columns=['Type'])

# combaine datas
frames0 = [fea0, fea2, fea3, fea7, fea11, fea12, fea13]
frames1 = [fea4, fea5, fea6, fea21, fea22, fea23, fea24]

prototype0 = pd.concat(frames0)
prototype1 = pd.concat(frames1)

prototype0=prototype0.sort_index(axis=0, level=None, ascending=True, inplace=False,
                                 kind='quicksort', na_position='last', sort_remaining=True,
                                 by=None)

prototype1=prototype1.sort_index(axis=0, level=None, ascending=True, inplace=False,
                                 kind='quicksort', na_position='last', sort_remaining=True,
                                 by=None)

###########################################
###########################################
#dummy encoding ///did work
prototype0=pd.get_dummies(prototype0, prefix=['Name'], columns=['Name'])
prototype1=pd.get_dummies(prototype1, prefix=['Name'], columns=['Name'])

#dropping the first two column that point feature names
prototype0 = prototype0[['Name_702', 'Name_1067', 'Name_1263', 'Frame_len', 'Ip_len', 'Type_pcap_file', 'group_no']]
prototype0 = prototype0.drop('Name_702', 1)
prototype0 = prototype0.drop('Name_1067', 1)
prototype0 = prototype0.drop('Name_1263', 1)

prototype1 = prototype1[['Name_7685', 'Name_7798', 'Name_7839', 'Frame_len', 'Ip_len', 'Type_pcap_file', 'group_no']]
prototype1 = prototype1.drop('Name_7685', 1)
prototype1 = prototype1.drop('Name_7798', 1)
prototype1 = prototype1.drop('Name_7839', 1)
###########################################
###########################################


del prototype1['Name']
del prototype0['Name']

#nan=0 for calculating
prototype01=prototype0.fillna(0)
prototype11=prototype1.fillna(0)

#calculating
aggregation_functions = {'Frame_len': 'sum', 'Frame.cap_len': 'sum', 'Ip_len': 'sum', 'Type_pcap_file': 'sum', 'Ip.flag.rb': 'sum', 'Ip.flag.df': 'sum', 'Ip.flag.mf': 'sum', 'group_no': 'first'}
prototype001 = prototype01.groupby(prototype01['group_no']).aggregate(aggregation_functions)
del prototype001['group_no']

prototype011 = prototype11.groupby(prototype11['group_no']).aggregate(aggregation_functions)
del prototype011['group_no']


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
prototype011['Tunnelling']=1

final_form = [prototype011, prototype001]
final_form = pd.concat(final_form)

X = final_form.iloc[:, 0:7].values
y = final_form.iloc[:, 7].values

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
classifier.add(Dense(activation="relu", input_dim=7, units=6, kernel_initializer="uniform"))

# adding the second hiden layer with Dropout
classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))
classifier.add(Dropout(p = 0.1))

# adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 30) 

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting the single packet
new_predection = classifier.predict(sc.transform(np.array([[3, 0, 0,]])))



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=7, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 200)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=7, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [16, 10, 32],
              'nb_epoch': [50, 10, 100],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_



##################################
#################################
######other attamps for rnn #######
#################################
################################



#recurrent neural network 

###data preprocessing###

# importing the libraries
import matplotlib.pyplot as plt 

# feature scalling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(final_form)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(100, 82736):
    X_train.append(training_set_scaled[i-100:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
 

###Building the network###

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularistion
regressor.add(LSTM(units = 50, return_sequences = True,
                   input_shape = (X_train.shape[1], 1)))
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


