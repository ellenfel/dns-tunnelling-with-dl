#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# preprocessing template

# importing the libraries
import math
import numpy as np
import pandas as pd

# Importing the datasets as DataFrame
# need to convert dataframe to float64 for feature scalling
dataset0 = pd.read_fwf('dnsnormal_1.json', sep=" ", header = None) #tunelsiz
dataset1 = pd.read_fwf('tunnel-data-4_1.txt', sep=" ", header = None) #tünelli

# adding group no
dataset0["group_no"] = dataset0.T.isnull().all().cumsum()
dataset1["group_no"] = dataset1.T.isnull().all().cumsum()

to_drop = [3]
dataset1.drop(to_drop, inplace=True, axis=1)

# dropping the first two column
to_drop = [0, 1]
dataset1.drop(to_drop, inplace=True, axis=1)
dataset0.drop(to_drop, inplace=True, axis=1)

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

# getting rid of data that we don't need
to_drop = [0, 1]
new0.drop(to_drop, inplace=True, axis=1)

to_drop = [0, 1]
new1.drop(to_drop, inplace=True, axis=1)


# Encoding categorical data/// working
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_new0 = LabelEncoder()
new0.values[:, 0] = labelencoder_new0.fit_transform(new0.values[:, 0])

labelencoder_new1 = LabelEncoder()
new1.values[:, 0] = labelencoder_new1.fit_transform(new1.values[:, 0])

# add group_no to new[]
new0['group_no'] = dataset0["group_no"]
new1['group_no'] = dataset1["group_no"]

# getting the futures that we gonna use in nn 
fea0=new0[new0.Name == 198]  # type(str)
fea2=new0[new0.Name == 3374] # ip_len(int)
fea3=new0[new0.Name == 2507] # frame_len(int) 
fea12=new0[new0.Name == 3366] #ip.flag.df
fea25=new0[new0.Name == 1791]  # dns.qry.name(str)
fea26=new0[new0.Name == 1792]  # dns.qry.name.len(int)
#fea1=new0[new0.Name == 916]  # data_len(int)
#fea7=new0[new0.Name == 1059] # frame.cap_len(int)  --id name wrong
#fea8=new0[new0.Name == 1240] # ip.hdr_len(int)
#fea9=new0[new0.Name == 1273] # ip.proto(int)
#fea10=new0[new0.Name == 1858] # tcp_len(int)
#fea11=new0[new0.Name == 3368] #ip.flag.rb
#fea13=new0[new0.Name == 3367] #ip.flag.mf
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

fea2['Feature'] = fea2['Feature'].map(lambda x: str(x)[:-1])
fea2['Feature'] = fea2['Feature'].str.replace(r'\D', '').astype(int)

fea3['Feature'] = fea3['Feature'].map(lambda x: str(x)[:-1])
fea3['Feature'] = fea3['Feature'].str.replace(r'\D', '').astype(int)

fea12['Feature'] = fea12['Feature'].map(lambda x: str(x)[:-1])
fea12['Feature'] = fea12['Feature'].str.replace(r'\D', '').astype(int)

fea25['Feature'] = fea25['Feature'].map(lambda x: str(x)[:-1])
fea25['Feature'] = fea25['Feature'].map(lambda x: str(x)[:-1])
fea25['Feature'] = fea25['Feature'].map(lambda x: str(x)[1:])
fea25['Feature'] = fea25['Feature'].map(lambda x: str(x)[1:])

fea26['Feature'] = fea26['Feature'].map(lambda x: str(x)[:-1])
fea26['Feature'] = fea26['Feature'].str.replace(r'\D', '').astype(int)

#fea1['Feature'] = fea1['Feature'].str.replace(r'\D', '').astype(int)

#fea7['Feature'] = fea7['Feature'].map(lambda x: str(x)[:-1])
#fea7['Feature'] = fea7['Feature'].str.replace(r'\D', '').astype(int)

#fea8['Feature'] = fea8['Feature'].map(lambda x: str(x)[:-1])
#fea8['Feature'] = fea8['Feature'].str.replace(r'\D', '').astype(int)

#fea9['Feature'] = fea9['Feature'].map(lambda x: str(x)[:-1])
#fea9['Feature'] = fea9['Feature'].str.replace(r'\D', '').astype(int)

#fea10['Feature'] = fea10['Feature'].map(lambda x: str(x)[:-1])
#fea10['Feature'] = fea10['Feature'].str.replace(r'\D', '').astype(int)

#fea11['Feature'] = fea11['Feature'].map(lambda x: str(x)[:-1])
#fea11['Feature'] = fea11['Feature'].str.replace(r'\D', '').astype(int)

#fea13['Feature'] = fea13['Feature'].map(lambda x: str(x)[:-1])
#fea13['Feature'] = fea13['Feature'].str.replace(r'\D', '').astype(int)

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




fea4=new1[new1.Name == 16651] # type(str)
fea5=new1[new1.Name == 16750] # ip_len(int)
fea6=new1[new1.Name == 16721] # frame_len(int)
fea23=new1[new1.Name == 16742] # ip.flag.df
fea27=new1[new1.Name == 16678] # dns.qry.name(str)
fea28=new1[new1.Name == 16679] # dns.qry.name.len(int)
#fea21=new1[new1.Name == 7790] # frame.cap_len(int)
#fea22=new1[new1.Name == 7833] # ip.flag_rb
#fea24=new1[new1.Name == 7832] # ip.flag.mf


fea4['Feature'] = fea4['Feature'].map(lambda x: str(x)[:-1])
fea4['Feature'] = fea4['Feature'].map(lambda x: str(x)[:-1])
fea4['Feature'] = fea4['Feature'].map(lambda x: str(x)[1:])
fea4['Feature'] = fea4['Feature'].map(lambda x: str(x)[1:])

fea5['Feature'] = fea5['Feature'].map(lambda x: str(x)[:-1])
fea5['Feature'] = fea5['Feature'].str.replace(r'\D', '').astype(int)

fea6['Feature'] = fea6['Feature'].map(lambda x: str(x)[:-1])
fea6['Feature'] = fea6['Feature'].str.replace(r'\D', '').astype(int)

fea23['Feature'] = fea23['Feature'].map(lambda x: str(x)[:-1])
fea23['Feature'] = fea23['Feature'].str.replace(r'\D', '').astype(int)

fea27['Feature'] = fea27['Feature'].map(lambda x: str(x)[:-1])
fea27['Feature'] = fea27['Feature'].map(lambda x: str(x)[:-1])
fea27['Feature'] = fea27['Feature'].map(lambda x: str(x)[1:])
fea27['Feature'] = fea27['Feature'].map(lambda x: str(x)[1:])

fea28['Feature'] = fea28['Feature'].map(lambda x: str(x)[:-1])
fea28['Feature'] = fea28['Feature'].str.replace(r'\D', '').astype(int)

#fea21['Feature'] = fea21['Feature'].map(lambda x: str(x)[:-1])
#fea21['Feature'] = fea21['Feature'].str.replace(r'\D', '').astype(int)

#fea22['Feature'] = fea22['Feature'].map(lambda x: str(x)[:-1])
#fea22['Feature'] = fea22['Feature'].str.replace(r'\D', '').astype(int)

#fea24['Feature'] = fea24['Feature'].map(lambda x: str(x)[:-1])
#fea24['Feature'] = fea24['Feature'].str.replace(r'\D', '').astype(int)


    ###########################################
    ###########################################
    

#calculating entrophy of dns.qry.name
import math, string, fileinput

def range_bytes (): return range(256)
def range_printable(): return (ord(c) for c in string.printable)
def H(data, iterator=range_bytes):
    if not data:
        return 0
    entropy = 0
    for x in iterator():
        p_x = float(data.count(chr(x)))/len(data)
        if p_x > 0:
            entropy += - p_x*math.log(p_x, 2)
    return entropy

def main ():
    for row in fileinput.input():
        string = row.rstrip('\n')
        print ("%s: %f" % (string, H(string, range_printable)))



dfToList = fea25['Feature'].tolist()
fea25j = []

for str in dfToList:   
    fea25j.append( ("%f" % H(str, range_printable)) )
    
fea25i = pd.DataFrame(fea25j) 
fea25i.columns = ['Feature']
fea25i['group_no'] = fea25['group_no'].values
fea25i['Name'] = fea25['Name'].values
fea25 = fea25i
fea25 = fea25[['Name', 'Feature', 'group_no']]


dfToList = fea27['Feature'].tolist()
fea27j = []

for str in dfToList:   
    fea27j.append( ("%f" % H(str, range_printable)) )
    
fea27i = pd.DataFrame(fea27j) 
fea27i.columns = ['Feature']
fea27i['group_no'] = fea27['group_no'].values
fea27i['Name'] = fea27['Name'].values
fea27 = fea27i
fea27 = fea27[['Name', 'Feature', 'group_no']]

"""
    ###########################################
    ###########################################
    
#saving all features seperitly    

#to save
def save(fea5,name):
    fea5.to_excel(name+".xlsx", index=False)    
save(fea5,"fea5")
    
def save(fea28,name):
    fea28.to_excel(name+".xlsx", index=False)    
save(fea28,"fea28")

def save(fea27,name):
    fea27.to_excel(name+".xlsx", index=False)    
save(fea27,"fea27")


def save(fea2,name):
    fea2.to_excel(name+".xlsx", index=False)    
save(fea2,"fea2")
    
def save(fea25,name):
    fea25.to_excel(name+".xlsx", index=False)    
save(fea25,"fea25")

def save(fea26,name):
    fea26.to_excel(name+".xlsx", index=False)    
save(fea26,"fea26")

#to save
#save(final_form,"ff")
#def save(final_form,name):
    #final_form.to_excel(name+".xlsx", index=False)    
    
#to save
#save(final_form,"ff")
#def save(final_form,name):
    #final_form.to_excel(name+".xlsx", index=False)

"""


# build a new DataFrame from the futures we pulled
fea0.columns = ['Name','Type','group_no']
fea2.columns = ['Name','Ip_len','group_no']
fea3.columns = ['Name','Frame_len','group_no']
fea12.columns = ['Name','Ip.flag.df','group_no']
fea25.columns = ['Name','dns.qry.name','group_no']
fea26.columns = ['Name','dns.qry.name.len','group_no']
fea0=pd.get_dummies(fea0, prefix=['Type'], columns=['Type'])
#fea7.columns = ['Name','Frame.cap_len','group_no']
#fea11.columns = ['Name','Ip.flag.rb','group_no']
#fea13.columns = ['Name','Ip.flag.mf','group_no']

fea4.columns = ['Name','Type','group_no']
fea5.columns = ['Name','Ip_len','group_no']
fea6.columns = ['Name','Frame_len','group_no']
fea23.columns = ['Name','Ip.flag.df','group_no']
fea27.columns = ['Name','dns.qry.name','group_no']
fea28.columns = ['Name','dns.qry.name.len','group_no']
fea4=pd.get_dummies(fea4, prefix=['Type'], columns=['Type'])
#fea21.columns = ['Name','Frame.cap_len','group_no']
#fea22.columns = ['Name','Ip.flag.rb','group_no']
#fea24.columns = ['Name','Ip.flag.mf','group_no']


# combaine DataFrames
frames0 = [fea0, fea2, fea3, fea12, fea25, fea26]
frames1 = [fea4, fea5, fea6, fea23, fea27, fea28]

prototype0 = pd.concat(frames0)
prototype1 = pd.concat(frames1)

prototype0=prototype0.sort_index(axis=0, level=None, ascending=True, inplace=False,
                                 kind='quicksort', na_position='last', sort_remaining=True,
                                 by=None)

prototype1=prototype1.sort_index(axis=0, level=None, ascending=True, inplace=False,
                                 kind='quicksort', na_position='last', sort_remaining=True,
                                 by=None)


"""########################################
###########################################
#dummy encoding ///it does work but don't do it
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
########################################"""


del prototype1['Name']
del prototype0['Name']

#nan=0 for calculating
prototype01=prototype0.fillna(0)
prototype11=prototype1.fillna(0)

prototype01['dns.qry.name'] = prototype01['dns.qry.name'].astype(float)
prototype11['dns.qry.name'] = prototype11['dns.qry.name'].astype(float)

#calculating
aggregation_functions = {'Frame_len': 'sum', 'Ip_len': 'sum', 'Type_pcap_file': 'sum', 'Ip.flag.df': 'sum', 'dns.qry.name.len': 'sum', 
                         'dns.qry.name': 'sum', 'group_no': 'first'}
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

i= 0
while(i<1):
        i= i + 1
        
        print(i)
        
        final_form = final_form.sample(frac=1)
        X = final_form.iloc[:, 0:6].values
        y = final_form.iloc[:, 6].values


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
        from keras.layers.advanced_activations import LeakyReLU, PReLU
        from keras.models import model_from_json
        
        # initialising the ANN
        classifier = Sequential()
        
        # adding the input layer and first hiden layer
        classifier.add(Dense(activation="relu", input_dim=6, units=18, kernel_initializer="uniform"))
        
        # adding the second hiden layer with Dropout
        classifier.add(Dense(activation="relu", units=20, kernel_initializer="uniform"))
        classifier.add(Dropout(p = 0.25))
        
        classifier.add(Dense(activation="relu", units=11, kernel_initializer="uniform"))
        
        classifier.add(Dense(activation="relu", units=7, kernel_initializer="uniform"))
        
        classifier.add(Dense(activation="relu", units=17, kernel_initializer="uniform"))
        classifier.add(Dropout(p = 0.2))
        
        # adding the output layer
        classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
        
        # compiling the ANN
        classifier.compile(optimizer = 'adamax', loss = 'mse', metrics = ['accuracy'])
        
        # fitting the ANN to the training set
        classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 50) 
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)


# Predicting the single packet
new_predection = classifier.predict(sc.transform(np.array([[3, 7, 0, 15, 0, 7]])))



# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=8, units=6, kernel_initializer="uniform"))
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
    classifier.add(Dense(activation="relu", input_dim=8, units=6, kernel_initializer="uniform"))
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



###################################
###################################
######other attamps for rnn #######
###################################
###################################



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
for i in range(10, 82736):
    X_train.append(training_set_scaled[i-10:i, 0])
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
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# define and fit the model
def get_model(trainX, trainy):
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
    return regressor

# fit model
regressor = get_model(X_train, y_train)

# predict probabilities for test set
yhat_probs = regressor.predict(X, verbose=0)
# predict crisp classes for test set
yhat_classes = regressor.predict_classes(X, verbose=0)

yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y, yhat_classes)
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(y, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y, yhat_classes)
print(matrix)


##############################################

# serialize model to JSON
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

########################################################

from keras.models import load_model
# evaluate the model
scores = classifier.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))

# save model and architecture to single file
classifier.save('/home/ellenfel/Desktop/classifier.h5')

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('/home/ellenfel/Desktop/classifier.h5')
print("Saved classifier to disk")

# Save predictions for future checks
predictions = classifier.predict(X_test)

# Check that the state is preserved
new_predictions = new_model.predict(X_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Note that the optimizer state is preserved as well:
# you can resume training where you left off.


# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
 
# load model
classifier = load_model('classifier.h5')
# summarize model.
classifier.summary()

# load dataset
dataset = loadtxt("dataset.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
y = dataset[:,6]
# evaluate the model
score = classifier.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100))

classifier = load_model(fp_path+"classifier.h5")
preds = classifier.predict_classes(X_test)
prob = classifier.predict_proba(X_test)
print(preds, prob)

#predict
new_predection = classifier.predict(sc.transform(np.array([[3, 0, 0, 0, 0, 0]])))


##########################################

#picle
import pickle
outfile=open("model.h",'wb')
pickle.dump(classifier,outfile)
outfile.close()
 
