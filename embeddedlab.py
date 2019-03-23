# preprocessing template

# importing the libraries
import numpy as np
import tensorflow as tf
import re
import time
import pandas as pd

# Importing the dataset as list
lines = open('captured', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Importing the dataset as list as DataFrame
dataset = pd.read_fwf('captured', sep=" ", header = None)

# adding group no
dataset["group_no"] = dataset.T.isnull().all().cumsum()

# dropping the firs two column
to_drop = [0, 1]
dataset.drop(to_drop, inplace=True, axis=1)

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
    

"""     
clean_linesX
for line in dataset:
    clean_linesX.append(clean_text(line))
"""

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
dataset_train, dataset_test= train_test_split(dataset, test_size = 0.2, random_state = 0)
    


    
    

