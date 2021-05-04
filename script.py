#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
import numpy as np

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Recreate the exact same model purely from the file
classifier = keras.models.load_model('/home/ellenfel/Desktop/lab/dnsdetection_exps/classifier.h5')
print("Saved classifier to disk")
classifier.summary()

new_predection = classifier.predict(sc.transform(np.array([[(ip_len + 14), ip_len, 1, ip_flag_df, query_len ,entrophy ]])))


while True:
    while True:
        try:
            # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
            ip_len     = int(input("Please enter ip_len input: "))
            ip_flag_df = int(input("Please enter ip_flag_df(1 or 0) input: "))
            entrophy   = int(input("Please enter entrophy input: "))
            query_len  = int(input("Please enter query_len input: "))
            classifier.fit_transform()
            new_predection = classifier.predict(sc.transform(np.array([[(ip_len + 14), ip_len, 1, ip_flag_df, query_len ,entrophy ]])))
            new_predection = (new_predection > 0.5)
            
            print(new_predection)
       
        except ValueError:
            print("Sorry, I didn't understand that.")
            #better try again... Return to the start of the loop
            continue         
             
        else:

            #we're ready to exit the loop.
            break
    if ip_len >= 3000: 
        print("error due to ---- ")

    elif ip_flag_df >= 3000: 
        print("error due to ---- ")

    elif entrophy >= 3000: 
        print("error due to ---- ")

    elif query_len >= 3000: 
        print("error due to ---- ")
    
    else:
        print("Starting next sequence")
