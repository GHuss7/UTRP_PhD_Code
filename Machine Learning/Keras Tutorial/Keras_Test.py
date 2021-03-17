# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:14:04 2021

@author: 17832020
"""
# https://www.datacamp.com/community/tutorials/deep-learning-python?utm_source=adwords_ppc&utm_campaignid=1658343524&utm_adgroupid=63833881895&utm_device=c&utm_keyword=%2Bkeras%20%2Btutorial&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=319558765402&utm_targetid=aud-299261629574:kwd-321066923947&utm_loc_interest_ms=&utm_loc_physical_ms=1028763&gclid=CjwKCAiAyc2BBhAaEiwA44-wW7FyBA_qSTOAkSySmSaY6BkTqVdn7q5PlcgC65VM8tnOZghxIDdVahoCDJ8QAvD_BwE

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

import tensorflow as tf
import os
cwd = os.getcwd() 
os.chdir("../DNN_own_UTRP")
import dnn_helper_functions as hf
os.chdir(cwd)

import numpy as np
import time

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

#%% Load data
X, Y = hf.load_data_UTRP_data("Data_for_analysis.csv", "Mandl_Data")
X = X.T
Y = Y.T


# %% Test

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, 
                                                    random_state=4, shuffle=True)

seed = 7
np.random.seed(seed)

if False:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in kfold.split(X, Y):
        model = Sequential()
        model.add(Dense(21, input_dim=x_train.shape[1], activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        model.fit(X[train], Y[train], epochs=10, verbose=1)


t1 = time.time()

model = Sequential()
model.add(Dense(21, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_train, y_train, epochs=30, verbose=1, batch_size=10)

t2 = time.time()

# evaluate the keras model
_, performance_metric = model.evaluate(x_test, y_test)
print('Performance_metric: %.2f' % (performance_metric*100))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# make probability predictions with the model
y_pred = model.predict(x_test)

r2 = round(r2_score(y_test, y_pred),4)
mae = round(mean_absolute_error(y_test, y_pred),4)
mse = round(mean_squared_error(y_test, y_pred),4)
accuracy = round(1 - mae/np.average(y_test),4)

training_time = round(t2-t1, 6)            
print (f"Test:\tA: {accuracy} \t R2: {r2} \t MAE: {mae} \t MSE: {mse} [{training_time}s]")

