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
#import os
#cwd = os.getcwd() 
#os.chdir("../DNN_own_UTRP")
import dnn_helper_functions as hf
#os.chdir(cwd)

import numpy as np
import pandas as pd
import time
import datetime

from sklearn.model_selection import StratifiedKFold


#%% Load data

X, Y = hf.load_data_UTRFSP_data("Training_data/Combined_Data.csv", "Mandl_Data") # nb the number of routes are still hardcoded

#%% Split data

param_ML = {
'train_ratio' : 0.80, # training ratio for data
'val_ratio' : 0.10, # validation ratio for data
'test_ratio' : 0.10} # testing ratio for data

x_train, x_val, x_test, y_train, y_val, y_test = hf.split_data(X,Y,param_ML)

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

#%% Model construction
model = Sequential()
model.add(Dense(21, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(2)) # default activation function is linear
model.compile(optimizer='adam', loss='mse',
              metrics=['mae',
                       tf.keras.metrics.MeanAbsolutePercentageError()])

#%% Model training
t1 = time.time()
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    update_freq="epoch",
    profile_batch=2,
    embeddings_freq=0,
    embeddings_metadata=None)

model.fit(x_train, y_train, epochs=50, verbose=1, batch_size=10, callbacks=[tb_callback])
t2 = time.time()
print (f"Training time: {t2-t1:.2f}s")

#%% Model evaluation
loss_mse, mae, mape = model.evaluate(x_val, y_val)
print(f'Performance metrics (AVG): \tMSE: {loss_mse:.2f} \tMAE: {mae:.2f} \tMAPE: {mape:.2f}')

#%% Model predictions and evaluations
y_pred = model.predict(x_test)
hf.print_evaluation(y_test, y_pred)

#%% Results saving
results_stacked = np.hstack([y_val, y_pred])
np.savetxt("Predictions/NN_predictions_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", results_stacked, delimiter=",")

#%% Save the model
model.save("Saved_models/Model_2")
