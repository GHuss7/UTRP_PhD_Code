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
import keras.backend as K

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
from matplotlib import pyplot

from sklearn.model_selection import StratifiedKFold

def custom_distance_loss_function(y_real, y_pred):
    """
    

    Parameters
    ----------
    y_real : tf.Tensor
        Real values of target variables y, where the rows are the entries and
        the columns are the different target variables, with shape 
        (batch_size, N_y).
    y_pred : tf.Tensor
        Predicted values of target variables y, where the rows are the entries 
        and the columns are the different target variables, with shape 
        (batch_size, N_y).

    Returns
    -------
    float64
        The sum of the euclidian distance between the points. Needs to be in 
        the form (batch_size,)

    """ 
    #N_y = y_pred.get_shape()[1] # number of predictions
    N_y = 2
    eval = K.square(y_pred - y_real)                           # (batch_size, 2)
    eval = K.sum(eval, axis=-1)
    eval = K.pow(eval,1/N_y)     # (batch_size,)
    #custom_loss_value = kb.mean(custom_loss_value)
    return eval

def mae_custom(y_true, y_pred):
            
    eval = K.square(y_pred - y_true)
    eval = K.mean(eval, axis=-1)
        
    return eval


#%% Load data

X, Y = hf.load_data_UTRFSP_data("../../../Data/Training_data_UTRFSP_Mandl/Combined_Data.csv", "Mandl_Data") # nb the number of routes are still hardcoded

#%% Split data

param_ML = {
'train_ratio' : 0.90, # training ratio for data
'val_ratio' : 0.05, # validation ratio for data
'test_ratio' : 0.05} # testing ratio for data

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
model.add(Dense(126, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(2)) # default activation function is linear
model.compile(optimizer='adam', loss=custom_distance_loss_function, #loss used to be mse
              metrics=['mae', 'mape'])

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

history = model.fit(x_train, y_train, epochs=35, verbose=1, batch_size=4)#, callbacks=[tb_callback])
t2 = time.time()

# plot metrics
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['mae'])
pyplot.plot(history.history['mape'])
pyplot.show()

print (f"Training time: {t2-t1:.2f}s")

#%% Model evaluation
loss_mse, mae, mape = model.evaluate(x_val, y_val)
print(f'Performance metrics (AVG): \tMSE: {loss_mse:.2f} \tMAE: {mae:.2f} \tMAPE: {mape:.2f}')

#%% Model predictions and evaluations
y_pred = model.predict(x_test)
hf.print_evaluation(y_test, y_pred)

#%% Results saving
results_stacked = np.hstack([y_test, y_pred])
np.savetxt("Predictions/NN_predictions_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", results_stacked, delimiter=",")

#%% Save the model
model.save("Saved_models/Model_Good_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
#model.save(f"Saved_models/Model_2_35_epochs_L126_90_21_2_MSE_{loss_mse*100:.0f}")
