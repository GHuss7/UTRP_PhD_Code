# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:14:04 2021

@author: 17832020
"""
# https://www.datacamp.com/community/tutorials/deep-learning-python?utm_source=adwords_ppc&utm_campaignid=1658343524&utm_adgroupid=63833881895&utm_device=c&utm_keyword=%2Bkeras%20%2Btutorial&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=319558765402&utm_targetid=aud-299261629574:kwd-321066923947&utm_loc_interest_ms=&utm_loc_physical_ms=1028763&gclid=CjwKCAiAyc2BBhAaEiwA44-wW7FyBA_qSTOAkSySmSaY6BkTqVdn7q5PlcgC65VM8tnOZghxIDdVahoCDJ8QAvD_BwE

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense, Dropout
import keras.backend as K
import kerastuner as kt
from kerastuner import HyperModel, RandomSearch, Hyperband, BayesianOptimization
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
import os
from matplotlib import pyplot

from sklearn.model_selection import StratifiedKFold

#%% Load dictionaries and required settings

param_ML = {
'train_ratio' : 0.90, # training ratio for data
'val_ratio' : 0.05, # validation ratio for data
'test_ratio' : 0.05, # testing ratio for data
'min_f_1' : 13,
'max_f_1' : 70,
'min_f_2' : 4,
'max_f_2' :82,
'hp_tuning' : True
}

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

def normalise_data_UTRFSP(X,Y,param_ML):
    """A function to normalise data"""
    Y_norm = np.zeros(Y.shape)
    Y_norm[:,0] = (Y[:,0] - param_ML['min_f_1'])/(param_ML['max_f_1'] - param_ML['min_f_1'])
    Y_norm[:,1] = (Y[:,1] - param_ML['min_f_2'])/(param_ML['max_f_2'] - param_ML['min_f_2'])
    return X, Y_norm

def recast_data_UTRFSP(X_norm,Y_norm,param_ML):
    """A function to recast normalised data"""
    Y_rec = np.zeros(Y_norm.shape)
    Y_rec[:,0] = Y_norm[:,0] * (param_ML['max_f_1'] - param_ML['min_f_1']) + param_ML['min_f_1']
    Y_rec[:,1] = Y_norm[:,1] * (param_ML['max_f_2'] - param_ML['min_f_2']) + param_ML['min_f_2']
    return X_norm, Y_rec

class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def build(self, hp):
        model = Sequential()
        model.add(
            Dense(
                units=hp.Int('units', 8, 64, 4, default=8),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),
                input_shape=input_shape
            )
        )
        
        model.add(
            Dense(
                units=hp.Int('units', 16, 64, 4, default=16),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu')
            )
        )
        
        model.add(
            Dropout(
                hp.Float(
                    'dropout',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.005,
                    step=0.01)
            )
        )
        
        model.add(Dense(1))
        
        model.compile(
            optimizer='rmsprop',loss='mse',metrics=['mse']
        )
        
        return model

#%% Load data

X, Y = hf.load_data_UTRFSP_data("../../../Data/Training_data_UTRFSP_Mandl/Combined_Data.csv", "Mandl_Data") # nb the number of routes are still hardcoded

# Data pre-processing: Normalisation
X_norm, Y_norm = normalise_data_UTRFSP(X,Y,param_ML)


# Split data 
x_train, x_val, x_test, y_train, y_val, y_test = hf.split_data(X_norm,Y_norm,param_ML)
    
seed = 7
np.random.seed(seed)

#%% Model construction
if not param_ML['hp_tuning']:
    model = Sequential()
    model.add(Dense(126, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(90, activation='relu'))
    model.add(Dense(21, activation='relu'))
    model.add(Dense(2)) # default activation function is linear
    model.compile(optimizer='adam', loss=custom_distance_loss_function, #loss used to be mse
                  metrics=['mae', 'mape'])
    
    hp = kt.HyperParameters()
    
    Dense(
        units=hp.Int(
            'units',
            min_value=32,
            max_value=512,
            step=32,
            default=128
        ),
        activation=hp.Choice(
            'dense_activation',
            values=['relu', 'tanh', 'sigmoid'],
            default='relu'
        )
    )

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
    
    _, y_test_rec = recast_data_UTRFSP(False, y_test, param_ML)
    _, y_pred_rec = recast_data_UTRFSP(False, y_pred, param_ML)
    
    
    hf.print_evaluation(y_test, y_pred)
    
    #%% Results saving
    prediction_results_stacked = np.hstack([y_test_rec, y_pred_rec])
    np.savetxt("Predictions/NN_predictions_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", prediction_results_stacked, delimiter=",")
    
    #%% Save the model
    model.save("Saved_models/Model_Test_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    #model.save(f"Saved_models/Model_2_35_epochs_L126_90_21_2_MSE_{loss_mse*100:.0f}")


#%% Other Tests
# https://towardsdatascience.com/hyperparameter-tuning-with-keras-tuner-283474fbfbe
else:
    input_shape = (x_train.shape[1],)
    hypermodel = RegressionHyperModel(input_shape)
    
    # RandomSearch
    tuner_rs = RandomSearch(
            hypermodel,
            objective=custom_distance_loss_function,
            seed=42,
            max_trials=10,
            executions_per_trial=2,
            directory=os.path.normpath('D:/ML_Keras_Tuner/UTRFSP/Tests_RandomSearch'),
            project_name='Test_1'
            )
    
    tuner_rs.search(x_train, y_train, epochs=10, validation_split=0.2, verbose=0)
    
    best_model = tuner_rs.get_best_models(num_models=1)[0]
    loss, mse = best_model.evaluate(x_test, y_test)
    print(f'Performance metrics: \tLoss: {loss:.4f} \tMSE: {mse:.4f}')
    best_model.save("Tuned_models/RS_Test_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    
    # Hyperband
    tuner_hb = Hyperband(
            hypermodel,
            max_epochs=5,
            objective=custom_distance_loss_function,
            seed=42,
            executions_per_trial=2,
            directory=os.path.normpath('D:/ML_Keras_Tuner/UTRFSP/Tests_HyperBand'),
            project_name='Test_1'
        )
    tuner_hb.search(x_train, y_train, epochs=10, validation_split=0.2, verbose=0)
    best_model = tuner_hb.get_best_models(num_models=1)[0]
    loss, mse = best_model.evaluate(x_test, y_test)
    print(f'Performance metrics: \tLoss: {loss:.4f} \tMSE: {mse:.4f}')
    best_model.save("Tuned_models/HB_Test_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # BayesianOptimization
    tuner_bo = BayesianOptimization(
            hypermodel,
            objective=custom_distance_loss_function,
            max_trials=10,
            seed=42,
            executions_per_trial=2,
            directory=os.path.normpath('D:/ML_Keras_Tuner/UTRFSP/Tests_BayesianOptimization'),
            project_name='Test_1'
        )
    tuner_bo.search(x_train, y_train, epochs=10, validation_split=0.2, verbose=0)
    best_model = tuner_bo.get_best_models(num_models=1)[0]
    loss, mse = best_model.evaluate(x_test, y_test)
    print(f'Performance metrics: \tLoss: {loss:.4f} \tMSE: {mse:.4f}')
    best_model.save("Tuned_models/BO_Test_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
