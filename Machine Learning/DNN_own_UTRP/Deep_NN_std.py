# -*- coding: utf-8 -*-
"""
Deep Neural Network for Image Classification: Application

Created on Tue Feb  9 07:48:17 2021

@author: 17832020
"""

#%% 1 - Packages

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import pandas as pd
import datetime
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import dnn_helper_functions as hf

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#np.random.seed(1)

#%% 2 - Dataset

train_x_orig, train_y, test_x_orig, test_y = hf.load_data_UTRP_routes("Data_for_analysis.csv", 6000, "Mandl_Data")

X, Y = hf.load_data_UTRP_data("Data_for_analysis.csv", "Mandl_Data")

# Explore your dataset 
m_train = train_x_orig.shape[1]
m_test = test_x_orig.shape[1]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples 
#train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
#test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_orig #/(1/5)
test_x = test_x_orig #/(1/5)

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

#%% 5 - L-layer Neural Network

# def initialize_parameters_deep(layers_dims):
#     ...
#     return parameters 
# def L_model_forward(X, parameters):
#     ...
#     return AL, caches
# def compute_cost(AL, Y):
#     ...
#     return cost
# def L_model_backward(AL, Y, caches):
#     ...
#     return grads
# def update_parameters(parameters, grads, learning_rate):
#     ...
#     return parameters

# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, return_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = hf.initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = hf.L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = hf.compute_quadratic_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = hf.L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = hf.update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 10 == 0:
            print (f"Cost after iteration {i}: {round(cost,4)}")
        if (print_cost or return_cost) and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs[1:]))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    if return_cost:
        return parameters, costs
    
    return parameters

### CONSTANTS ###
if False:
    layers_dims = [train_x_orig.shape[0], 50, 50, 30, 20, 1] #  4-layer model
    
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.009, num_iterations = 1, print_cost = True)
    
    pred_train = hf.predict_real_numbers(train_x, train_y, parameters)
    
    pred_test = hf.predict_real_numbers(test_x, test_y, parameters)
    pred_test = hf.predict_real_numbers(test_x, test_y, parameters,save_predictions=False)


#%% 6) Results Analysis


if True:
    layers_dims_configs= [[train_x_orig.shape[0], 15, 30, 15, 1],
                          [train_x_orig.shape[0], 21, 30, 15, 1],
                          [train_x_orig.shape[0], 50, 10, 1]]
    
    learning_rates = [0.01, 0.009, 0.0085, 0.008]
    num_iterations = 1000
    counter = 0
    experiment_costs = np.zeros((len(layers_dims_configs)*len(learning_rates),num_iterations//10))

    experiment_results = pd.DataFrame(columns=["Exp", "Layers", "Learning rate", "Training time [s]", "Accuracy", "R2", "MAE", "MSE", "Iterations"])
    
    for layers_dims in layers_dims_configs:
        for j, learning_rate in enumerate(learning_rates):
            t1 = time.time()
            parameters, costs = L_layer_model(train_x, train_y, layers_dims=layers_dims, learning_rate=learning_rate, 
                                       num_iterations=num_iterations, print_cost=False, return_cost=True)
            t2 = time.time()
            
            pred_train = hf.predict_real_numbers(train_x, train_y, parameters)
            pred_test = hf.predict_real_numbers(test_x, test_y, parameters)
            
            r2 = round(r2_score(test_y.T, pred_test.T),4)
            mae = round(mean_absolute_error(test_y.T, pred_test.T),4)
            mse = round(mean_squared_error(test_y.T, pred_test.T),4)
            accuracy = round(1 - mae/np.average(test_y),4)
            training_time = round(t2-t1, 6)
            
            experiment_costs[counter,:] = np.array(costs)/m_train # the average cost per training example
            experiment_results.loc[len(experiment_results)] = ["E"+str(counter), str(layers_dims), learning_rate, training_time, accuracy, 
                                                               r2, mae, mse, num_iterations]
            counter += 1
            print (f"Test {str(layers_dims)} HL given LR of {learning_rate}:\tA: {accuracy} \t R2: {r2} \t MAE: {mae} \t MSE: {mse} [{training_time}s]")
    
    # Save the test results
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_results.to_csv("Analyses/NN_analysis_results_"+time_stamp+".csv")
    experiment_costs_df = pd.DataFrame(experiment_costs.T, columns = ["E"+str(nr) for nr in range(experiment_costs.shape[0])])
    experiment_costs_df.to_csv("Analyses/NN_analysis_results_"+time_stamp+"_costs.csv")
    print(time_stamp)
    
    # Plot the legend entries
    combined_list = []
    for ld in layers_dims_configs:
        for lr in learning_rates:
            combined_list.append(str(ld) +" "+ str(lr))
    print(combined_list)
    plt.legend(combined_list)
    
if False:
    t1 = time.time()
    parameters, costs = L_layer_model(train_x, train_y, layers_dims=[train_x_orig.shape[0], 21, 30, 15, 1], learning_rate=0.01, 
                                       num_iterations=2000, print_cost=False, return_cost=True) # best config so far
    t2 = time.time()
            
    pred_train = hf.predict_real_numbers(train_x, train_y, parameters)
    pred_test = hf.predict_real_numbers(test_x, test_y, parameters)
    
    r2 = round(r2_score(test_y.T, pred_test.T),4)
    mae = round(mean_absolute_error(test_y.T, pred_test.T),4)
    mse = round(mean_squared_error(test_y.T, pred_test.T),4)
    accuracy = round(1 - mae/np.average(test_y),4)
    training_time = round(t2-t1, 6)            
    print (f"Test {str(layers_dims)} HL given LR of {learning_rate}:\tA: {accuracy} \t R2: {r2} \t MAE: {mae} \t MSE: {mse} [{training_time}s]")

