import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

# %% Load data functions

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_data_UTFSP_frequencies(file_location, num_test_data):
    data = np.array(pd.read_csv(file_location))
    np.random.shuffle(data)
    train_x_orig = data[:-num_test_data,3:].T
    train_y = data[:-num_test_data,1]
    train_y = train_y.reshape(train_y.shape[0],1).T
    
    test_x_orig = data[-num_test_data:,3:].T
    test_y = data[-num_test_data:,1]
    test_y = test_y.reshape(test_y.shape[0],1).T
    
    return train_x_orig, train_y, test_x_orig, test_y

def load_data_UTRP_routes(file_location, num_test_data, name_input_data):

    mx_dist = pd.read_csv("../../Input_Data/"+name_input_data+"/Distance_Matrix.csv")
    mx_dist = mx_dist.iloc[:,1:mx_dist.shape[1]]
    mx_dist = mx_dist.values
    
    data = pd.read_csv(file_location)
    data = data.sample(frac=1)
    
    edge_list, edge_weights = get_links_list_and_distances(mx_dist)
    
    recast_decision_variable = np.zeros((len(edge_list)*6, data.shape[0]))
    
    num_edges = len(edge_list)
    
    for m_example in range(data.shape[0]): 
        temp_route_set = convert_routes_str2list(data.iloc[m_example,3])
        for route_nr, route in enumerate(temp_route_set): 
            for edge_nr in range(len(route) - 1):
                if route[edge_nr]<route[edge_nr+1]:
                    temp_edge = (route[edge_nr],route[edge_nr+1])
                else:
                    temp_edge = (route[edge_nr+1],route[edge_nr])
                    
                recast_decision_variable[route_nr*num_edges + edge_list.index(temp_edge),m_example] = 1
    
    train_x_orig = recast_decision_variable[:,:-num_test_data]
    train_y = np.array(data.iloc[:-num_test_data,1])
    train_y = train_y.reshape(train_y.shape[0],1).T
    
    test_x_orig = recast_decision_variable[:,-num_test_data:]
    test_y = np.array(data.iloc[-num_test_data:,1])
    test_y = test_y.reshape(test_y.shape[0],1).T
    
    return train_x_orig, train_y, test_x_orig, test_y

def load_data_UTRP_data(file_location, name_input_data):

    mx_dist = pd.read_csv("../../Input_Data/"+name_input_data+"/Distance_Matrix.csv")
    mx_dist = mx_dist.iloc[:,1:mx_dist.shape[1]]
    mx_dist = mx_dist.values
    
    data = pd.read_csv(file_location)
    data = data.sample(frac=1)
    
    edge_list, edge_weights = get_links_list_and_distances(mx_dist)
    
    recast_decision_variable = np.zeros((len(edge_list)*6, data.shape[0]))
    
    num_edges = len(edge_list)
    
    for m_example in range(data.shape[0]): 
        temp_route_set = convert_routes_str2list(data.iloc[m_example,3])
        for route_nr, route in enumerate(temp_route_set): 
            for edge_nr in range(len(route) - 1):
                if route[edge_nr]<route[edge_nr+1]:
                    temp_edge = (route[edge_nr],route[edge_nr+1])
                else:
                    temp_edge = (route[edge_nr+1],route[edge_nr])
                    
                recast_decision_variable[route_nr*num_edges + edge_list.index(temp_edge),m_example] = 1
    
    data_x_orig = recast_decision_variable
    data_y = np.array(data.iloc[:,1])
    data_y = data_y.reshape(data_y.shape[0],1).T
    
    
    return data_x_orig, data_y

def load_data_UTRFSP_data(file_location, name_input_data, load_all_y_values=True):

    mx_dist = pd.read_csv("../../Input_Data/"+name_input_data+"/Distance_Matrix.csv")
    mx_dist = mx_dist.iloc[:,1:mx_dist.shape[1]]
    mx_dist = mx_dist.values
    
    data = pd.read_csv(file_location)
    data = data.sample(frac=1)
    
    edge_list, edge_weights = get_links_list_and_distances(mx_dist)
    parameters_constraints = json.load(open("../../Input_Data/"+name_input_data+"/parameters_constraints.json"))
    con_r = parameters_constraints["con_r"]
    
    recast_decision_variable = np.zeros((len(edge_list)*con_r + con_r, data.shape[0]))
    
    num_edges = len(edge_list)
    
    for m_example in range(data.shape[0]): 
        temp_route_set = convert_routes_str2list(data["R_x"].iloc[m_example])
        for route_nr, route in enumerate(temp_route_set): 
            for edge_nr in range(len(route) - 1):
                if route[edge_nr]<route[edge_nr+1]:
                    temp_edge = (route[edge_nr],route[edge_nr+1])
                else:
                    temp_edge = (route[edge_nr+1],route[edge_nr])
                    
                recast_decision_variable[route_nr*num_edges + edge_list.index(temp_edge),m_example] = 1
        
        loc_f_0 = data.columns.get_loc("f_0")
        recast_decision_variable[-con_r:,m_example] = data.iloc[m_example, loc_f_0:con_r+loc_f_0] # adds the frequencies
    
    data_x_orig = recast_decision_variable
    if load_all_y_values:
        data_y = np.array(data[["F_3","F_4"]])
        data_y = data_y.reshape(data_y.shape[0],2)
    else:
        data_y = np.array(data["F_3"])
        data_y = data_y.reshape(data_y.shape[0],1)
    
    return data_x_orig.T, data_y

#%% Split data functions
def split_data(X,Y,param_ML):
    """
    

    Parameters
    ----------
    X : np.array
        Input data in matrix form with shape (m, N_x)
    Y : np.array
        Output data in matrix form with shape (m, N_y).
    param_ML : dictionary
        A dictionary containing the values for 'train_ratio', 'val_ratio'
        and 'test_ratio'.

    Returns
    -------
    x_train, x_val, x_test, y_train, y_val, y_test: all np.arrays
        Data split into the three sets needed for ML training.
    """
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=param_ML['test_ratio'], 
                                                        random_state=1, shuffle=True)
    # calculate the test ratio for the counterpart of the testing data
    calc_ratio = param_ML['val_ratio']/(param_ML['val_ratio']+param_ML['train_ratio'])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                      test_size=calc_ratio, 
                                                      random_state=1, shuffle=True)
    
    return x_train, x_val, x_test, y_train, y_val, y_test

#%% Route string to array manipulations
def get_links_list_and_distances(matrix_dist):
    # Creates a list of all the links in a given adjacency matrix and a 
    # corresponding vector of distances associated with each link
    #  Output: (links_list, links_distances) [list,float64]
    
    max_distance = matrix_dist.max().max() # gets the max value in the matrix
    matrix_dist_shape = (len(matrix_dist),len(matrix_dist[0])) # row = entry 0, col = entry 1, stores the values (efficiency)
    links_list_dist_mx = list() # create an empty list to store the links

    # from the distance matrix, get the links list
    for i in range(matrix_dist_shape[0]):
        for j in range(matrix_dist_shape[1]):
            val = matrix_dist[i,j]
            if val != 0 and val != max_distance and i<j: # i > j yields only single edges, and not double arcs
                links_list_dist_mx.append((i,j))

    # Create the array to store all the links' distances
    links_list_distances = np.int64(np.empty(shape=(len(links_list_dist_mx),1))) # create the array

    # from the distance matrix, store the distances for each link
    for i in range(len(links_list_dist_mx)): 
        links_list_distances[i] = matrix_dist[links_list_dist_mx[i]]
    
    return links_list_dist_mx, links_list_distances

def convert_routes_str2list(routes_R_str):
    # converts a string standarised version of routes list into a routes list
    routes_R_list = list()
    temp_list = list()
    flag_end_node = True
    for i in range(len(routes_R_str)):
        if routes_R_str[i] != "-" and routes_R_str[i] != "*":
            if flag_end_node:
                temp_list.append(int(routes_R_str[i]))
                flag_end_node = False
            else:
                temp_list[len(temp_list)-1] = int(str(temp_list[len(temp_list)-1]) + routes_R_str[i])
        else:   
            if routes_R_str[i] == "*":          # indicates the end of the route
                routes_R_list.append(temp_list)
                temp_list = list()
                flag_end_node = True
            else:
                if routes_R_str[i] == "-":
                    flag_end_node = True
    return routes_R_list

#%% Evaluation functions
def print_evaluation(y_val, y_pred):
    r2 = round(r2_score(y_val[:,0], y_pred[:,0]),4)
    mae = round(mean_absolute_error(y_val[:,0], y_pred[:,0]),4)
    mse = round(mean_squared_error(y_val[:,0], y_pred[:,0]),4)
    accuracy = round(1 - mae/np.average(y_val[:,0]),4)
    print (f"Test F_3:\tA: {accuracy} \t R2: {r2} \t MAE: {mae} \t MSE: {mse}")
    
    if y_val.shape[1] == 2:
        r2 = round(r2_score(y_val[:,1], y_pred[:,1]),4)
        mae = round(mean_absolute_error(y_val[:,1], y_pred[:,1]),4)
        mse = round(mean_squared_error(y_val[:,1], y_pred[:,1]),4)
        accuracy = round(1 - mae/np.average(y_val[:,1]),4)
        print (f"Test F_4:\tA: {accuracy} \t R2: {r2} \t MAE: {mae} \t MSE: {mse}")
    
        r2 = round(r2_score(y_val, y_pred),4)
        mae = round(mean_absolute_error(y_val, y_pred),4)
        mse = round(mean_squared_error(y_val, y_pred),4)
        accuracy = round(1 - mae/np.average(y_val),4)       
        print (f"Test AVG:\tA: {accuracy} \t R2: {r2} \t MAE: {mae} \t MSE: {mse}")
    


# %% Coursera course helper functions
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters     


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "relu")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def compute_quadratic_cost(AL, Y):
    """
    Implement the cost function defined by quadratic cost equation.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- quadratic cost
    """
    
    # Compute loss from aL and y.
    cost = 0.5 * np.sum(np.power(AL - Y, 2))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def grad_quadratic_cost(AL, Y):
    """
    Implement the gradient function for the Quadratic cost equation.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    grad -- gradient of quadratic cost
    """
    
    # Compute loss from aL and y.
    grad = AL - Y
    
    assert(grad.shape == AL.shape)
    
    return grad

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = grad_quadratic_cost(AL, Y)
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "relu")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def predict_real_numbers(X, Y, parameters, save_predictions=False):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    
    # Forward propagation
    predictions, caches = L_model_forward(X, parameters)
    accuracy = np.sum(1 - (np.abs(predictions - Y))/Y)/m
    
    if save_predictions:
        results_stacked = np.hstack([Y.T, predictions.T])
        np.savetxt("Predictions/NN_predictions_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", results_stacked, delimiter=",")


    return predictions
        
    

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))