# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:00:29 2021

@author: 17832020
"""

"""Reference: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def sigmoid_derivative(x):
    return (sigmoid(x)*(1 - sigmoid(x)))


class NeuralNetwork:
    def __init__(self, x, y, neurons_per_layer):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],neurons_per_layer) 
        self.weights2   = np.random.rand(neurons_per_layer,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
    def estimate(self, x):
        layer1 = sigmoid(np.dot(x, self.weights1))
        output = sigmoid(np.dot(layer1, self.weights2))
        return output
        
        
x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0],[1],[1],[0]])


#data = np.array(pd.read_csv("Data_for_analysis.csv"))
#x = data[:,3:]
#y1 = data[:,1]

NN = NeuralNetwork(x,y,4)

iterations = 3000

loss = np.zeros(iterations)

for i in range(iterations):
    NN.feedforward()
    NN.backprop()
    loss[i] = sum((NN.y - NN.output)**2)
    
NN.output

plt.plot(np.linspace(1, iterations,iterations), loss) 
plt.xlabel("Iteration") 
plt.ylabel("Loss") 
plt.show()

NN.estimate(np.array([0,0,1]))

NN.estimate(np.array([0,1,1]))

