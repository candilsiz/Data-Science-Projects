#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:04:49 2023

@author: candilsiz
"""

# import numpy as np


# Test1 = np.loadtxt('test1.txt')

# inputTest1 = Test1[:,0]
# outputTest1 = Test1[:,1]


# Train1 = np.loadtxt('train1.txt')

# inputTrain1 = Train1[:,0]
# outputTrain1 = Train1[:,1]

# print(len(inputTest1))
# print(len(outputTest1))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    
    def __init__(self, x, y, h_neurons,lr = 0.00001): # LR 0.00001

        self.input = x
        self.h_neurons = h_neurons
        self.lr = lr
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.weights1 = self.initialize_weights(self.input.shape[1], h_neurons)
        self.weights2 = self.initialize_weights(h_neurons, 1)                

    
    def initialize_weights(self, size_in, size_out):
        return np.random.uniform(low=-0.1, high=0.1, size=(size_in, size_out))

    def forward_Propogate(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = np.dot(self.layer1, self.weights2)

    def backward_Propogate(self):
        calculated_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output)) 
        calculated_weights1 = np.dot(self.input.T,  np.dot(2*(self.y - self.output), self.weights2.T) * sigmoid_derivative(self.layer1)) 
    
        self.weights1 += self.lr * calculated_weights1
        self.weights2 += self.lr * calculated_weights2

    def train(self, epochs=80000): #100000
    
        for _ in range(epochs):
            self.forward_Propogate()
            self.backward_Propogate()

    def predict(self, x):
        self.input = x
        self.forward_Propogate()
        return self.output
    
def load_dataset(filename):
    
    df = pd.read_csv(filename, sep="\t", header=None)
    
    return df[0], df[1]


def evaluate_loss(nn, x, y):
    
    predictions = nn.predict(x).flatten()
    loss = np.mean((y - predictions) ** 2)
    
    return loss, np.std((y - predictions) ** 2)


def normalize_data(data):
    
    avg = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    return (data - avg) / std

def denormalize_data(normalized_data, original_data):
    
    avg = np.mean(original_data, axis=0)
    std = np.std(original_data, axis=0)
    
    return normalized_data * std + avg


def plot_dataset(x, y, y_original, title):
    
    plt.scatter(x, y, color='blue', label='Actual')
    x_curve = np.linspace(min(x), max(x), 1000).reshape(-1, 1)
    #print(x_curve)
    y_curve = denormalize_data(Neural_N.predict(normalize_data(x_curve)), y_original)
    #print(y_curve)
    plt.plot(x_curve, y_curve, color='red', label='Estimated')
    plt.title(title)
    plt.legend()
    plt.show()
    
    
def multiple_layer(hidden_units):

    results = []
    
    for h_units in hidden_layers:
        
        Neural_N = NeuralNetwork(input_train, output_train, h_neurons = h_units, lr=0.00001)
        Neural_N.train(epochs=80000)
    
        # Plotting each ANN with different number of hidden units
        plot_dataset(denormalize_data(input_train, input_train_orig), denormalize_data(output_train, output_train_orig), output_train_orig, f'Training data - {h_units} hidden units')
    
        train_loss, train_std = evaluate_loss(Neural_N, input_train, output_train)
        test_loss, test_std = evaluate_loss(Neural_N, input_test, output_test)
    
        results.append({
            'Hidden Units': h_units,
            'Train Loss': train_loss,
            'Train Std': train_std,
            'Test Loss': test_loss,
            'Test Std': test_std
        })
    
    df_Table_results = pd.DataFrame(results)

    
if __name__ == "__main__":
    input_train, output_train = load_dataset('train1.txt')
    
    #print("X:",input_train,"Y:",output_train)
    input_test, output_test = load_dataset('test1.txt')

    input_train_orig = np.array(input_train).reshape(-1, 1)
    output_train_orig = np.array(output_train).reshape(-1, 1)
    x_test_orig = np.array(input_test).reshape(-1, 1)

    input_train = normalize_data(input_train_orig)
    output_train = normalize_data(output_train_orig)
    input_test = normalize_data(x_test_orig)

    Neural_N = NeuralNetwork(input_train, output_train, h_neurons = 4, lr = 0.001)
    Neural_N.train(epochs=5000)

    plot_dataset(denormalize_data(input_train, input_train_orig), denormalize_data(output_train, output_train_orig), output_train_orig, 'Train data')
    #plot_dataset(denormalize_data(input_test, x_test_orig), denormalize_data(Neural_N.predict(x_test), output_train_orig), output_train_orig, 'Test set')
    
    hidden_layers = [2, 4, 8, 16, 32]
    
    multiple_layer(hidden_layers)

