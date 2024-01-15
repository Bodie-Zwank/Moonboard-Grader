# neural network using primarily OOP
from layers import Dense, ReLU, Tanh, Softmax, Sigmoid
from loss_functions import mse, mse_prime, bce, bce_prime
import numpy as np
from get_data import *
from network_evaluation import *
from grade_conversion import font_to_num, num_to_font

def train_network(X, Y, epochs, learning_rate, network):
    # loop to adjust network many times (gradient descent)
    for epoch in range(epochs):
        lower = (epoch % 5) * 500
        upper = (epoch % 5) * 500 + 500
        x_batch, y_batch = X[lower:upper], Y[lower:upper]
        error = 0
        # loop to train across all data
        for x, y in zip(x_batch, y_batch):
            # forward prop; x is first "output" since it is what comes out of the input layer
            output = x
            for layer in network:
                output = layer.forward(output)
            # keep running sum of error so average can be calculated
            #error += mse(y, output)
            error += bce(y, output)
            
            # backward prop; first gradient is gradient of error function
            #gradient = mse_prime(y, output)
            gradient = bce_prime(y, output)
            for layer in reversed(network):
                # layer.to_string()
                gradient = layer.backward(gradient, learning_rate)
        # to get average error
        error /= len(X)
        print(f"Epoch: {epoch}\tError: {error}")
    return network

def main():
    # getting data
    climbs = open_file()
    # training data and raw grades for evaluating network concisely
    x_train, y_train, raw_grades = parse_file(climbs)
    # initializing network options and layers
    epochs = 100
    learning_rate = 0.1
    network = [
        Dense(198, 16),
        Tanh(),
        Dense(16, 16),
        Sigmoid()
    ]
    # training network
    trained_network = train_network(x_train, y_train, epochs, learning_rate, network)
    climb = ["F5", "G8", "C10", "G13", "G17", "D18"]
    climb = np.reshape(one_hot(climb), (1, 198, 1))
    print(num_to_font[predict_nn(network, climb[0])])
    # evaluating average distance from correct grade (ex. 6C is one grade above 6B+ so it has a "distance" of 1)
    print(f"Average distance from correct grade: {evaluate_nn(x_train, raw_grades, trained_network)}")
    
main()
