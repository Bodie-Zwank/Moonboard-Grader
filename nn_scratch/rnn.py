from layers import Dense, Recurrent, Tanh, Sigmoid, Softmax, ReLU
from networks import RNN
import numpy as np
from loss_functions import mse, mse_prime, bce, bce_prime
from get_data import *
from network_evaluation import *
import math


def train_network(X, Y, epochs, learning_rate, network):
    for epoch in range(epochs):
        error = 0
        lower = (epoch % 5) * 500
        upper = (epoch % 5) * 500 + 500
        x_batch, y_batch = X[lower:upper], Y[lower:upper]
        for x, y in zip(x_batch, y_batch):
            sequenced_x = [hold.reshape((-1, 1)) for hold in to_sequence(x)]
            for layer in network:
                if layer.to_string() == "recurrent":
                    for hold in sequenced_x:
                        output = layer.forward(hold)
                else:
                    output = layer.forward(output)
            error += bce(y, output)
            # backprop; initial gradient is gradient from final activation function
            gradient = bce_prime(y, output)
            for layer in reversed(network):
                if layer.to_string() == "recurrent":
                    for i in range(len(sequenced_x) - 1):
                        gradient = layer.backward(gradient, learning_rate)
                else:
                    gradient = layer.backward(gradient, learning_rate)
        error /= len(X)
        print(f"Epoch: {epoch}\tError: {error}")
    return network


def main():
    # getting data
    climbs = open_file()
    # training data and raw grades for evaluating network concisely
    x_train, y_train, raw_grades = parse_file(climbs)
    # initializing network options and layers
    network = [Recurrent(198, 50, Tanh()), Dense(50, 20), Tanh(), Dense(20, 16), Softmax()]
    epochs = 20
    learning_rate = 0.05
    trained_network = train_network(x_train, y_train, epochs, learning_rate, network)
    print(f"Average distance from correct grade: {evaluate_rnn(x_train, raw_grades, trained_network)}")

main()