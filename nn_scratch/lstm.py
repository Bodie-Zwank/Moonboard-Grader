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
            sequenced_x = to_sequence(x)
            for hold in sequenced_x:
                hold = hold.reshape((-1, 1))
                output = network.recurrent_layer.forward(hold)
                output = network.recurrent_activation.forward(output)
            # passing through two dense layers
            output = network.dense_layer.forward(output)
            output = network.dense_activation.forward(output)
            output = network.dense_layer2.forward(output)
            output = network.final_activation_layer.forward(output)
            error += bce(y, output)
            # backprop; initial gradient is gradient from final activation function
            gradient = bce_prime(y, output)
            gradient = network.final_activation_layer.backward(gradient, learning_rate)
            gradient = network.dense_layer2.backward(gradient, learning_rate)
            gradient = network.dense_activation.backward(gradient, learning_rate)
            gradient = network.dense_layer.backward(gradient, learning_rate)
            for i in range(len(sequenced_x) - 1):
                gradient = network.recurrent_activation.backward(gradient, learning_rate)
                gradient = network.recurrent_layer.backward(gradient, learning_rate)
        error /= len(X)
        print(f"Epoch: {epoch}\tError: {error}")
    return network


def main():
    # getting data
    climbs = open_file()
    # training data and raw grades for evaluating network concisely
    x_train, y_train, raw_grades = parse_file(climbs)
    # initializing network options and layers
    network = RNN(Recurrent(198, 50), ReLU(), Dense(50, 20), Tanh(), Dense(20, 16), Softmax())
    epochs = 20
    learning_rate = 0.05
    trained_network = train_network(x_train, y_train, epochs, learning_rate, network)
    print(f"Average distance from correct grade: {evaluate_rnn(x_train, raw_grades, trained_network)}")

main()