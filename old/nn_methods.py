import json
from old.hold import Hold
from old.climb import *
import numpy as np
import math

# NEURAL NET METHODS AND DICTS

font_to_num = {
    "6B+": 0,
    "6C": 1,
    "6C+": 2,
    "7A": 3,
    "7A+": 4,
    "7B": 5,
    "7B+": 6,
    "7C": 7,
    "7C+": 8,
    "8A": 9,
    "8A+": 10,
    "8B": 11,
    "8B+": 12,
    "8C": 13,
    "8C+": 14,
    "9A": 15
}

def ReLU(output):
    return np.max(0, output)

def softmax(output):
    return np.exp(output) / np.sum(np.exp(output))

def one_hot(holds):
    vector = [0 for _ in range(18 * 11)]
    for hold in holds:
        row = 18 - int(hold[1])
        column = ord(hold[0]) - 65
        index = 11 * row + column
        vector[index] = 1
    return vector

def init_params():
    W1 = np.random.randn(16, 198)
    b1 = np.random.randn(16, 1)
    W2 = np.random.randn(16, 16)
    b2 = np.random.randn(16, 1)
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, input):
    U1 = W1 @ input + b1
    A1 = ReLU(U1)
    U2 = W2 @ input + b2
    A2 = softmax(U2)
    return U1, A1, U2, A2

def back_prop(U1, A1, U2, A2, W2, input, output):
    pass

def gradient_descent(X, Y, iterations, rate, W1, b1, W2, b2):
    # return updated params
    pass