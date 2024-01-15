from grade_conversion import font_to_num
from get_data import to_sequence
import numpy as np

def predict_nn(network, input):
    # next line not necessary, but follows other structures more cleanly
    output = input
    # forward propagation with given input
    for layer in network:
        output = layer.forward(output)
    # finding greatest value in probabilities
    return np.argmax(output)

def evaluate_nn(climbs, grades, network):
    count = 0
    distance = 0
    for i, climb in enumerate(climbs):
        # get predicted grade
        grade = predict_nn(network, climb)
        # find difference between predicted grade and actual grade
        off_by = abs(grades[i] - grade)
        distance += off_by
        count += 1
    return distance / count

def predict_rnn(network, x):
    # change climb to sequence of holds
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
    # return index of max value in predictions
    return np.argmax(output)

def evaluate_rnn(climbs, grades, network):
    count = 0
    distance = 0
    for i, climb in enumerate(climbs):
        # get predicted grade
        grade = predict_rnn(network, climb)
        # find difference between predicted grade and actual grade
        off_by = abs(grades[i] - grade)
        distance += off_by
        count += 1
    return distance / count