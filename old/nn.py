from old.nn_methods import *
import json

def open_file():
    with open("climbs.json", "r") as f:
        raw_text = f.read()
    return json.loads(raw_text)

# organizing data
# all climbs get put in numpy array of size 198x25907
def parse_file(climbs):
    data = []
    grades = []
    for index, climb in enumerate(climbs):
        climb = climbs[climb]
        holds = [move["Description"] for move in climb["Moves"]]
        grade = font_to_num[climb["Grade"]]
        vector = one_hot(holds)
        data.append(vector)
        data[index].insert(0, grade)
    return np.array(data)

def train_and_test(data):
    data_train = data[:20000].T
    y_train = data_train[0]
    x_train = data_train[1:]
    data_test = data[20000:].T
    y_test = data_test[0]
    x_test = data_test[1:]
    return x_train, y_train, x_test, y_test

def main():
    climbs = open_file()
    data = parse_file(climbs)
    x_train, y_train, x_test, y_test = train_and_test(data)
    # initial (randomized) weights and biases
    W1, b1, W2, b2 = init_params()
    # weights and biases after training
    W1, b1, W2, b2 = gradient_descent(x_train, y_train, 10, 0.1, W1, b1, W2, b2)

main()