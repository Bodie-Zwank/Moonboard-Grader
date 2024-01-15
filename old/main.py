import json
from old.hold import Hold
from old.climb import Climb
import tensorflow as tf
import numpy as np
import math

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

with open("climbs.json", "r") as f:
    raw_text = f.read()

climbs = json.loads(raw_text)

climb_vectors = []
benchmark_climb_vectors = []
grades = []
benchmark_grades = []

def initialize_climb_vectors():
    for index, climb in enumerate(climbs):
        climb = climbs[climb]
        current_climb = Climb([hold for hold in climb["Moves"]], climb["IsBenchmark"], climb["Name"], climb["Grade"])
        climb_vector = current_climb.to_vector()
        climb_vectors.append(climb_vector)
        grades.append(font_to_num[climb["Grade"]])
        if current_climb.benchmark:
            benchmark_climb_vectors.append(climb_vector)
            benchmark_grades.append(font_to_num[climb["Grade"]])

def train_model():
    x_train, x_test = np.array(climb_vectors[:20000]), np.array(climb_vectors[20000:])
    y_train, y_test = np.array(grades[:20000]), np.array(grades[20000:])
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=16, activation='tanh', input_dim=len(climb_vectors[0]), input_shape=(len(climb_vectors[0]),)))
    model.add(tf.keras.layers.Dense(units=16, activation='sigmoid'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    return model

def predict_grade(climb, model):
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probabilities = probability_model(climb)[0]
    most_likely_grade = 0
    max = 0
    for i in range(16):
        if probabilities[i] > max:
            max = probabilities[i]
            most_likely_grade = i
    grade = [grade for grade in font_to_num.keys() if font_to_num[grade] == most_likely_grade]
    return grade[0]

def evaluate_within_range(climbs, grades, model, range):
    total = 0
    within_range = 0
    for i, climb in enumerate(climbs):
        print(i)
        climb = np.array(climb)
        climb = climb.reshape((1, -1))
        grade = predict_grade(climb, model)
        off_by = abs(grades[i] - font_to_num[grade])
        if off_by <= range:
            within_range += 1
        total += 1
    print(f"Within {range}: {within_range / total}")

def average_distance(climbs, grades, model):
    count = 0
    distance = 0
    for i, climb in enumerate(climbs):
        climb = np.array(climb)
        climb = climb.reshape((1, -1))
        grade = predict_grade(climb, model)
        off_by = abs(grades[i] - font_to_num[grade])
        distance += off_by
        count += 1
    return distance / count

def main():
    initialize_climb_vectors()
    model = train_model()
    my_climb = np.array(Climb(["I5", "K9", "I12", "K14", "I18"]).to_vector())
    my_climb = my_climb.reshape((1, -1))
    print(f"Average distance from actual grade: {average_distance(climb_vectors[:1000], grades, model)}")
    print(predict_grade(my_climb, model))

main()
