class RNN:
    def __init__(self, recurrent_layer, recurrent_activation, dense_layer, dense_activation, dense_layer2, final_activation_layer):
        self.recurrent_layer = recurrent_layer
        self.recurrent_activation = recurrent_activation
        self.dense_activation = dense_activation
        self.dense_layer = dense_layer
        self.dense_layer2 = dense_layer2
        self.final_activation_layer = final_activation_layer