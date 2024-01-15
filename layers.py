import numpy as np

# overall structure for any neural network layer
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    # takes in input, changes it based on type of layer (ex. Dense layer will apply weights and biases)
    def forward(self, input):
        pass
    # takes in previous gradient to continue back propagation
    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # randomize parameters
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
    def forward(self, input):
        # Y = WX + B
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    def backward(self, output_gradient, learning_rate):
         # all gradients come from dE/dY, or error with respect to output
         # breakig down dE/dY into dE/dY_1 + dE/dY_2 + dE/dY_3 + ... and finding individual derivatives (ex. dE/dW_21) can lead back to these generalized gradients
         next_output_gradient = np.dot(self.weights.T, output_gradient)
         weights_gradient = np.dot(output_gradient, self.input.T)
         self.weights -= learning_rate * weights_gradient
         biases_gradient = output_gradient
         self.biases -= learning_rate * biases_gradient
         # backprop method in each layer takes the previous output gradient as input, so we return it
         return next_output_gradient
    def to_string(self):
        print("Dense")
    
class Activation(Layer):
    def __init__(self, nonlinear, nonlinear_prime):
        # nonlinear function is defined in sub-class; this class is essentially abstract - a structure
        self.nonlinear = nonlinear
        self.nonlinear_prime = nonlinear_prime
    def forward(self, input):
        self.input = input
        # running non-linear function on input
        return self.nonlinear(self.input)
    def backward(self, output_gradient, learning_rate):
        # derivative of error with respect to non-linear function can be found similar to gradients in Dense class
        return np.multiply(output_gradient, self.nonlinear_prime(self.input))
    
class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        def relu_prime(x):
            slopes = []
            for value in x:
                if value <= 0:
                    slopes.append(0)
                else:
                    slopes.append(1)
            return np.array(slopes).reshape(len(slopes), 1)
        super().__init__(relu, relu_prime)
    def to_string(self):
        print("ReLU")

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        # known derivative of tanh
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        # sending parameters to parent class
        super().__init__(tanh, tanh_prime)
    def to_string(self):
        print("Tanh")

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
    def to_string(self):
        print("Softmax")
    
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
    def to_string(self):
        print("Sigmoid")

class Recurrent(Layer):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # initialize parameters
        self.hidden_weights = np.random.randn(hidden_size, hidden_size)
        self.input_weights = np.random.randn(hidden_size, input_size)
        self.biases = np.random.randn(hidden_size, 1)

        # hidden state that is passed to the next layer
        self.hidden_state = np.zeros((hidden_size, 1))