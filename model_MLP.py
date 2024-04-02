import numpy as np

class MLP_solver():
    def __init__(self, deepth, layer_size, input_size, output_size):
        self.deepth = deepth
        self.layer_size = layer_size
        self.input_size = input_size
        self.output_size = output_size

        #initialize weights and bias
        self.weights = [np.random.randn(input_size, layer_size)]
        self.bias = [np.random.randn(layer_size)]
        for i in range(1, deepth-1):
            self.weights.append(np.random.randn(layer_size, layer_size))
            self.bias.append(np.random.randn(layer_size))
        self.output_weights = np.random.randn(layer_size, output_size)
        self.output_bias = np.random.randn(output_size)

        #define activation function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        #define forward pass
        def forward(self, x):
            hidden = x
            for i in range(self.deepth):
                hidden = sigmoid(np.dot(hidden, self.weights[i]) + self.bias[i])
                hidden = self.sigmoid(hidden)
            output = np.dot(hidden, self.output_weights) + self.output_bias
            return output


