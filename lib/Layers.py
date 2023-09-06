from lib.Tensor import Tensor
import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Compute the output of this layer using `input_data`.
        """

        raise NotImplementedError
    
    def backward(self, output_gradient):
        """
        Compute the input gradient using `output_gradient` and
        chain it with the local gradient.
        """
        raise NotImplementedError
    
    def __call__(self, input_data):
        """
        A convenient way to chain operations.
        """
        return self.forward(input_data)


class Dense(Layer):
    def __init__(self, input_dim, output_dim, lr):
        self.lr = lr
        self.weights = Tensor(np.random.randn(input_dim, output_dim) * 0.01) # init weights
        self.biases = Tensor(np.zeros((1, output_dim))) # init biases

    def forward(self, input_data):
        self.input = input_data
        self.output = input_data @ self.weights + self.biases # o = xW + b
        return self.output
    
    def backward(self, output_gradient):
        output_gradient = Tensor(output_gradient)  # make sure it's a Tensor

        # calculate gradients
        input_gradient = output_gradient @ self.weights.T # dL/dx = dL/do * W^T
        weights_gradient = self.input.T @ output_gradient # dL/dW = x^T * dL/do
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True) # dL/db = sum(dL/do)

        # update parameters
        self.weights -= self.lr * weights_gradient # update weights using gradient descent
        self.biases -= self.lr * biases_gradient # update biases using gradient descent

        return input_gradient
    
    def __repr__(self):
        return f"Dense({self.weights.data.shape[0]}, {self.weights.data.shape[1]})"