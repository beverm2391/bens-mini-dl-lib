from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from lib.Tensor_old import Tensor

class Activation(ABC):

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backward(self, output_gradient):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class ReLU(Activation):
    """
    ReLU Function:
    $$ ReLU(x) = max(0, x) $$

    Derivative:
    $$ ReLU'(x) = 1 if x > 0 else 0 $$
    """ 

    def forward(self, input_data: Tensor):
        self.input = input_data # save input data for backward pass, use data attribute because it is a Tensor
        self.output = Tensor(np.maximum(0, input_data.data)) # contains $ReLU(x)$
        return self.output

    def backward(self, output_gradient: Tensor):
        input_gradient = Tensor(np.where(self.input > 0, output_gradient.data, 0)) # contains $ReLU'(x)$
        return input_gradient


class Sigmoid(Activation):

    def forward(self, input_data):
        self.input = input_data # contains $x$
        self.output = 1 / (1 + np.exp(-input_data)) # contains $\sigma(x)$
        return self.output
    
    def backward(self, output_gradient):
        input_gradient = output_gradient * self.output * (1 - self.output) # contains $\sigma'(x)$
        return input_gradient


class Tanh(Activation):

    def forward(self, input_data):
        self.input = input_data
        self.output = np.tanh(input_data) # $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
        return self.output
    
    def backward(self, output_gradient):
        input_gradient = output_gradient * (1 - self.output ** 2) # $$\tanh'(x) = 1 - \tanh^2(x)$$
        return input_gradient


class SoftMax(Activation):

    def forward(self, input_data):
        self.input = input_data
        exp_data = np.exp(input_data - np.max(input_data, axis=1, keepdims=True)) # subtract max to avoid overflow
        self.output = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        input_gradient = np.zeros_like(self.output) # create zero matrix of same shape as output

        # loop over the samples (assuming a batched input)
        for i in range(self.input.shape[0]):
            # get jacobian matrix of softmax for each sample
            jacobian_matrix = np.diagflat(self.output[i]) - np.outer(self.output[i], self.output[i])
            # multiply jacobian with output gradient
            input_gradient[i] = output_gradient[i].dot(jacobian_matrix)

        return input_gradient