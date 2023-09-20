from __future__ import annotations # for type hinting

import numpy as np
import warnings
from typing import List

from lib.TensorV2 import Tensor, force_tensor_method

# ! BASE CLASSES =========================================================
class Module:
    """
    Base class for all neural network modules.
    """
    def __init__(self):
        self._modules = {} # dictionary of sub-modules

    def add_module(self, name: str, module: Module):
        """Add a sub-module to the current module."""
        self._modules[name] = module
    
    def get_module(self, name: str):
        """Retrieve a sub-module by name."""
        return self._modules.get(name, None)

    def zero_grad(self):
        for module in self._modules.values():
            for p in module.parameters():
                p.zero_grad()

    def parameters(self): return []
    def forward(self, *args, **kwargs): raise NotImplementedError
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)


# ! Activation Functions ==================================================
class ReLU(Module):
    @force_tensor_method
    def forward(self, x: Tensor) -> Tensor:
        # Create a new Tensor that holds the result of the ReLU operation.
        out_data = np.maximum(0, x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            x.grad += (out_data > 0) * out.grad  # gradient is passed through where input > 0
        out._backward = _backward

        return out

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        out_data = 1 / (1 + np.exp(-x.data))
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            x.grad += out_data * (1 - out_data) * out.grad
        out._backward = _backward

        return out

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        out_data = np.tanh(x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            x.grad += (1 - out_data ** 2) * out.grad
        out._backward = _backward

        return out

class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        out_data = np.where(x.data > 0, x.data, self.alpha * x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            x.grad += np.where(x.data > 0, 1, self.alpha) * out.grad
        out._backward = _backward

        return out

# ! Loss Functions =======================================================

class Loss(Module):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor: raise NotImplementedError

class MSELoss(Loss):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.input = (x, y)
        diff = x - y
        mse = (diff * diff).mean()
        return mse

class CrossEntropyLoss(Loss):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.input = (x, y)
        epsilon = 1e-12
        x_clipped = x.clip(epsilon, 1. - epsilon) # clip to avoid log(0)
        ce = - (y * x_clipped.log() + (1. - y) * (1. - x_clipped).log()).mean() # cross entropy
        return ce

# ! Layers ===============================================================

class Layer(Module):
    def __init__(self):
        super().__init__() # init Module

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the layer
        """
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__() # init Layer
        self.weights = self._init_weights(input_dim, output_dim)
        self.biases = self._init_biases(output_dim)

    def _init_weights(self, input_dim: int, output_dim: int) -> Tensor:
        assert input_dim > 0 and output_dim > 0
        arr = np.random.randn(input_dim, output_dim) * 0.01
        return Tensor(arr, requires_grad=True) # defaulting this to True for sanity
    
    def _init_biases(self, output_dim: int) -> Tensor:
        assert output_dim > 0
        arr = np.zeros((1, output_dim))
        return Tensor(arr, requires_grad=True) # defaulting this to True for sanity
    
    def parameters(self) -> List[Tensor]:
        return [self.weights, self.biases]

    @force_tensor_method
    def forward(self, x: Tensor) -> Tensor:
        input_features = x.shape[1] # input_dim
        if input_features != self.weights.shape[0]: # input_dim != weights.shape[0] - make sure the tensor matches the way the weights were initialized
            raise RuntimeError(f"Input tensor with {input_features} features should match layer input dim {self.weights.shape[0]}")

        #? not sure if i need to handle the case where batch_size = 1, and x is a vector
        # xW or Wx? any transposition?
        # https://stackoverflow.com/questions/63006388/should-i-transpose-features-or-weights-in-neural-network
        # "Should I transpose features or weights in Neural network?" - in torch convention, you should transpose weights, but use matmul with the features first.
        return x @ self.weights.T + self.biases # matrix multiplication