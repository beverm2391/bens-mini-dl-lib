from __future__ import annotations # for type hinting

import numpy as np
import warnings
from typing import List
from functools import wraps

from lib.Tensor import Tensor

def force_tensor(func):
    @wraps(func)
    def wrapper(x, *args, **kwargs):
        if not isinstance(x, Tensor):
            # warnings.warn(f"Input data to layer {func.__name__} is not a Tensor. Converting to Tensor.")
            raise TypeError(f"Input data to layer {func.__name__} need to be a tensor.")
        return func(x, *args, **kwargs)
    return wrapper


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

    def parameters(self):
        return []


class Layer(Module):
    def __init__(self):
        super().__init__() # init Module

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the output of this layer using `x`.
        """

        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        """
        A convenient way to chain operations.
        """
        return self.forward(x)


class ReLU(Module):
    @force_tensor
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        return x.maximum(Tensor(np.zeros_like(x.data)))

    def backward(self, output_grad):
        return (self.input > 0) * output_grad


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
    
    @force_tensor
    def forward(self, x: Tensor) -> Tensor:
        # force input to be a Tensor
        if not isinstance(x, Tensor):
            warnings.warn(f"Input data to layer {self.__class__.__name__} is not a Tensor. Converting to Tensor.")
            x = Tensor(x, requires_grad=True)

        input_features = x.shape[1] # input_dim
        if input_features != self.weights.shape[0]: # input_dim != weights.shape[0] - make sure the tensor matches the way the weights were initialized
            raise RuntimeError(f"Input tensor with {input_features} features should match layer input dim {self.weights.shape[0]}")

        #? not sure if i need to handle the case where batch_size = 1, and x is a vector
        return x @ self.weights + self.biases # matrix multiplication


class MLP(Module):
    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int, output_dim: int):
        super().__init__() # init Module

        self.add_module("dense1", Dense(input_dim, hidden_dim_1))
        self.add_module("relu1", ReLU())
        self.add_module("dense2", Dense(hidden_dim_1, hidden_dim_2))
        self.add_module("relu2", ReLU())
        self.add_module("dense3", Dense(hidden_dim_2, output_dim))

    @force_tensor
    def forward(self, x: Tensor) -> Tensor:
        # first dense layer
        x = self.get_module("dense1")(x)
        x = self.get_module("relu1")(x)

        # second dense layer
        x = self.get_module("dense2")(x)
        x = self.get_module("relu2")(x)

        # third dense layer
        x = self.get_module("dense3")(x)

        return x

    def parameters(self) -> List[Tensor]:
        return [p for module in self._modules.values() for p in module.parameters()]