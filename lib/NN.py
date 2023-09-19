from __future__ import annotations # for type hinting

import numpy as np
import warnings
from typing import List

from lib.TensorV2 import Tensor, force_tensor_method

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


class ReLU(Module): 
    @force_tensor_method
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        return x.max(Tensor(np.zeros_like(x.data)))

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

    @force_tensor_method
    def forward(self, x: Tensor) -> Tensor:
        # force input to be a Tensor
        # if not isinstance(x, Tensor):
        #     warnings.warn(f"Input data to layer {self.__class__.__name__} is not a Tensor. Converting to Tensor.")
        #     x = Tensor(x, requires_grad=True)

        input_features = x.shape[1] # input_dim
        if input_features != self.weights.shape[0]: # input_dim != weights.shape[0] - make sure the tensor matches the way the weights were initialized
            raise RuntimeError(f"Input tensor with {input_features} features should match layer input dim {self.weights.shape[0]}")

        #? not sure if i need to handle the case where batch_size = 1, and x is a vector
        return x @ self.weights + self.biases # matrix multiplication