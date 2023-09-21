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
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = Tensor.randn(output_dim, input_dim) * 0.01 # init backwards (output_dim, input_dim) for computational efficiency
        self.biases = Tensor.zeros(output_dim)

    @force_tensor_method
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weights.T + self.biases # transpose weights for computational efficiency
    
    def parameters(self) -> List[Tensor]:
        return [self.weights, self.biases]


class BatchNorm(Layer):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = Tensor.ones(num_features)
        self.beta = Tensor.zeros(num_features)
        self.eps = 1e-5

    def forward(self, inputs: Tensor) -> Tensor:
        mean = np.mean(inputs.data, axis=0)
        var = np.var(inputs.data, axis=0)
        normalized = (inputs.data - mean) / np.sqrt(var + self.eps)
        out = self.gamma.data * normalized + self.beta.data
        return Tensor(out)

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Dropout(Layer):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, inputs: Tensor) -> Tensor:
        mask = np.random.binomial(1, 1 - self.p, size=inputs.data.shape)
        out = inputs.data * mask / (1 - self.p)
        return Tensor(out)

    def parameters(self) -> List[Tensor]:
        return []  # Dropout has no learnable parameters