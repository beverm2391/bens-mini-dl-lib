from __future__ import annotations # this is needed to use Tensor in the type hint of Tensor

import numpy as np
import warnings
from functools import wraps
from typing import Union

class Tensor:
    """
    Tensor with Auto Differentiation, v2
    """
    def __init__(self, data: Union[int, float, list, np.ndarray], children=(), op='', requires_grad: bool = False, axis=None):
        self.data = self._process_data(data)
        self.grad = None
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.zero_grad()
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op
        self.axis = axis

    # ! Some Utility Functions =================================================
    def _check_type(self, data):
        """
        Check data type against allowed types, numpy dtypes, etc.
        """
        allowed_types = (int, float, list, np.ndarray)
        if not isinstance(data, allowed_types):
            if not np.issubdtype(data.dtype, np.number):
                print(f"Data must be one of {[t.__name__ for t in allowed_types]}, is {type(data)}")
                print(f"data {data}")
                raise TypeError(f"Invalid Data Type")
        return

    def _process_data(self, data: Union[int, float, list, np.ndarray]) -> np.ndarray:
        self._check_type(data) # check the type of the data
        if isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float64) # force float64 to avoid broadcasting issues down the line

    def make_tensor(func) -> Tensor:
        """
        Decorator to convert the 'other' arg to a tensor if its not already a tensor
        """
        @wraps(func) # make sure the wrapper function has the same name, docstring, etc. as the original function
        def wrapper(self, other):
            if not isinstance(other, Tensor):
                other = Tensor(other, requires_grad=self.requires_grad)
            return func(self, other)
        return wrapper
    
    def no_scalars(func):
        """
        Decorator to ensure that the function is not called on a scalar
        """
        @wraps(func)
        def wrapper(self, other):
            if np.isscalar(other):
                raise TypeError(f"Cannot call {func.__name__} on a scalar")
            return func(self, other)
        return wrapper

    # ! Backprop ==============================================================
    def backward(self):
        """
        Backpropagation
        """
        if self.data.ndim > 0:
            warnings.warn("Backpropagation only supported for scalar values, not arrays")
            return

        # topological order all of the children in the graph
        topo = [] # empty list
        visited = set() # empty set
        def build_topo(v): 
            if v not in visited: # if we haven't visited the node yet
                if v.requires_grad: # if the node requires grad
                    visited.add(v) # mark as visited
                for child in v._prev: # recursively build topological ordering
                    build_topo(child) # recursive call
                topo.append(v) # add to topological sort
        build_topo(self) # start from self

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data) # gradient of final node is 1
        for v in reversed(topo): # iterate in reverse topological order
            v._backward() # call the _backward method

    # ! Main Ops ==============================================================
    @make_tensor
    def __add__(self, other: Tensor) -> Tensor:
        """
        Add two tensors
        """
        rg = self.requires_grad or other.requires_grad
        out = np.add(self.data, other.data)
        out = Tensor(out, (self, other), 'add', requires_grad=rg)

        def _backward():
            self.grad += out.grad.reshape(self.data.shape) # must reshape to broadcast correctly
            other.grad += out.grad.reshape(other.data.shape) # must reshape to broadcast correctly
        out._backward = _backward

        return out

    @make_tensor
    def __mul__(self, other: Tensor) -> Tensor: 
        """
        Multiply two tensors
        """
        rg = self.requires_grad or other.requires_grad
        out = np.multiply(self.data, other.data)
        out = Tensor(out, (self, other), 'mul', requires_grad=rg)

        def _backward():
            """
            d/dx (x * y) = y
            d/dy (x * y) = x
            """
            def _qscalar(x):
                """Quasi-scalar: a scalar or a 1-element array"""
                return x.size == 1 or np.isscalar(x) or x.ndim == 0 or x.shape == ()
            
            # ? debug broadcasting issues
            # print(f"Self data shape: {self.data.shape}")
            # print(f"Other data shape: {other.data.shape}")
            # print(f"Self grad shape: {self.grad.shape}")
            # print(f"Out grad shape: {out.grad.shape}")

            if _qscalar(self.data):
                self.grad += np.sum(other.data * out.grad) # must sum to multiply correctly (if self.data is a scalar)
            else:
                self.grad += (other.data * out.grad).reshape(self.data.shape) # must reshape to broadcast correctly

            if _qscalar(other.data):
                other.grad += np.sum(self.data * out.grad) # must sum to  multiply correctly (if other.data is a scalar)
            else:
                other.grad += (self.data * out.grad).reshape(other.data.shape) # must reshape to broadcast correctly

        out._backward = _backward

        return out

    def __pow__(self, n: Union[int, float]):
        if not isinstance(n, (int, float)):
            raise NotImplementedError("Only supporting int/float powers for now")

        out = np.power(self.data, n)
        out = Tensor(out, (self,), f'pow', requires_grad=self.requires_grad)

        def _backward():
            """
            d/dx (x^n) = n * x^(n-1)
            """
            self.grad += ((n * self.data**(n - 1)) * out.grad).reshape(self.data.shape) # must reshape to broadcast correctly
        out._backward = _backward

        return out

    @no_scalars
    @make_tensor
    def __matmul__(self, other: Tensor) -> Tensor:
        """
        Matrix Multiplication
        """
        out = np.matmul(self.data, other.data)
        out = Tensor(out, (self, other), 'matmul', requires_grad=self.requires_grad)

        def _backward():
            """
            (A @ B)' = A' @ B + A @ B'
            """
            self.grad += np.matmul(out.grad, other.data.T).reshape(self.data.shape)
            other.grad += np.matmul(self.data.T, out.grad).reshape(other.data.shape)

        out._backward = _backward
        return out


    # ! Other Ops =========================================================
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    @make_tensor # ensure other is tensor (before we transpose it)
    def __rmatmul__(self, other): return self @ other.T

    def sum(self, axis=None):
        """
        Sum the tensor along the given axis
        """
        out = np.sum(self.data, axis=axis)
        out = Tensor(out, (self,), 'sum', requires_grad=self.requires_grad)
        
        def _backward():
            """
            d/dx (sum(x)) = 1
            """
            self.grad += (np.ones_like(self.data) * out.grad).reshape(self.data.shape)
        out._backward = _backward

        return out
    
    def transpose(self):
        """
        Transpose the tensor
        """
        out = np.transpose(self.data)
        out = Tensor(out, (self,), 'transpose', requires_grad=self.requires_grad)

        def _backward():
            """
            d/dx (transpose(x)) = 1
            """
            self.grad += np.transpose(out.grad) # transpose handles the shape correctly
        out._backward = _backward

        return out

    @property
    def T(self): return self.transpose()
    def __hash__(self): return id(self) # so we can add Tensors to set in backward()

    def __eq__(self, other: Tensor) -> bool:
        if not isinstance(other, Tensor):
            raise NotImplementedError("Cannot compare Tensor to non-Tensor")
        return np.array_equal(self.data, other.data)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

# ! Utility Functions =========================================================
def force_tensor(func):
    @wraps(func)
    def wrapper(x, *args, **kwargs):
        if not isinstance(x, Tensor):
            # warnings.warn(f"Input data to layer {func.__name__} is not a Tensor. Converting to Tensor.")
            raise TypeError(f"Input data to layer {func.__name__} need to be a tensor.")
        return func(x, *args, **kwargs)
    return wrapper