from __future__ import annotations # this is needed to use Tensor in the type hint of Tensor

import numpy as np
import warnings
from functools import wraps
from typing import Union
from contextlib import contextmanager

class Tensor:
    """
    Tensor with Auto Differentiation, v2
    """
    compute_gradients = True # class attribute to control whether to compute gradients or not

    @staticmethod
    @contextmanager
    def no_grad():
        original_state = Tensor.compute_gradients
        Tensor.compute_gradients = False
        try:
            yield
        finally:
            Tensor.compute_gradients = original_state

    def __init__(self, data: Union[int, float, list, np.ndarray], children=(), op='', requires_grad: bool = False, axis=None):
        self.data = self._process_data(data)
        self.shape = self.data.shape
        self.size = self.data.size
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype
        self.grad = None
        self.requires_grad = requires_grad and Tensor.compute_gradients
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

    
    def _qscalar(self, x):
        """Quasi-scalar: a scalar or a 1-element array"""
        return x.size == 1 or np.isscalar(x) or x.ndim == 0 or x.shape == ()
    
    def no_qscalars(func):
        """
        Decorator to ensure that the function is not called on a quasi-scalar
        """
        @wraps(func)
        def wrapper(self, other):
            if self._qscalar(other):
                raise TypeError(f"Cannot call {func.__name__} on a quasi-scalar")
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
            # print(f"Creation_op {v._op}, shape {v.data.shape}") # ? Debug
            if hasattr(v, '_backward'): # handle no_grad context manager
                v._backward()

    # ! Main Ops ==============================================================
    @make_tensor
    def __add__(self, other: Tensor) -> Tensor:
        """
        Updated add method to handle reshape error
        """
        rg = self.requires_grad or other.requires_grad
        out = np.add(self.data, other.data)
        out = Tensor(out, (self, other), 'add', requires_grad=rg)

        def _backward():
            is_self_qscalar = self._qscalar(self.data)
            is_other_qscalar = self._qscalar(other.data)

            # Case 1: both are q-scalars
            if is_self_qscalar and is_other_qscalar:
                self.grad += np.sum(out.grad)
                other.grad += np.sum(out.grad)
            # Case 2: self is q-scalar, other is not
            elif is_self_qscalar:
                self.grad += np.sum(out.grad)
                other.grad += out.grad
            # Case 3: other is q-scalar, self is not
            elif is_other_qscalar:
                self.grad += out.grad
                other.grad += np.sum(out.grad)
            # Case 4: both are tensors but different shapes (hardest case)
            # ? Can't use reshape like i did first becaus it cant handle different sized tensors
            elif self.data.shape != other.data.shape:
                # Identify the axis that should be summed over
                sum_axes_self = tuple(np.nonzero(np.array(self.data.shape) < np.array(out.data.shape))[0])
                sum_axes_other = tuple(np.nonzero(np.array(other.data.shape) < np.array(out.data.shape))[0])
                # Update the gradients for self and other
                if sum_axes_self:
                    self.grad += np.sum(out.grad, axis=sum_axes_self).reshape(self.data.shape)
                else:
                    self.grad += out.grad
                
                if sum_axes_other:
                    other.grad += np.sum(out.grad, axis=sum_axes_other).reshape(other.data.shape)
                else:
                    other.grad += out.grad

            # Case 5: both are tensors and same shape
            else:
                self.grad += out.grad
                other.grad += out.grad

        if Tensor.compute_gradients:
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
            is_self_qscalar = self._qscalar(self.data)
            is_other_qscalar = self._qscalar(other.data)

            # Case 1: both are q-scalars
            if is_self_qscalar and is_other_qscalar:
                self.grad += np.sum(other.data * out.grad)
                other.grad += np.sum(self.data * out.grad)
            # Case 2: self is q-scalar, other is not
            elif is_self_qscalar:
                self.grad += np.sum(other.data * out.grad)
                other.grad += self.data * out.grad
            # Case 3: other is q-scalar, self is not
            elif is_other_qscalar:
                self.grad += other.data * out.grad
                other.grad += np.sum(self.data * out.grad)
            # Case 4: both are tensors but different shapes
            elif self.data.shape != other.data.shape:
                # Identify axes along which summing needs to occur for the gradients
                sum_axes_self = tuple(np.nonzero(np.array(self.data.shape) < np.array(out.data.shape))[0])
                sum_axes_other = tuple(np.nonzero(np.array(other.data.shape) < np.array(out.data.shape))[0])
                
                grad_wrt_self = other.data * out.grad
                grad_wrt_other = self.data * out.grad
                
                if sum_axes_self:
                    self.grad += np.sum(grad_wrt_self, axis=sum_axes_self).reshape(self.data.shape)
                else:
                    self.grad += grad_wrt_self
                
                if sum_axes_other:
                    other.grad += np.sum(grad_wrt_other, axis=sum_axes_other).reshape(other.data.shape)
                else:
                    other.grad += grad_wrt_other
            # Case 5: both are tensors and same shape
            else:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

        if Tensor.compute_gradients:
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
            is_self_qscalar = self._qscalar(self.data)

            # Case 1: self is q-scalar
            if is_self_qscalar:
                self.grad += np.sum(n * self.data ** (n - 1) * out.grad)
            # Case 2: self is tensor
            else:
                self.grad += (n * self.data ** (n - 1) * out.grad).reshape(self.data.shape)

        if Tensor.compute_gradients:
            out._backward = _backward

        return out

    @no_qscalars
    @make_tensor
    def __matmul__(self, other: Tensor) -> Tensor:
        """
        Matrix Multiplication
        """
        # ? make sure to prevent scalars and non-tensors from being passed in (decorators)
        out = np.matmul(self.data, other.data)
        out = Tensor(out, (self, other), 'matmul', requires_grad=self.requires_grad)

        def _backward():
            """
            (A @ B)' = A' @ B + A @ B'
            """
            self.grad += np.matmul(out.grad, other.data.T).reshape(self.data.shape)
            other.grad += np.matmul(self.data.T, out.grad).reshape(other.data.shape)

        if Tensor.compute_gradients:
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

    def __floordiv__(self): raise NotImplementedError(f"Operation not implemented")
    def __mod__(self): raise NotImplementedError(f"Operation not implemented")

    # handle in place (which will mess up the comutation graph) - so no backward methods needed
    # THESE ARE UNTESTED.... use at your own risk
    @make_tensor
    def __iadd__(self, other: Tensor) -> Tensor:
        warnings.warn(f"The operation {self.__iadd__.__name__} is untested. Use at your own risk.")
        rg = self.requires_grad or other.requires_grad
        if rg:
            raise RuntimeError("In-place operations not supported for tensors with requires_grad=True")
        self.data += other.data
        return self
    
    @make_tensor
    def __isub__(self, other: Tensor) -> Tensor:
        warnings.warn(f"The operation {self.__iadd__.__name__} is untested. Use at your own risk.")
        rg = self.requires_grad or other.requires_grad
        if rg:
            raise RuntimeError("In-place operations not supported for tensors with requires_grad=True")
        self.data -= other.data
        return self
    
    @make_tensor
    def __imul__(self, other: Tensor) -> Tensor:
        warnings.warn(f"The operation {self.__iadd__.__name__} is untested. Use at your own risk.")
        rg = self.requires_grad or other.requires_grad
        if rg:
            raise RuntimeError("In-place operations not supported for tensors with requires_grad=True")
        self.data *= other.data
        return self
    
    @make_tensor
    def __itruediv__(self, other: Tensor) -> Tensor:
        warnings.warn(f"The operation {self.__iadd__.__name__} is untested. Use at your own risk.")
        rg = self.requires_grad or other.requires_grad
        if rg:
            raise RuntimeError("In-place operations not supported for tensors with requires_grad=True")
        self.data /= other.data
        return self

    # these will only be able to handle no grad tensors, like above
    def __ifloordiv__(self, other): raise NotImplementedError(f"Operation not implemented")
    def __imod__(self, other): raise NotImplementedError(f"Operation not implemented")
    def __ipow__(self, other): raise NotImplementedError(f"Operation not implemented")

    def log(self):
        # Check for domain errors
        if np.any(self.data <= 0):
            raise ValueError("Log is only defined for x > 0")

        out = np.log(self.data)
        out = Tensor(out, (self,), 'log', requires_grad=self.requires_grad)

        def _backward():
            """
            d/dx (log(x)) = 1 / x
            """
            is_self_qscalar = self._qscalar(self.data)

            # Case 1: True or Quasi-scalar
            if is_self_qscalar:
                self.grad += np.sum(out.grad / self.data)
            # Case 2: General tensor
            else:
                self.grad += (out.grad / self.data).reshape(self.data.shape)

        if Tensor.compute_gradients:
            out._backward = _backward

        return out

    def exp(self):
        out = np.exp(self.data)

        if np.any(np.isinf(out)):
            raise OverflowError("Exponential resulted in overflow")

        out = Tensor(out, (self,), 'exp', requires_grad=self.requires_grad)

        def _backward():
            """
            d/dx (exp(x)) = exp(x)
            """
            is_self_qscalar = self._qscalar(self.data)
            # Case 1: True or Quasi-scalar
            if is_self_qscalar:
                self.grad += np.sum(out.grad * out.data)
            # Case 2: General tensor
            else:
                self.grad += (out.grad * out.data).reshape(self.data.shape)

        if Tensor.compute_gradients:
            out._backward = _backward

        return out

    # reduction ops
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
            is_self_qscalar = self._qscalar(self.data)

            # Case 1: True or Quasi-scalar
            if is_self_qscalar:
                self.grad += np.sum(out.grad)

            # Case 2 & 3: General tensor (with or without axis)
            else:
                expanded_grad = np.ones_like(self.data) * out.grad
                if axis is not None:
                    # reshape or expand 
                    expanded_grad = np.expand_dims(out.grad, axis=axis)
                self.grad += expanded_grad
        if Tensor.compute_gradients:
            out._backward = _backward
        return out

    def max(self, axis=None):
        """
        Max of the tensor along the given axis
        """
        out_data = np.max(self.data, axis=axis)
        out = Tensor(out_data, (self,), 'max', requires_grad=self.requires_grad)

        def _backward():
            """
            d/dx (max(x)) = 1 if x is the max, 0 otherwise
            """
            is_self_qscalar = self._qscalar(self.data)

            # Case 1: True or Quasi-scalar
            if is_self_qscalar:
                self.grad += np.sum(out.grad)

            # Case 2 & 3: General tensor (with or without axis)
            else:
                if axis is None:
                    self.grad += np.where(self.data == out_data, out.grad, 0)
                else:
                    out_expanded = np.expand_dims(out_data, axis=axis)
                    self.grad += np.where(self.data == out_expanded, np.expand_dims(out.grad, axis=axis), 0)
        if Tensor.compute_gradients:
            out._backward = _backward
        return out

    def mean(self, axis=None):
        """
        Mean of the tensor along the given axis
        """
        out_data = np.mean(self.data, axis=axis)
        out = Tensor(out_data, (self,), 'mean', requires_grad=self.requires_grad)

        def _backward():
            """
            d/dx (mean(x)) = 1 / n
            """
            is_self_qscalar = self._qscalar(self.data)

            # Case 1: True or Quasi-scalar
            if is_self_qscalar:
                self.grad += np.sum(out.grad)

            # Case 2 & 3: General tensor (with or without axis)
            else:
                n = self.data.size if axis is None else self.data.shape[axis]
                grad_multiplier = out.grad / n
                if axis is not None:
                    grad_multiplier = np.expand_dims(grad_multiplier, axis=axis)
                self.grad += np.ones_like(self.data) * grad_multiplier
        if Tensor.compute_gradients:
            out._backward = _backward
        return out

    def clip(self, min, max) -> Tensor:
        out = np.clip(self.data, min, max)
        out = Tensor(out, (self,), 'clip', requires_grad=self.requires_grad)

        def _backward():
            """
            d/dx (clip(x)) = 1 if min <= x <= max, 0 otherwise
            """
            grad_clip = np.where((self.data >= min) & (self.data <= max), 1, 0)
            self.grad += (out.grad * grad_clip).reshape(self.data.shape)

        if Tensor.compute_gradients:
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
        if Tensor.compute_gradients:
            out._backward = _backward

        return out

    @staticmethod
    def zeros(shape, **kwargs):
        return Tensor(np.zeros(shape), **kwargs)
    
    @staticmethod
    def ones(shape, **kwargs):
        return Tensor(np.ones(shape), **kwargs)
    
    @staticmethod
    def randn(*shape, **kwargs):
        return Tensor(np.random.randn(*shape), **kwargs)
    
    @staticmethod
    def rand(*shape, **kwargs):
        return Tensor(np.random.rand(*shape), **kwargs)

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
def force_tensor_func(func):
    """
    Test if the first argument is a Tensor, and if not, throw an error
    Used for functions that take a Tensor as the first argument
    """
    @wraps(func)
    def wrapper(x: Tensor, *args, **kwargs):
        if not isinstance(x, Tensor):
            # warnings.warn(f"Input data to layer {func.__name__} is not a Tensor. Converting to Tensor.")
            raise TypeError(f"Input data to layer {func.__name__} need to be a Tensor, is {x.__class__.__name__}")
        return func(x, *args, **kwargs)
    return wrapper


def force_tensor_method(method):
    """
    Test if the second argument is a Tensor, and if not, throw an error.
    Used for class methods that take self as the first argument and a Tensor as the second argument
    """
    @wraps(method)
    def wrapper(self, x: Tensor, *args, **kwargs):
        if not isinstance(x, Tensor):
            raise TypeError(f"Input data to layer {method.__name__} need to be a Tensor, is {x.__class__.__name__}")
        return method(self, x, *args, **kwargs)
    return wrapper