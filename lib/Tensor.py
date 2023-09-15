from __future__ import annotations # this is needed to use Tensor in the type hint of Tensor

import numpy as np
import warnings
from functools import wraps
from typing import Union

class Tensor:
    """
    Tensor with Auto Differentiation, v2
    """
    def __init__(self, data: Union[int, float, list, np.ndarray], requires_grad: bool = False, parents=None, creation_op=None):
        self.data = self._process_data(data) # The data of the tensor

        self.shape = self.data.shape # The shape of the tensor
        self.ndim = self.data.ndim # The number of dimensions of the tensor
        self.size = self.data.size # The number of elements in the tensor

        self.requires_grad = requires_grad # Whether or not to compute gradients for this tensor
        self.grad = None # The gradient of this tensor (this should be an array, like the data attribute)

        if self.requires_grad: # If we need to compute gradients for this tensor
            self.zero_grad() # Initialize the gradient to 0

        self.parents = parents or [] # Tensors from which this one was created
        self.creation_op = creation_op # The operation that created this tensor

        self.epsilon = 1e-8 # A small number to prevent divide by zero errors

    def _process_data(self, data: Union[int, float, list, np.ndarray]) -> np.ndarray:
        allowed_types = (int, float, list, np.ndarray)
        if not isinstance(data, allowed_types):
            raise TypeError(f"Data must be one of {allowed_types}")
        if isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    @property
    def backward_ops(self):
        """
        I did this to clearly see what's implemented and what's not
        """
        ops = {
            "add": self.backward_add,
            "sub": self.backward_sub,
            "mul": self.backward_mul,
            "div": self.backward_div,
            "pow": self.backward_pow,
            "matmul": self.backward_matmul,
            "transpose": self.backward_transpose,
            "sum": self.backward_sum,
        }
        return ops

    def zero_grad(self):
        """
        Zero out the gradient
        """
        self.grad = np.zeros_like(self.data)


    def backward(self, grad: np.ndarray = None):
        # the grad should be an array (not a Tensor) just like the data attribute
        """
        This function will be called recursively to backpropogate gradients (auto differentiation)
        """
        if not self.requires_grad: # if gradient is not required, return
            # ? Most frameworks silently return here but I'm keeping the warning in for now
            warnings.warn("You called backward on a tensor that does not require gradients")
            return
        
        # if we dont have a gradient passed in, initialize it to 1
        if grad is None:  # if we call backward without passing a gradient, initialize the gradient to 1
            grad = np.ones_like(self.data)

        # ! this was a bunch of logic to handle shape mismatches, now im handling it individually in each op
        # # ? DEBUG -----------------------------------------------
        # print(f"Backpropagating through tensor with creation_op {self.creation_op}")
        # print(f"Current grad shape: {grad.shape}, self.data shape: {self.data.shape}")

        # # trying some broadcasting logic
        # if grad.shape != self.data.shape:
        #     # try to broadcast the gradient to the shape of the data
        #     try:
        #         grad = np.broadcast_to(grad, self.data.shape)
        #     except ValueError:
        #         raise ValueError(f"Cannot broadcast gradient of shape {grad.shape} to shape {self.data.shape}")
        # # ? ------------------------------------------------------

        # if grad.shape != self.data.shape:
        #     raise ValueError(f"The shape of the grad {grad.shape} does not match the shape of the data {self.data.shape}")

        # if we do have a gradient passed in, either initalize self.grad or accumulate it

        # print(f"Grad: {grad}")
        # print(f"Self.grad: {self.grad}")

        if self.grad is None:
            self.grad = grad
        else:
            # ! this doesnt work (in place operation) because of broadcasting
            # self.grad += grad  # accumulate gradient
            # ! have to do this because of the broadcasting
            self.grad = self.grad + grad  # accumulate gradient

        # if the tensor was created by the user, return
        if self.creation_op is None:
            return
        
        # print(f"Operation: {self.creation_op}")
    
        # time to backpropogate
        backward_op = self.backward_ops.get(self.creation_op, None) # get the correct backward op
        if backward_op:
            backward_op() # call the backward op
        else:
            raise NotImplementedError(f"Backward op for {self.creation_op} not implemented")

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

    def keep_stable(func):
        """
        Decorator to handle numerical stability/instability. Values like nan, inf, -inf, etc. can cause problems
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            tensors = [arg for arg in args if isinstance(arg, Tensor)]
            for tensor in tensors:
                if np.any(np.isinf(tensor.data) | np.isnan(tensor.data)):
                    # raise ValueError(f"Numerical instability detected in {func.__name__}: Inf or NaN values in tensor.data")
                    warnings.warn(f"Numerical instability detected in {func.__name__}: Inf or NaN values in tensor.data")

            result = func(*args, **kwargs)

            if isinstance(result, Tensor):
                if np.any(np.isinf(result.data) | np.isnan(result.data)):
                    # raise ValueError(f"Numerical instability detected in {func.__name__}: Inf or NaN values in tensor.data")
                    warnings.warn(f"Numerical instability detected in {func.__name__}: Inf or NaN values in tensor.data")
                
            return result
        return wrapper
            
    # Basic Operations ===========================================
    @make_tensor
    def __add__(self,  other: Union[int, float, Tensor]) -> Tensor:
        """
        a + b
        """
        result = np.add(self.data, other.data)
        return Tensor(result, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], creation_op="add")
    
    def backward_add(self):
        """
        (a + b)' = a' + b'
        """
        self.parents[0].backward(self.grad) 
        self.parents[1].backward(self.grad)

    @make_tensor
    def __sub__(self, other: Union[int, float, Tensor]) -> Tensor:
        """
        a - b
        """
        result = np.subtract(self.data, other.data)
        return Tensor(result, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], creation_op="sub")
    
    def backward_sub(self):
        """
        (a - b)' = a' - b'
        The first parent receives the gradient directly, the second parent receives the negation of the gradient.
        """
        self.parents[0].backward(self.grad)
        self.parents[1].backward(-self.grad)
    
    @make_tensor
    def __mul__(self, other: Union[int, float, Tensor]) -> Tensor:
        result = np.multiply(self.data, other.data)
        return Tensor(result, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], creation_op="mul")
    
    def backward_mul(self):
        """
        (a * b)' = a' * b + a * b'
        The gradient is scaled by the other parent for each respective parent.
        """
        self.parents[0].backward(self.grad * self.parents[1].data) # a' * b
        self.parents[1].backward(self.grad * self.parents[0].data) # a * b'
    
    @make_tensor
    def __truediv__(self, other: Union[int, float, Tensor]) -> Tensor:
        if other.data > self.epsilon: # if denominator is greater than epsilon (fuzz value)
            result = np.divide(self.data, other.data)
        else:
            result = np.divide(self.data, other.data + self.epsilon) # add epsilon to denominator to prevent divide by zero errors
            warnings.warn(f"Almost divide by zero in {self.__truediv__}, denominator was {other.data}, prevented by adding epsilon {self.epsilon}")
        return Tensor(result, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], creation_op="div")
    
    def backward_div(self):
        """
        (a / b)' = (a' * b - a * b') / b^2
        The first parent receives the scaled gradient, and the second parent receives the scaled and negated gradient.
        """
        self.parents[0].backward(self.grad / self.parents[1].data)  # a' / b
        self.parents[1].backward(-self.grad * self.parents[0].data / (self.parents[1].data ** 2))  # -a / b^2
        
    # ! I accidentally implemented f(a, b) = a^b instead of f(x) = x^n
    # @make_tensor
    # def __pow__(self, other: Union[int, float, Tensor]) -> Tensor:
    #     result = np.power(self.data, other.data)
    #     return Tensor(result, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], creation_op="pow")
    
    # def backward_pow(self):
    #     """
    #     f(a, b) = a^b
    #     df/da = b * a^(b - 1)
    #     df/db = a^b * ln(a)
    #     """
    #     a = self.parents[0].data
    #     b = self.parents[1].data

    #     # find partial derivatives
    #     grad_wrt_a = self.grad * b * (a ** (b - 1)) # Partial derivative with respect to 'a'

    #     # Partial derivative with respect to 'b', replacing NaNs and Infs with zero
    #     with np.errstate(divide='ignore', invalid='ignore'): # do this to ignore divide by zero errors
    #         grad_wrt_b = self.grad * (a ** b) * np.log(a)
    #     grad_wrt_b = np.where(np.isfinite(grad_wrt_b), grad_wrt_b, 0) # replace inf and nan with 0

    #     # backpropogate
    #     self.parents[0].backward(grad_wrt_a)
    #     self.parents[1].backward(grad_wrt_b)

    # ! Here's the correct implementation
    @keep_stable
    def __pow__(self, n: Union[int, float]) -> Tensor:
        """
        f(x) = x^n
        """
        if not isinstance(n, (int, float)):
            raise NotImplementedError("Only supporting int/float powers for now")
        
        result = np.power(self.data, n) # x^n (element-wise)
        return Tensor(result, requires_grad=self.requires_grad, parents=[self, n], creation_op="pow") # return a new tensor
    
    @keep_stable
    def backward_pow(self):
        """
        df/dx = n * x^(n - 1)
        """
        # n = self.parents[0].data
        x = self.parents[0].data
        n = self.parents[1]

        # find partial derivatives
        grad_wrt_x = self.grad * n * (x ** (n - 1))

        # backpropogate
        self.parents[0].backward(grad_wrt_x)

    @make_tensor
    @keep_stable # make sure this goes after make_tensor
    def __matmul__(self, other):
        """
        Matrix multiplication
        """
        result = np.matmul(self.data, other.data)
        return Tensor(result, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], creation_op="matmul")

    @keep_stable
    def backward_matmul(self):
        """
        (A @ B)' = A' @ B + A @ B'
        """
        # find partial derivatives
        grad_wrt_first_parent = np.matmul(self.grad, self.parents[1].data.T)
        grad_wrt_second_parent = np.matmul(self.parents[0].data.T, self.grad)

        # backpropogate
        self.parents[0].backward(grad_wrt_first_parent)
        self.parents[1].backward(grad_wrt_second_parent)


    # Reverse Operations ============================================
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return Tensor(other - self.data)  # Note the order

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return Tensor(other / self.data)  # Note the order
    
    def __rpow__(self, other):
        return Tensor(other ** self.data)
    
    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        # The other array should be on the left-hand side now
        # assuming self.data and other.data are NumPy arrays
        result = np.matmul(other.data, self.data)
        return Tensor(result)

    # Unary Operations ===========================================

    # no decorator because no args to convert to tensors
    def __neg__(self):
        return self * -1
    
    # no decorator because no args to convert to tensors
    def __abs__(self):
        return Tensor(np.abs(self.data), requires_grad=self.requires_grad, parents=[self], creation_op='abs')

    # Reduction Operations =======================================
    def sum(self, axis=None):
        result = np.sum(self.data, axis=axis)
        return Tensor(result, requires_grad=self.requires_grad, parents=[self], creation_op='sum')
    
    def backward_sum(self):
        """
        (sum(a))' = 1 for each element in a
        """
        self.parents[0].backward(self.grad * np.ones_like(self.data)) # 1 for each element in a

    # Shape Operations ===========================================
    def transpose(self):
        return Tensor(self.data.transpose(), requires_grad=self.requires_grad, parents=[self], creation_op='transpose')

    def backward_transpose(self):
        """
        (A^T)' = (A')
        """
        self.parents[0].backward(self.grad.transpose()) # A'
    
    @property
    def T(self):
        return self.transpose()

    # Comparison Operations ======================================

    # Indexing, Slicing, Joining, Mutating Ops ==================

    # Utils =====================================================

    # Other =====================================================

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            raise NotImplementedError("Cannot compare Tensor to non-Tensor")
        return np.array_equal(self.data, other.data)
    
    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        if not isinstance(other, Tensor):
            raise NotImplementedError("Cannot compare Tensor to non-Tensor")
        return np.allclose(self.data, other.data, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"


def trace_requires_grad(tensor):
    """
    Util function to trace the requires_grad attribute of a tensor and its parents, for debugging
    """
    print(f"Tensor: {tensor}, requires_grad={tensor.requires_grad}")
    if hasattr(tensor, 'parents'):
        for parent in tensor.parents:
            trace_requires_grad(parent)