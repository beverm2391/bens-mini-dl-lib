import numpy as np
import warnings

class Tensor:
    """
    New Tensor class with auto differentiation
    """
    def __init__(self, data, requires_grad=True, parents=None, creation_op=None):
        self.data = np.array(data) # data
        self.shape = self.data.shape # shape of data
        self.requires_grad = requires_grad # whether to calculate gradients
        self.parents = parents or []
        self.creation_op = creation_op
        self.grad = self.zero_grad() if requires_grad else None # gradient of data, if needed
        self.is_scalar = self.data.ndim == 0 # whether the data is a scalar

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    def backward(self, grad=None):
        if not self.requires_grad: # if this tensor doesn't require gradients, return
            return
        if self.creation_op == None: # if this is a leaf node, return
            return

        if grad is None and self.grad is None:
            # if self is a leaf node, we can start from 1
            grad = Tensor(np.ones_like(self.data))
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad # if self is a leaf node, we accumulate gradients
        # time to backpropogate
        if self.creation_op == 'add':
            self.parents[0].backward(self.grad)
            self.parents[1].backward(self.grad)
        elif self.creation_op == 'sub':
            self.parents[0].backward(self.grad)
            self.parents[1].backward(-self.grad)
        elif self.creation_op == 'mul':
            self.parents[0].backward(self.grad * self.parents[1])
            self.parents[1].backward(self.grad * self.parents[0])
        elif self.creation_op == 'div':
            self.parents[0].backward(self.grad / self.parents[1])
            self.parents[1].backward(-self.grad * self.parents[0] / self.parents[1] ** 2)
        elif self.creation_op == 'pow':
            self.parents[0].backward(self.grad * self.parents[1].data * (self.parents[0].data ** (self.parents[1].data - 1)))
            self.parents[1].backward(self.grad * (self.parents[0].data ** self.parents[1].data) * np.log(self.parents[0].data))
        # TODO handle scalar multiplication problems
        elif self.creation_op == 'matmul':
            raise NotImplementedError("Matrix multiplication backprop not implemented")    
        # update to handle edge cases
        elif self.creation_op == 'neg':
            self.parents[0].backward(-self.grad)
        elif self.creation_op == 'transpose':
            self.parents[0].backward(self.grad.T)
        elif self.creation_op == 'abs':
            self.parents[0].backward(self.grad * np.sign(self.parents[0].data))
        elif self.creation_op == 'sum':
            self.parents[0].backward(self.grad * np.ones_like(self.parents[0].data))
        elif self.creation_op == 'mean':
            self.parents[0].backward(self.grad * np.ones_like(self.parents[0].data) / np.size(self.parents[0].data))
        elif self.creation_op == 'max':
            self.parents[0].backward(self.grad * (self.parents[0].data == self.data))
        elif self.creation_op == 'min':
            self.parents[0].backward(self.grad * (self.parents[0].data == self.data))
        elif self.creation_op == 'std':
            mean_val = np.mean(self.parents[0].data)
            N = self.parents[0].data.size
            self.parents[0].backward(self.grad * (self.parents[0].data - mean_val) / (self.data * np.sqrt(N)))
        elif self.creation_op == 'reshape':
            self.parents[0].backward(self.grad.reshape(self.parents[0].shape)) # reshape to the shape of the parent
        elif self.creation_op == 'squeeze':
            axis = self.meta_info['axis']
            if axis is not None:
                grad_expanded = np.expand_dims(grad, axis=axis)
            else:
                grad_expanded = grad
            self.parents[0].backward(grad_expanded)
        elif self.creation_op == 'unsqueeze':
            axis = self.meta_info['axis']
            self.parents[0].backward(np.squeeze(grad, axis=axis))
        else:
            raise NotImplementedError(f"Gradient for {self.creation_op} not implemented")
        # TODO: add broadcasting, indexing

    # Basic Operations ============================================
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data + other.data, requires_grad=True, parents=[self, other], creation_op='add')
        
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data - other.data, requires_grad=True, parents=[self, other], creation_op='sub')
        
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data * other.data, requires_grad=True, parents=[self, other], creation_op='mul')
        
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data / other.data, requires_grad=True, parents=[self, other], creation_op='div')
        
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data ** other.data, requires_grad=True, parents=[self, other], creation_op='pow')
        
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if self.is_scalar:
            return Tensor(self.data * other.data)
        
        if self.data.ndim == 0 and other.data.ndim == 0: # if both are scalars
            warnings.warn("Both of your inputs are scalars. Using element-wise multiplication instead. Use the * operator insead of @.")
            return self.data * other.data
        
        if self.data.ndim == 0 and other.data.ndim > 0: # if self is a scalar and other is not
            warnings.warn("One of your inputs is a scalar. Using element-wise multiplication instead. Use the * operator insead of @.")
            return Tensor(self.data * other.data, requires_grad=True, parents=[self, other], creation_op='mul')
        
        if self.data.ndim > 0 and other.data.ndim == 0: # if self is not a scalar and other is
            warnings.warn("One of your inputs is a scalar. Using element-wise multiplication instead. Use the * operator insead of @.")
            return Tensor(self.data * other.data, requires_grad=True, parents=[self, other], creation_op='mul')
        
        # v * v
        if self.data.ndim == 1 and other.data.ndim == 1: # if both are vectors
            if self.data.shape[0] != other.data.shape[0]: # if the vectors are not the same length
                raise ValueError(f"Cannot perform matrix multiplication on tensors with shapes {self.data.shape} and {other.data.shape}.")

        # v * m
        if self.data.ndim == 1 and other.data.ndim > 1:
            if self.data.shape[0] != other.data.shape[-2]:
                raise ValueError(f"Cannot perform matrix multiplication on tensors with shapes {self.data.shape} and {other.data.shape}.")
            
        # m * v
        if self.data.ndim > 1 and other.data.ndim == 1:
            if self.data.shape[-1] != other.data.shape[0]:
                raise ValueError(f"Cannot perform matrix multiplication on tensors with shapes {self.data.shape} and {other.data.shape}.")

        #  m * m
        if self.data.ndim > 1 and other.data.ndim > 1:
            if self.data.shape[-1] != other.data.shape[-2]:
                raise ValueError(f"Cannot perform matrix multiplication on tensors with shapes {self.data.shape} and {other.data.shape}.")

        result = np.matmul(self.data, other.data)
        return Tensor(result, requires_grad=True, parents=[self, other], creation_op='matmul')
    
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
    
    # Unary Operations ============================================
    def __neg__(self):
        return self * -1
    
    def __abs__(self):
        return Tensor(np.abs(self.data), requires_grad=self.requires_grad, parents=[self], creation_op='abs')
    
    # Reduction Operations ========================================
    def sum(self, axis=None):
        if self.is_scalar:
            return self # can't sum a scalar
        return Tensor(self.data.sum(axis=axis), requires_grad=self.requires_grad, parents=[self], creation_op='sum')

    def mean(self, axis=None):
        if self.is_scalar:
            return self # can't mean a scalar
        return Tensor(self.data.mean(axis=axis), requires_grad=self.requires_grad, parents=[self], creation_op='mean')
        
    def max(self, axis=None):
        if self.is_scalar:
            return self # can't max a scalar
        return Tensor(self.data.max(axis=axis), requires_grad=self.requires_grad, parents=[self], creation_op='max')
        
    def min(self, axis=None):
        if self.is_scalar:
            return self # can't min a scalar
        return Tensor(self.data.min(axis=axis), requires_grad=self.requires_grad, parents=[self], creation_op='min')
    
    def std(self, axis=None):
        if self.is_scalar:
            return self # can't std a scalar
        return Tensor(self.data.std(axis=axis), requires_grad=self.requires_grad, parents=[self], creation_op='std')
    
    # Shape Operations ==================================================
    def reshape(self, *new_shape):
        return Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad, parents=[self], creation_op='reshape', \
                    meta_info = {"original_shape" : self.shape})
    
    def squeeze(self, axis=None):
        return Tensor(self.data.squeeze(axis=axis), requires_grad=self.requires_grad, parents=[self], creation_op='squeeze', \
                        meta_info = {"axis" : axis} if axis is not None else None)
    
    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.data, axis=axis), requires_grad=self.requires_grad, parents=[self], creation_op='unsqueeze', \
                        meta_info = {"axis" : axis})

    def transpose(self, axes=None):
        return Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad, parents=[self], creation_op='transpose')
    
    # Comparison Operations =======================================
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data > other)
        elif isinstance(other, Tensor):
            if self.data.shape != other.data.shape:
                raise ValueError("Shape mismatch")
            return Tensor(self.data > other.data)
        else:
            raise TypeError("Unsupported type for comparison")

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data < other)
        elif isinstance(other, Tensor):
            if self.data.shape != other.data.shape:
                raise ValueError("Shape mismatch")
            return Tensor(self.data < other.data)
        else:
            raise TypeError("Unsupported type for comparison")

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data >= other)
        elif isinstance(other, Tensor):
            if self.data.shape != other.data.shape:
                raise ValueError("Shape mismatch")
            return Tensor(self.data >= other.data)
        else:
            raise TypeError("Unsupported type for comparison")

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data <= other)
        elif isinstance(other, Tensor):
            if self.data.shape != other.data.shape:
                raise ValueError("Shape mismatch")
            return Tensor(self.data <= other.data)
        else:
            raise TypeError("Unsupported type for comparison")

    # Indexing Operations ================================================
    def __getitem__(self, index):
        return Tensor(self.data[index])
    
    def __setitem__(self, index, value):
        self.data[index] = value if isinstance(value, Tensor) else Tensor(value)

    # Utility Methods ====================================================
    def _broadcast_tensors(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        a_shape, b_shape = np.broadcast_arrays(self.data, other.data).shape
        return Tensor(self.data * np.ones(a_shape)), Tensor(other.data * np.ones(b_shape))

    def clone(self):
        return Tensor(self.data.copy(), self.requires_grad)
    
    def detach(self):
        return Tensor(self.data, requires_grad=False)
    
    def to(self, device):
        # TODO: Implement device transfer
        # Here is where device transfer would go. NumPy arrays are always on the CPU, so this is a placeholder
        self.device = device
        return self

    # Other Methods =======================================================
    @property
    def T(self, axes=None):
        return self.transpose(axes=axes)
    
    @property
    def size(self):
        s = 1
        for dim in self.shape:
            s *= dim
        return s

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return np.array_equal(self.data, other.data)
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"