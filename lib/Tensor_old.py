import numpy as np
import warnings

# This Tesor class does not support auto differentiation. It is a placeholder for the auto-differentiating Tensor class.

class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data) # data
        self.shape = self.data.shape # shape of data
        self.requires_grad = requires_grad # whether to calculate gradients
        self.grad = self.zero_grad() if requires_grad else None # gradient of data, if needed
        self.is_scalar = self.data.ndim == 0 # whether the data is a scalar

    # Basic Operations ============================================
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data + other)
        elif isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            raise ValueError(f"Unsupported type for addition: {type(other)}")

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data - other)
        elif isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            raise ValueError(f"Unsupported type for subtraction: {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data * other)
        elif isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            raise ValueError(f"Unsupported type for multiplication: {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data / other)
        elif isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            raise ValueError(f"Unsupported type for division: {type(other)}")

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data ** other.data)
    
    def __matmul__(self, other):
        if self.is_scalar:
            return Tensor(self.data * other.data)
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        if self.data.ndim == 0 and other.data.ndim == 0: # if both are scalars
            warnings.warn("Both of your inputs are scalars. Using element-wise multiplication instead. Use the * operator insead of @.")
            return self.data * other.data
        
        if self.data.ndim == 0 and other.data.ndim > 0: # if self is a scalar and other is not
            warnings.warn("One of your inputs is a scalar. Using element-wise multiplication instead. Use the * operator insead of @.")
            return Tensor(self.data * other.data)
        
        if self.data.ndim > 0 and other.data.ndim == 0: # if self is not a scalar and other is
            warnings.warn("One of your inputs is a scalar. Using element-wise multiplication instead. Use the * operator insead of @.")
            return Tensor(self.data * other.data)
        
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
        return Tensor(result)
    
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

    # Reduction Operations ========================================
    def sum(self, axis=None):
        if self.is_scalar:
            return self # can't sum a scalar
        return Tensor(self.data.sum(axis=axis))

    def mean(self, axis=None):
        if self.is_scalar:
            return self # can't mean a scalar
        return Tensor(self.data.mean(axis=axis))
        
    def max(self, axis=None):
        if self.is_scalar:
            return self # can't max a scalar
        return Tensor(self.data.max(axis=axis))
        
    def min(self, axis=None):
        if self.is_scalar:
            return self # can't min a scalar
        return Tensor(self.data.min(axis=axis))
    
    def std(self, axis=None):
        if self.is_scalar:
            return self # can't std a scalar
        return Tensor(self.data.std(axis=axis))
    
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

    # Shape Operations ==================================================
    def reshape(self, *new_shape):
        return Tensor(self.data.reshape(new_shape))
    
    def squeeze(self, axis=None):
        return Tensor(self.data.squeeze(axis=axis))
    
    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.data, axis=axis))

    def transpose(self, axes=None):
        return Tensor(np.transpose(self.data, axes))
    
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

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

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