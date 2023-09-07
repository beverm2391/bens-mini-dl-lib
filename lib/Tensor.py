import numpy as np
import warnings

# ! Old Tensor class, kept for reference
class TensorV1:
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape

        assert self.data.ndim > 0, "Tensor must have at least one dimension (can't be Scalar)."

    # Other refers to the other tensor
    def __add__(self, other):
        if not isinstance(other, TensorV1):
            return TensorV1(self.data + other)
        return TensorV1(self.data + other.data) # element-wise addition

    def __sub__(self, other):
        if not isinstance(other, TensorV1):
            return TensorV1(self.data - other)
        return TensorV1(self.data - other.data) # element-wise subtraction
    
    def __mul__(self, other):
        if not isinstance(other, TensorV1):
            return TensorV1(self.data * other)
        return TensorV1(self.data * other.data) # element-wise multiplication

    def __matmul__(self, other): # matrix multiplication
        # Make sure that the other is a tensor
        # This ensures neither are scalar because the Tensor class wont accept scalars
        if not isinstance(other, TensorV1):
            raise TypeError("The 'other' must be an instance of Tensor.")
        
        # check if the last dimension of self is equal to the second last dimension of other (if neither are vectors)
        if self.data.ndim > 1 and other.data.ndim > 1:
            if self.data.shape[-1] != other.data.shape[-2]:
                raise ValueError(f"Cannot perform matrix multiplication on tensors with shapes {self.data.shape} and {other.data.shape}.")

        result_data = np.matmul(self.data, other.data)

        if np.array(result_data).ndim == 0: # if the result is a scalar, return it (because the Tensor class wont accept scalars)
            return result_data

        return TensorV1(result_data)

    @property
    def T(self):
        return np.transpose(self.data)
    
    def __repr__(self):
        return f"Tensor({self.data.__repr__()})"

# ! Current Tensor class
class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data) # data
        self.shape = self.data.shape # shape of data
        self.requires_grad = requires_grad # whether to calculate gradients
        self.grad = self.zero_grad() if requires_grad else None # gradient of data, if needed
        self.is_scalar = self.data.ndim == 0 # whether the data is a scalar

    # Basic Operations ============================================
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data + other.data)
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data * other.data)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data / other.data)
    
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
    
    # Shape Operations ==================================================
    def reshape(self, *new_shape):
        return Tensor(self.data.reshape(new_shape))
    
    def squeeze(self, axis=None):
        return Tensor(self.data.squeeze(axis=axis))
    
    def unsqueeze(self, axis=None):
        return Tensor(self.data.expand_dims(axis=axis))

    def transpose(self, axes=None):
        return Tensor(np.transpose(self.data, axes))
    
    # Indexing Operations ================================================
    def __getitem__(self, index):
        return Tensor(self.data[index])
    
    def __setitem__(self, index, value):
        self.data[index] = value if isinstance(value, Tensor) else Tensor(value)

    # Utility Methods ====================================================
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

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return np.array_equal(self.data, other.data)
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"