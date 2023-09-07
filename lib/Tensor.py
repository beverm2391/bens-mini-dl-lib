import numpy as np

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

    # Constructor Methods =========================================
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

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
    
    # Others ==================================================
    def reshape(self, *new_shape):
        return Tensor(self.data.reshape(new_shape))
    
    @property
    def T(self):
        return np.transpose(self.data)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return np.array_equal(self.data, other.data)
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"