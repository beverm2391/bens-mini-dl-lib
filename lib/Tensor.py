import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape

        assert self.data.ndim > 0, "Tensor must have at least one dimension (can't be Scalar)."

    # Other refers to the other tensor
    def __add__(self, other):
        if not isinstance(other, Tensor):
            return Tensor(self.data + other)
        return Tensor(self.data + other.data) # element-wise addition

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            return Tensor(self.data - other)
        return Tensor(self.data - other.data) # element-wise subtraction
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            return Tensor(self.data * other)
        return Tensor(self.data * other.data) # element-wise multiplication

    def __matmul__(self, other): # matrix multiplication
        # Make sure that the other is a tensor
        # This ensures neither are scalar because the Tensor class wont accept scalars
        if not isinstance(other, Tensor):
            raise TypeError("The 'other' must be an instance of Tensor.")
        
        # check if the last dimension of self is equal to the second last dimension of other (if neither are vectors)
        if self.data.ndim > 1 and other.data.ndim > 1:
            if self.data.shape[-1] != other.data.shape[-2]:
                raise ValueError(f"Cannot perform matrix multiplication on tensors with shapes {self.data.shape} and {other.data.shape}.")

        result_data = np.matmul(self.data, other.data)

        if np.array(result_data).ndim == 0: # if the result is a scalar, return it (because the Tensor class wont accept scalars)
            return result_data

        return Tensor(result_data)

    @property
    def T(self):
        return np.transpose(self.data)
    
    def __repr__(self):
        return f"Tensor({self.data.__repr__()})"