import numpy as np

from lib.Layers import Dense
from lib.Tensor import Tensor

def test_dense_forward():
    np.random.seed(0) # set seed for reproducibility
    layer = Dense(3, 2, 0.01) # init layer, 2 x 3 matrix, learning rate of 0.01

    input_data = Tensor(np.array([1, 2, 3])) # init input data
    output = layer(input_data) # forward pass (output Tensor class instance)
    expected_output = np.matmul(input_data.data, layer.weights.data) + layer.biases.data # expected output
    assert np.allclose(output.data, expected_output), f"Expected {expected_output}, got {output.data}"

def test_dense_backward():
    np.random.seed(0) # set seed for reproducibility
    layer = Dense(3, 2, 0.01) # init layer, 2 x 3 matrix, learning rate of 0.01

    input_data = Tensor(np.array([1, 2, 3])) # init input data
    output = layer(input_data) # forward pass (output Tensor class instance)

    target = Tensor(np.array([1, 0])) # init target data
    
    # TODO finish this test after implementing loss functions
    assert True