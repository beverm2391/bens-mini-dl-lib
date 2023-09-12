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
    
    loss = (output - target) ** 2 # calculate loss
    loss.backward() # backpropagate loss

    # calculate expected gradients
    expected_weights_grad = np.array([[2, 4, 6], [0, 0, 0]])
    expected_biases_grad = np.array([2, 4])

    assert np.allclose(layer.weights.grad.data, expected_weights_grad), f"Expected {expected_weights_grad}, got {layer.weights.grad.data}"