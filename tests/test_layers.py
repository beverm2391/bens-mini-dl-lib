import numpy as np
import torch

from lib.NN import Dense
from lib.TensorV2 import Tensor

def BYPASS_test_dense_forward():
    np.random.seed(0) # set seed for reproducibility
    layer = Dense(3, 2, 0.01) # init layer, 2 x 3 matrix, learning rate of 0.01

    input_data = Tensor(np.array([1, 2, 3])) # init input data
    output = layer(input_data) # forward pass (output Tensor class instance)
    expected_output = np.matmul(input_data.data, layer.weights.data) + layer.biases.data # expected output
    assert np.allclose(output.data, expected_output), f"Expected {expected_output}, got {output.data}"

def BYPASS_test_dense_backward():
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

    assert np.allclose(layer.weights.grad, expected_weights_grad), f"Expected {expected_weights_grad}, got {layer.weights.grad.data}"


def BYPASS_test_dense_fc():
    t_data = np.random.rand(2, 3) # 2 samples, 3 features
    w_data = np.random.rand(3, 4) # 3 features, 4 outputs
    b_data = np.random.rand(1, 4) # 1 bias, 4 outputs

    x = Tensor(t_data, requires_grad=True) # init input data
    w = Tensor(w_data, requires_grad=True) # init weights
    b = Tensor(b_data, requires_grad=True) # init biases

    layer = Dense(3, 4) # init layer with shape (2, 3)
    layer.weights = w # set weights
    layer.biases = b # set biases
    out = layer(x) # forward pass

    x_torch = torch.tensor(t_data, dtype=torch.float32, requires_grad=True) # init input data
    w_torch = torch.tensor(w_data, dtype=torch.float32, requires_grad=True) # init weights
    b_torch = torch.tensor(b_data, dtype=torch.float32, requires_grad=True) # init biases

    layer_torch = torch.nn.Linear(3, 4)
    layer_torch.weight.data = w_torch # set weights
    layer_torch.bias.data = b_torch # set biases
    out_torch = layer_torch(x_torch) # forward pass

    assert np.allclose(out.data, out_torch.detach().numpy()), f"Expected {out_torch.detach().numpy()}, got {out.data}" 