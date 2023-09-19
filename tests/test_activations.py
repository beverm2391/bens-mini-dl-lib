import numpy as np
import torch
import pytest
from lib.TensorV2 import Tensor
from lib.NN import ReLU, Sigmoid, Tanh, LeakyReLU

def test_relu_activation():
    data = np.random.rand(2, 3) * 2 - 1
    a = Tensor(data, requires_grad=True)
    relu = ReLU()
    result_a = relu(a)
    result_a.sum().backward()

    a_torch = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    relu_torch = torch.nn.ReLU()
    result_a_torch = relu_torch(a_torch)
    result_a_torch.sum().backward()

    assert np.allclose(result_a.data, result_a_torch.data.numpy())
    assert np.allclose(a.grad, a_torch.grad.data.numpy())

def test_sigmoid_activation():
    data = np.random.rand(2, 3)
    a = Tensor(data, requires_grad=True)
    sigmoid = Sigmoid()
    result_a = sigmoid(a)
    result_a.sum().backward()

    a_torch = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    sigmoid_torch = torch.nn.Sigmoid()
    result_a_torch = sigmoid_torch(a_torch)
    result_a_torch.sum().backward()

    assert np.allclose(result_a.data, result_a_torch.data.numpy())
    assert np.allclose(a.grad, a_torch.grad.data.numpy())

def test_tanh_activation():
    data = np.random.rand(2, 3)
    a = Tensor(data, requires_grad=True)
    tanh = Tanh()
    result_a = tanh(a)
    result_a.sum().backward()

    a_torch = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    tanh_torch = torch.nn.Tanh()
    result_a_torch = tanh_torch(a_torch)
    result_a_torch.sum().backward()

    assert np.allclose(result_a.data, result_a_torch.data.numpy())
    assert np.allclose(a.grad, a_torch.grad.data.numpy())

def test_leaky_relu_activation():
    data = np.random.rand(2, 3) * 2 - 1  # Random data in range [-1, 1]
    a = Tensor(data, requires_grad=True)
    leaky_relu = LeakyReLU()
    result_a = leaky_relu(a)
    result_a.sum().backward()

    a_torch = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    leaky_relu_torch = torch.nn.LeakyReLU()
    result_a_torch = leaky_relu_torch(a_torch)
    result_a_torch.sum().backward()

    assert np.allclose(result_a.data, result_a_torch.data.numpy())
    assert np.allclose(a.grad, a_torch.grad.data.numpy())