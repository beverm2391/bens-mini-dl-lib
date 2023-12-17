import numpy as np
import torch
import pytest
from lib.Tensor import Tensor
from lib.NN import ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, LogSoftmax, NegativeLogLikelihoodLoss

np.random.seed(0) # Set seed for reproducibility

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


def test_softmax():
    data = np.random.rand(2, 3) * 2 - 1 
    
    a = Tensor(data, requires_grad=True)
    softmax = Softmax()

    a_pt = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    softmax_pt = torch.nn.Softmax(dim=-1)

    result_a = softmax(a)
    result_a_pt = softmax_pt(a_pt)

    assert np.allclose(result_a.sum(axis=-1).data, np.ones((2, 1)))
    assert np.allclose(result_a.data, result_a_pt.data.numpy())

def test_log_softmax():
    data = np.random.rand(2, 3) * 2 - 1

    a = Tensor(data, requires_grad=True)
    log_softmax = LogSoftmax()

    a_pt = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    log_softmax_pt = torch.nn.LogSoftmax(dim=-1)

    result_a = log_softmax(a)
    result_a_pt = log_softmax_pt(a_pt)

    assert np.all(result_a.data <= 0) and result_a.sum().data < 0 # for log softmax, all values should be <= 0 and sum should be < 0
    assert np.allclose(result_a.data, result_a_pt.data.numpy())