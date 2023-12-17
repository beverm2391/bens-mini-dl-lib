import pytest
import numpy as np
import torch

from lib.Tensor import Tensor
from lib.NN import MSELoss, BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss, LogSoftmax, NegativeLogLikelihoodLoss

def test_MSE():
    x_data = np.array([0.1, 0.2, 0.3, 0.4])
    y_data = np.array([0.0, 0.2, 0.4, 0.6])

    x = Tensor(x_data, requires_grad=True)
    y = Tensor(y_data, requires_grad=True)

    criterion = MSELoss()
    mse = criterion(x, y)
    mse.backward()

    x_torch = torch.tensor(x_data, requires_grad=True)
    y_torch = torch.tensor(y_data, requires_grad=True)

    criterion_torch = torch.nn.MSELoss()
    mse_torch = criterion_torch(x_torch, y_torch)
    mse_torch.backward()

    assert np.allclose(mse.data, mse_torch.data.numpy()), f"mse.data: {mse.data}\nmse_torch.data: {mse_torch.data.numpy()}"
    assert np.allclose(x.grad, x_torch.grad.numpy()), f"x.grad: {x.grad}\nx_torch.grad: {x_torch.grad.numpy()}"
    assert np.allclose(y.grad, y_torch.grad.numpy()), f"y.grad: {y.grad}\ny_torch.grad: {y_torch.grad.numpy()}"

def test_negative_log_likelihood_loss():
    data = np.random.rand(2, 3) * 2 - 1

    a = Tensor(data, requires_grad=True)
    log_softmax = LogSoftmax()
    nll = NegativeLogLikelihoodLoss()

    result = nll(log_softmax(a), Tensor(np.array([0, 1])))
    result.backward()

    a_pt = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    log_softmax_pt = torch.nn.LogSoftmax(dim=-1)
    nll_pt = torch.nn.NLLLoss()

    result_pt = nll_pt(log_softmax_pt(a_pt), torch.tensor([0, 1]))
    result_pt.backward()


    assert np.allclose(result.data, result_pt.data.numpy())
    assert np.allclose(a.grad, a_pt.grad.numpy())

def test_BinaryCrossEntropyLoss():
    x_data = np.array([0.1, 0.8, 0.4, 0.6])
    y_data = np.array([0, 1, 0, 1])

    x = Tensor(x_data, requires_grad=True)
    y = Tensor(y_data, requires_grad=True)

    criterion = BinaryCrossEntropyLoss()
    loss = criterion(x, y)
    loss.backward()

    # Using PyTorch's built-in BCELoss
    x_torch = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y_data, dtype=torch.float32, requires_grad=True)

    criterion_torch = torch.nn.BCELoss()
    loss_torch = criterion_torch(x_torch, y_torch)
    loss_torch.backward()

    assert np.allclose(loss.data, loss_torch.data.numpy()), f"loss.data: {loss.data}\nloss_torch.data: {loss_torch.data.numpy()}"
    assert np.allclose(x.grad, x_torch.grad.numpy()), f"x.grad: {x.grad}\nx_torch.grad: {x_torch.grad.numpy()}"
    assert np.allclose(y.grad, y_torch.grad.numpy()), f"y.grad: {y.grad}\ny_torch.grad: {y_torch.grad.numpy()}"

def test_CategoricalCrossEntropyLoss():
    x_data = np.array([[0.1, 0.6, 0.3], [0.8, 0.1, 0.1], [0.3, 0.1, 0.6], [0.2, 0.2, 0.6]]) # prob distributions
    y_data = np.array([1, 0, 2, 2])

    x = Tensor(x_data, requires_grad=True)
    y = Tensor(y_data, requires_grad=False)

    criterion = CategoricalCrossEntropyLoss()
    loss = criterion(x, y)
    loss.backward()

    # Using PyTorch's built-in BCELoss
    x_torch = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y_data, dtype=torch.long, requires_grad=False)

    criterion_torch = torch.nn.CrossEntropyLoss()
    loss_torch = criterion_torch(x_torch, y_torch)
    loss_torch.backward()

    assert np.allclose(loss.data, loss_torch.data.numpy()), f"loss.data: {loss.data}\nloss_torch.data: {loss_torch.data.numpy()}"
    assert np.allclose(x.grad, x_torch.grad.numpy()), f"x.grad: {x.grad}\nx_torch.grad: {x_torch.grad.numpy()}"