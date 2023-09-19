import pytest
import numpy as np
import torch

from lib.TensorV2 import Tensor
from lib.NN import MSELoss

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
