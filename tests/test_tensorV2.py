import numpy as np
import torch
import pytest

from lib.TensorV2 import Tensor

# ! Scalar Ops =========================================================
def test_scalar_addition_common():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    
    a_torch = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
    
    c = a + b
    c_torch = a_torch + b_torch
    
    c.backward()
    c_torch.backward()
    
    assert np.allclose(a.grad, a_torch.grad.item(), atol=1e-6)

def test_scalar_addition_edge():
    a = Tensor(1e-9, requires_grad=True)
    b = Tensor(1e9, requires_grad=True)
    
    a_torch = torch.tensor(1e-9, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(1e9, dtype=torch.float32, requires_grad=True)
    
    c = a + b
    c_torch = a_torch + b_torch
    
    c.backward()
    c_torch.backward()
    
    assert np.allclose(a.grad, a_torch.grad.item(), atol=1e-6)

def test_scalar_multiplication_by_zero():
    a = Tensor(np.array([1.0]), requires_grad=True)
    b = Tensor(np.array([0.0]), requires_grad=True)
    c = a * b
    loss = c.sum()  # Ensuring it's a scalar
    loss.backward()

    a_torch = torch.tensor([1.0], requires_grad=True)
    b_torch = torch.tensor([0.0], requires_grad=True)
    c_torch = a_torch * b_torch
    loss_torch = c_torch.sum()
    loss_torch.backward()

    assert np.allclose(a.grad, a_torch.grad.numpy())

# ! Vector Ops =========================================================
def test_vector_addition_common():
    a = Tensor(np.array([1, 2]), requires_grad=True)
    b = Tensor(np.array([3, 4]), requires_grad=True)
    c = a + b
    loss = c.sum()  # Ensuring it's a scalar
    loss.backward()
    
    a_torch = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor([3, 4], dtype=torch.float32, requires_grad=True)
    c_torch = a_torch + b_torch
    loss_torch = c_torch.sum() # Ensuring it's a scalar
    loss_torch.backward()
    
    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_vector_of_zeros():
    a = Tensor(np.array([1, 2]), requires_grad=True)
    b = Tensor(np.zeros(2), requires_grad=True)
    c = a + b
    loss = c.sum()
    loss.backward()

    a_torch = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
    b_torch = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    c_torch = a_torch + b_torch
    loss_torch = c_torch.sum()
    loss_torch.backward()

    assert np.allclose(a.grad, a_torch.grad.numpy())


# ! Matrix Ops =========================================================
def test_matrix_addition_common():
    a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
    b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True)
    c = a + b
    loss = c.sum()  # Ensuring it's a scalar
    loss.backward()
    
    a_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, requires_grad=True)
    c_torch = a_torch + b_torch
    loss_torch = c_torch.sum() # Ensuring it's a scalar
    loss_torch.backward()
    
    assert np.allclose(a.grad, a_torch.grad.numpy())

def test_matrix_row_of_zeros():
    a = Tensor(np.array([[1, 2], [0, 0]]), requires_grad=True)
    b = Tensor(np.array([[3, 4], [5, 6]]), requires_grad=True)
    c = a + b
    loss = c.sum()
    loss.backward()

    a_torch = torch.tensor([[1, 2], [0, 0]], dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor([[3, 4], [5, 6]], dtype=torch.float32, requires_grad=True)
    c_torch = a_torch + b_torch
    loss_torch = c_torch.sum()
    loss_torch.backward()

    assert np.allclose(a.grad, a_torch.grad.numpy())

# ! More Tests =========================================================

def test_multiple():
    def _ops(a, b):
        c = a + b
        d = a * b
        e = c * d
        f = e ** 2
        f.sum().backward()
    
    data1 = np.random.rand(2, 3)
    data2 = np.random.rand(2, 3)

    a = Tensor(data1, requires_grad=True)
    b = Tensor(data2, requires_grad=True)

    _ops(a, b)

    a_torch = torch.tensor(data1, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(data2, dtype=torch.float32, requires_grad=True)

    _ops(a_torch, b_torch)

    assert np.allclose(a.grad, a_torch.grad.numpy()), f"Expected {a_torch.grad.numpy()} but got {a.grad}"
    assert np.allclose(b.grad, b_torch.grad.numpy()), f"Expected {b_torch.grad.numpy()} but got {b.grad}"

def test_matmul():
    data1 = np.random.rand(2, 3)
    data2 = np.random.rand(3, 2)

    a = Tensor(data1, requires_grad=True)
    b = Tensor(data2, requires_grad=True)

    c = a @ b
    c.sum().backward()

    a_torch = torch.tensor(data1, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(data2, dtype=torch.float32, requires_grad=True)

    c_torch = a_torch @ b_torch
    c_torch.sum().backward()

    assert np.allclose(a.grad, a_torch.grad.numpy()), f"Expected {a_torch.grad.numpy()} but got {a.grad}"
    assert np.allclose(b.grad, b_torch.grad.numpy()), f"Expected {b_torch.grad.numpy()} but got {b.grad}"

def test_multiple_2():
    def _ops(a, b):
        c = a + b # add, radd
        d = c - b # sub, rsub
        e = c * 2 # mul scalar
        x = d * e # mul tensor (elementwise)
        return x.sum()

    data1 = np.random.rand(2, 3)
    data2 = np.random.rand(2, 3)

    a = Tensor(data1, requires_grad=True)
    b = Tensor(data2, requires_grad=True)

    loss = _ops(a, b)
    loss.backward()

    a_torch = torch.tensor(data1, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(data2, dtype=torch.float32, requires_grad=True)

    loss_torch = _ops(a_torch, b_torch)
    loss_torch.backward()

    assert np.allclose(a.grad, a_torch.grad.numpy()), f"Expected {a_torch.grad.numpy()} but got {a.grad}"
    assert np.allclose(b.grad, b_torch.grad.numpy()), f"Expected {b_torch.grad.numpy()} but got {b.grad}"

def test_max():
    data1 = np.random.rand(2, 3)
    data2 = np.random.rand(2, 3)

    a = Tensor(data1, requires_grad=True)
    b = Tensor(data2, requires_grad=True)

    c = a + b
    c.max().backward()

    a_torch = torch.tensor(data1, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(data2, dtype=torch.float32, requires_grad=True)

    c_torch = a_torch + b_torch
    c_torch.max().backward()

    assert np.allclose(a.grad, a_torch.grad.numpy()), f"Expected {a_torch.grad.numpy()} but got {a.grad}"
    assert np.allclose(b.grad, b_torch.grad.numpy()), f"Expected {b_torch.grad.numpy()} but got {b.grad}"

def test_scalar_division():
    data1 = np.random.rand(2, 3)
    data2 = np.random.rand(2, 3)

    a = Tensor(data1, requires_grad=True)
    b = Tensor(data2, requires_grad=True)

    b = a / 2
    c = a + b
    c.sum().backward()

    a_torch = torch.tensor(data1, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(data2, dtype=torch.float32, requires_grad=True)

    b_torch = a_torch / 2
    c_torch = a_torch + b_torch
    c_torch.sum().backward()

    assert np.allclose(a.grad, a_torch.grad.numpy()), f"Expected {a_torch.grad.numpy()} but got {a.grad}"


def test_log():
    data = np.random.rand(2, 3) + 1  # Add 1 to avoid log(0)

    a = Tensor(data, requires_grad=True)
    log_a = a.log()
    log_a.sum().backward()

    a_torch = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    log_a_torch = torch.log(a_torch)
    log_a_torch.backward(torch.ones_like(log_a_torch))

    assert np.allclose(log_a.data, log_a_torch.data.numpy()), f"Expected {log_a_torch.data.numpy()} but got {log_a.data}"
    assert np.allclose(a.grad, a_torch.grad.numpy()), f"Expected {a_torch.grad.numpy()} but got {a.grad}"

def test_exp():
    data = np.random.rand(2, 3)

    a = Tensor(data, requires_grad=True)
    exp_a = a.exp()
    exp_a.sum().backward()

    a_torch = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    exp_a_torch = torch.exp(a_torch)
    exp_a_torch.backward(torch.ones_like(exp_a_torch))

    assert np.allclose(exp_a.data, exp_a_torch.data.numpy()), f"Expected {exp_a_torch.data.numpy()} but got {exp_a.data}"
    assert np.allclose(a.grad, a_torch.grad.numpy()), f"Expected {a_torch.grad.numpy()} but got {a.grad}"


def BYPASStest_clip():
    data = np.random.rand(2, 3) * 10

    a = Tensor(data, requires_grad=True)
    a = a.clip(2, 8)
    a.sum().backward()

    a_torch = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    a_torch = torch.clip(a_torch, 2, 8)
    a_torch.sum().backward()

    assert np.allclose(a.data, a_torch.data.numpy()), f"Expected {a_torch.data.numpy()} but got {a.data}"
    assert np.allclose(a.grad, a_torch.grad.numpy()), f"Expected {a_torch.grad.numpy()} but got {a.grad}"

# TODO ================================================================
def test_in_place_operations():
    pass

def test_broadcasting():
    pass

def test_advanced_operations():
    # exp, log, sin, cos, tan, etc.
    pass

def test_backward_with_non_unity_gradient():
    # cases where the backward() method is called with a gradient other than the default (which is usually 1)
    pass

def test_invalid_shapes():
    # with pytest.raises(ValueError):
    pass