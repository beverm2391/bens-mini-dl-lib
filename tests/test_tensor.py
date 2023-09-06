import pytest
import numpy as np
from lib.Tensor import Tensor

def test_addition():
    a = Tensor(np.array([1, 2]))
    b = Tensor(np.array([3, 4]))
    result = a + b
    expected = np.array([4, 6])
    assert np.all(result.data == expected)

def test_subtraction():
    a = Tensor(np.array([5, 6]))
    b = Tensor(np.array([1, 2]))
    result = a - b
    expected = np.array([4, 4])
    assert np.all(result.data == expected)

def test_multiplication():
    a = Tensor(np.array([1, 2]))
    b = Tensor(np.array([3, 4]))
    result = a * b
    expected = np.array([3, 8])
    assert np.all(result.data == expected)

def test_matmul_matrix_matrix():
    a = Tensor(np.array([[1, 2], [3, 4]]))
    b = Tensor(np.array([[5, 6], [7, 8]]))
    result = a @ b
    expected = np.array([[19, 22], [43, 50]])  # Matrix multiplication
    assert np.all(result.data == expected)

def test_matmul_matrix_vector():
    a = Tensor(np.array([[1, 2], [3, 4]]))
    b = Tensor(np.array([5, 6]))
    result = a @ b
    expected = np.array([17, 39])  # [1*5 + 2*6, 3*5 + 4*6]
    assert np.all(result.data == expected)

def test_matmul_vector_vector():
    a = Tensor(np.array([1, 2]))
    b = Tensor(np.array([3, 4]))
    result = a @ b
    expected = 11  # 1*3 + 2*4
    assert result == expected

def test_matmul_dimension_mismatch():
    a = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    b = Tensor(np.array([[5, 6], [7, 8], [9, 10]]))
    with pytest.raises(ValueError):
        a @ b

def test_transpose():
    a = Tensor(np.array([[1, 2], [3, 4]]))
    result = a.T
    expected = np.array([[1, 3], [2, 4]])
    assert np.all(result == expected)

def test_addition_with_scalar():
    a = Tensor(np.array([1, 2]))
    result = a + 2
    expected = np.array([3, 4])
    assert np.all(result.data == expected)

def test_subtraction_with_scalar():
    a = Tensor(np.array([5, 6]))
    result = a - 1
    expected = np.array([4, 5])
    assert np.all(result.data == expected)

def test_multiplication_with_scalar():
    a = Tensor(np.array([1, 2]))
    result = a * 3
    expected = np.array([3, 6])
    assert np.all(result.data == expected)