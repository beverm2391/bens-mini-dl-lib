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

#! Basic Tensor Operations ============================================
def test_scalar_and_scalar():
    t = Tensor(5)
    s = 2
    assert t + s == Tensor(7)
    assert t - s == Tensor(3)
    assert t * s == Tensor(10)
    assert t / s == Tensor(2.5)

def test_vector_and_scalar():
    t = Tensor([1, 2, 3, 4, 5])
    s = 2
    assert t + s == Tensor([3, 4, 5, 6, 7])
    assert t - s == Tensor([-1, 0, 1, 2, 3])
    assert t * s == Tensor([2, 4, 6, 8, 10])
    assert t / s == Tensor([0.5, 1, 1.5, 2, 2.5])
    # assert t @ s == Tensor([2, 4, 6, 8, 10]) # TODO fix matmul
    
def test_vector_and_vector():
    v = Tensor([1, 2, 3, 4, 5])
    v2 = Tensor([2, 3, 4, 5, 6])

    assert v + v2 == Tensor([3, 5, 7, 9, 11])
    assert v - v2 == Tensor([-1, -1, -1, -1, -1])
    assert v * v2 == Tensor([2, 6, 12, 20, 30])
    assert v / v2 == Tensor([0.5, 2/3, 3/4, 4/5, 5/6])
    assert v @ v2 == Tensor(70)

def test_vector_and_tensor():
    v = Tensor([1, 2, 3])
    v2 = Tensor([1, 2, 3, 4])
    t = Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    assert v @ t == Tensor([30, 36, 42])

    with pytest.raises(ValueError):
        v2 @ t

def test_tensor_and_tensor():
    t = Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    t2 = Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    t3 = Tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])

    assert t @ t2 == Tensor([
        [30, 36, 42],
        [66, 81, 96],
        [102, 126, 150]
    ])

    with pytest.raises(ValueError):
        t @ t3

def test_all_basic_methods():
    test_scalar_and_scalar()
    test_vector_and_scalar()
    test_vector_and_vector()
    test_vector_and_tensor()
    test_tensor_and_tensor()