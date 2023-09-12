import numpy as np
import pytest

from lib.Tensor_old import Tensor

@pytest.fixture
def vars():
    # define some variables to use in tests
    s = Tensor(5)
    v = Tensor([1, 2, 3, 4, 5])
    v2 = Tensor([2, 3, 4, 5, 6])
    v3 = Tensor([1, 2, 3, 4])
    m = Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    m2 = Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    m3 = Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])
    return s, v, v2, v3, m, m2, m3

# ! Basic Tensor Operations ============================================
def test_addition(vars):
    s, v, v2, v3, m, m2, m3 = vars
    # scalar + scalar
    assert s + s == Tensor(10)
    # scalar + vector
    assert s + v == Tensor([6, 7, 8, 9, 10])
    # vector + scalar
    assert v + s == Tensor([6, 7, 8, 9, 10])
    # vector + vector
    assert v + v2 == Tensor([3, 5, 7, 9, 11])
    # vector + vector (different lengths)
    with pytest.raises(ValueError):
        v + v3
    # vector + matrix
    with pytest.raises(ValueError):
        v + m
    # matrix + scalar
    assert m + s == Tensor([
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14]
    ])
    # matrix + vector
    with pytest.raises(ValueError):
        m + v
    # matrix + matrix
    assert m + m2 == Tensor([
        [2, 4, 6],
        [8, 10, 12],
        [14, 16, 18]
    ])
    # matrix + matrix (different shapes)
    with pytest.raises(ValueError):
        m + m3

def test_subtraction(vars):
    s, v, v2, v3, m, m2, m3 = vars
    # Vector - Vector
    assert (v - v2) == Tensor([-1, -1, -1, -1, -1])
    # Matrix - Matrix
    assert (m - m2) == Tensor(np.zeros((3, 3), dtype=np.int8))

def test_multiplication(vars):
    s, v, v2, v3, m, m2, m3 = vars
    # Scalar * Vector
    assert (s * v) == Tensor([5, 10, 15, 20, 25])
    # Vector * Matrix
    with pytest.raises(ValueError):
        v * m
    # Matrix * Matrix
    assert (m * m2) == Tensor(np.array([
        [1, 4, 9],
        [16, 25, 36],
        [49, 64, 81]
    ]))

def test_division(vars):
    s, v, v2, v3, m, m2, m3 = vars
    # Scalar / Vector
    assert (s / v) == Tensor([5, 2.5, 5/3, 1.25, 1])
    # Vector / Scalar
    assert (v / s) == Tensor([1/5, 2/5, 3/5, 4/5, 1])

def test_matmul(vars):
    s, v, v2, v3, m, m2, m3 = vars
    # Scalar @ Scalar
    assert (s @ s) == Tensor(25)
    # Scalar @ Vector
    assert (s @ v) == Tensor([5, 10, 15, 20, 25])
    # Vector @ Vector
    assert (v @ v2) == Tensor(70)
    # Vector @ Vector (different lengths)
    with pytest.raises(ValueError):
        v @ v3
    # Vector @ Matrix
    assert (v3 @ m3.T) == Tensor([30, 70])
    # Matrix @ Scalar
    assert m @ s == Tensor([
        [5, 10, 15],
        [20, 25, 30],
        [35, 40, 45]
    ])
    # Matrix @ Matrix
    assert (m @ m2) == Tensor([
        [30, 36, 42],
        [66, 81, 96],
        [102, 126, 150]
    ])
    # Matrix @ Matrix (different shapes)
    with pytest.raises(ValueError):
        m @ m3

#! Reduction Operations ========================================
# TODO: test reduction operations

# ! Shape Operations ========================================
# TODO: test shape operations

# ! Indexing Operations ========================================
# TODO: test indexing operations

# ! Utility Methods ========================================
# TODO: test utility methods

# ! Other Methods ========================================
# TODO: test other methods