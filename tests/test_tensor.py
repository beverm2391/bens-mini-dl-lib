import numpy as np
import pytest

from lib.Tensor import Tensor


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