import numpy as np

from lib.Tensor import Tensor

# What do I actually need to test here?
# things dont throw errors that shouldnt
# backprop works (do a proof or two)

# ! Basic Tensor Operations ============================================
def test_addition():
    v1 = Tensor([1, 2, 3])
    v2 = Tensor([4, 5, 6])

    v3 = v1 + v2
    # print(v3)
    expected = Tensor([5, 7, 9])

    assert v3 == expected, "Vector addition failed"

def test_matmul():

    m1 = Tensor([
        [1, 2, 3],
        [4, 5, 6]
    ], requires_grad=True)

    m2 = Tensor([
        [7, 8],
        [9, 10],
        [11, 12]
    ], requires_grad=True)

    m3 = m1 @ m2

    expected = Tensor([
        [58, 64],
        [139, 154]
    ])

    assert m3 == expected, "Matrix multiplication forward failed"

    # test backward
    m3.backward()

    expected_grad_1 = np.array([
        [15, 19, 23],
        [15, 19, 23]
    ])

    expected_grad_2 = np.array([
        [5, 5],
        [7, 7],
        [9, 9]
    ])

    assert np.all(np.isclose(m1.grad, expected_grad_1)), "Matrix multiplication backward failed"
    assert np.all(np.isclose(m2.grad, expected_grad_2)), "Matrix multiplication backward failed"

if __name__ == "__main__":
    test_addition()
    test_matmul()