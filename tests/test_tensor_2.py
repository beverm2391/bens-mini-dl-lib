import numpy as np
import torch

from lib.Tensor import Tensor

def test():
    data1, data2 = np.random.randn(2, 3), np.random.randn(2, 3) # random data

    # ! Custom Tensor class
    t1 = Tensor(data1, requires_grad=True) # init
    t2 = Tensor(data2, requires_grad=True) # init
    pt1 = torch.Tensor(data1).double().requires_grad_(True) # pytorch
    pt2 = torch.Tensor(data2).double().requires_grad_(True) # pytorch

    def ops(a, b):
        c = a + b 
        d = c * b
        return d.sum()

    result1 = ops(t1, t2)
    result2 = ops(pt1, pt2) 

    result1.backward()
    result2.backward()

    assert np.allclose(result1.data, result2.data.numpy(), atol=1e-6), "Forward pass failed."
    assert np.allclose(t1.grad, pt1.grad.numpy(), atol=1e-6), "Backward pass failed."
    assert np.allclose(t2.grad, pt2.grad.numpy(), atol=1e-6), "Backward pass failed."

if __name__ == '__main__':
    test()