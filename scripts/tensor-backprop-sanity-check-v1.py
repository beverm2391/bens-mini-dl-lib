import numpy as np
import torch

from lib.Tensor import Tensor

def create_tensors(data1 = None, data2=None):
    if data1 is None:
        data1 = np.random.randn(2, 3)
    if data2 is None:
        data2 = np.random.randn(2, 3)

    t1 = Tensor(data1, requires_grad=True) # init
    t2 = Tensor(data2, requires_grad=True) # init
    pt1 = torch.Tensor(data1).double().requires_grad_(True) # pytorch
    pt2 = torch.Tensor(data2).double().requires_grad_(True) # pytorch
    return t1, t2, pt1, pt2

def tester(a, b):
    c = a + b # add, radd
    d = c - b # sub, rsub
    e = c * 2 # mul scalar
    x = d * e # mul tensor (elementwise)
    return x.sum()

if __name__ == '__main__':
    t1, t2, pt1, pt2 = create_tensors()
    t3 = tester(t1, t2)
    pt3 = tester(pt1, pt2)

    t3.backward()
    pt3.backward()

    t3.trace_requires_grad()