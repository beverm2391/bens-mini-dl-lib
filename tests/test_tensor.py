import numpy as np
from typing import Union
import torch

from lib.Tensor import Tensor

# What do I actually need to test here?
# things dont throw errors that shouldnt
# backprop works (do a proof or two)

# ! Basic Tensor Operations ============================================
def test_basic_ops():
    def _basic_ops(a : Union[Tensor, torch.Tensor], b: Union[Tensor, torch.Tensor]):
        c = a + b # add, radd
        d = c - b # sub, rsub
        e = c * 2 # mul scalar
        # ! This is failing below - backward pass is not working 
        x = d * e # mul tensor (elementwise)
        # f = e / d # div, rdiv
        # g = f**2 # pow
        # h = -g # neg
        # h += a # iadd
        # h -= b # isub
        # h *= c # imul
        # h /= d # idiv
        # h **= 2 # ipow

        x = x.sum() # xhave to call sum to get txhe gradient to be computed (because h needs to be a scalar   )
        x.backward() # backprop
        return a, b, x # return for later comparison
    
    data1, data2 = np.random.randn(2, 3), np.random.randn(2, 3) # random data

    # ! Custom Tensor class
    t1_custom = Tensor(data1, requires_grad=True) # init
    t2_custom = Tensor(data2, requires_grad=True) # init

    # ! PyTorch Tensor class
    t1_pt = torch.Tensor(data1).double() # init
    t2_pt = torch.Tensor(data2).double() # init
    t1_pt.requires_grad = True # need gradients
    t2_pt.requires_grad = True # need gradients

    # ! Run the operations
    a_custom, b_custom, x_custom = _basic_ops(t1_custom, t2_custom)
    a_pt, b_pt, x_pt = _basic_ops(t1_pt, t2_pt)

    # ! Check the results
    tol = 1e-6

    # test forward
    assert np.allclose(x_custom.data, x_pt.data.numpy(), atol=tol), "Forward pass failed."
    # test backward\
    print(f"custom grad: {a_custom.grad}")
    print(f"pt grad: {a_pt.grad.numpy()}")
    assert np.allclose(a_custom.grad, a_pt.grad.numpy(), atol=tol), "Backward pass failed."
    assert np.allclose(b_custom.grad, b_pt.grad.numpy(), atol=tol), "Backward pass failed."

# ! More Tensor Operations =============================================
def test_matmul():
    pass

def test_reduction():
    pass

# ! Activation Functions ===============================================
def test_activation_functions():
    pass

if __name__ == '__main__':
    test_basic_ops() # Run the basic operations test
    print("Basic operations test passed!")