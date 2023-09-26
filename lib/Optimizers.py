from typing import List

from lib.Tensor import Tensor

class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self):
        self.lr = None

    def step(self) -> None:
        raise NotImplementedError
    
    def zero_grad(self) -> None:
        raise NotImplementedError
    
class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer.
    """
    def __init__(self, params: List[Tensor], lr: float):
        self.lr = lr
        self.params = params
    
    def step(self) -> None:
        for param in self.params:
            if param.requires_grad:
                param.data = param.data - self.lr * param.grad # subtract becase we are minimizing loss
                # param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.requires_grad:
                param.zero_grad()