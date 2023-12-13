from typing import Union
import math
import numpy as np

from lib.Tensor import Tensor

class SimpleDataLoader:
    def __init__(self, X: Union[np.ndarray, Tensor], Y: Union[np.ndarray, Tensor], batch_size=64, shuffle=True):
        self.X = Tensor(X) if not isinstance(X, Tensor) else X
        self.Y = Tensor(Y) if not isinstance(Y, Tensor) else Y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        xlen  = self.X.shape[0]
        indices = np.random.permutation(xlen) if self.shuffle else np.arange(xlen) # shuffled indices if self.shuffle else range

        for start_idx in range(0, xlen, self.batch_size):
            end_idx = min(self.batch_size + start_idx, xlen)
            batch_indices = Tensor(indices[start_idx:end_idx]) # this has to be tensor because indexing with np.ndarray or list raises error in tinygrad

            yield self.X[batch_indices], self.Y[batch_indices]

    def __len__(self): # return the number of batches
        return math.ceil(self.X.shape[0] / self.batch_size) 