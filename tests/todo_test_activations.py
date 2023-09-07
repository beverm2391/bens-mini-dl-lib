import pytest
import numpy as np

from lib.Tensor import Tensor
from lib.Activations import *

def test_relu_forward():
    relu = ReLU()
    input_data = Tensor(np.array([[-1, -0.5], [0, 0.5], [1, 2]]))
    expected_forward_output = Tensor(np.array([[0, 0], [0, 0.5], [1, 2]]))
    assert np.allclose(relu(input_data), expected_forward_output)