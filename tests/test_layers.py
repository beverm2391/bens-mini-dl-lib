import numpy as np
import torch

from lib.NN import Dense
from lib.TensorV2 import Tensor

def test_dense():
    input_dim = 5
    output_dim = 3
    batch_size = 10

    pt_dense = torch.nn.Linear(input_dim, output_dim)
    dense = Dense(input_dim, output_dim)

    # copy the weights and biases from custom layer to pytorch layer
    with torch.no_grad():
        pt_dense.weight.copy_(torch.tensor(dense.weights.data))
        pt_dense.bias.copy_(torch.tensor(dense.biases.data))

    pt_input = torch.randn(batch_size, input_dim) # pytorch tensor
    input = Tensor(pt_input.numpy()) # copy the pytorch tensor

    pt_output = pt_dense(pt_input) # pytorch output
    output = dense(input) # custom output

    # compare the outputs
    assert np.allclose(pt_output.detach().numpy(), output.data, atol=1e-6)