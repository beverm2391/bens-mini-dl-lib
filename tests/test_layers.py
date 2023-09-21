import numpy as np
import torch

from lib.NN import Dense, BatchNorm, Dropout
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


def BYPASS_test_batchnorm():
    num_features = 4
    batch_size = 10

    # Create instances of custom and PyTorch BatchNorm layers
    custom_bn = BatchNorm(num_features)
    pt_bn = torch.nn.BatchNorm1d(num_features, affine=True, eps=1e-5, momentum=0)

    # Initialize both layers with the same gamma and beta
    with torch.no_grad():
        pt_bn.weight.copy_(torch.tensor(custom_bn.gamma.data))
        pt_bn.bias.copy_(torch.tensor(custom_bn.beta.data))

    # Generate random input
    pt_input = torch.randn(batch_size, num_features)
    custom_input = Tensor(pt_input.numpy())

    # Forward pass
    pt_output = pt_bn(pt_input)
    custom_output = custom_bn(custom_input)

    # Check if the outputs are close
    assert np.allclose(pt_output.detach().numpy(), custom_output.data, atol=1e-6)


def BYPASS_test_dropout():
    p = 0.5
    shape = (10, 4)

    # Create custom Dropout layer
    custom_dropout = Dropout(p=p)

    # Generate random input
    custom_input = Tensor(np.random.randn(*shape))

    # Forward pass
    custom_output = custom_dropout(custom_input)

    # Check if the Dropout layer has zeroed approximately p fraction of input
    zero_fraction = np.mean(custom_output.data == 0)
    assert np.isclose(zero_fraction, p, atol=0.1)