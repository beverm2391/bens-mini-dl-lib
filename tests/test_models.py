import torch
import numpy as np

from lib.TensorV2 import Tensor
from lib.NN import MLP, ReLU

def test_mlp():
    input_dim = 5
    hidden_dim = 10
    output_dim = 2
    batch_size = 32

    layer_dims = [input_dim, hidden_dim, output_dim]

    # Create your custom MLP and PyTorch MLP
    mlp = MLP(layer_dims, activation_fn=ReLU)
    pt_mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim)
    )

    with torch.no_grad():
        my_dense_0_params = mlp.get_module("dense_0").parameters()
        pt_mlp[0].weight.copy_(torch.tensor(my_dense_0_params[0].data))
        pt_mlp[0].bias.copy_(torch.tensor(my_dense_0_params[1].data))

        my_dense_1_params = mlp.get_module("dense_1").parameters()
        pt_mlp[2].weight.copy_(torch.tensor(my_dense_1_params[0].data))
        pt_mlp[2].bias.copy_(torch.tensor(my_dense_1_params[1].data))

        # Create random input tensor
        pt_input = torch.randn(batch_size, input_dim)
        my_input = Tensor(pt_input.numpy())  # Assuming you have a way to convert numpy arrays to your Tensor class

        # Forward pass
        pt_output = pt_mlp(pt_input)
        my_output = mlp(my_input)

        # print(f"My output: {my_output.data[:2]}")
        # print(f"PyTorch output: {pt_output[:2]}")

        assert np.allclose(my_output.data, pt_output.detach().numpy(), atol=1e-6)