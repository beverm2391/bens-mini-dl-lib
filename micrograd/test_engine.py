import torch  # Importing PyTorch library
from micrograd.engine import Value  # Importing the Value class from micrograd's engine

def test_sanity_check():
    # This function performs a sanity check to validate the behavior of the Micrograd engine.
    # It will perform the same computations using both Micrograd and PyTorch and compare the results.

    #! Micrograd
    x = Value(-4.0)  # Initialize a Value object for Micrograd with initial value -4.0
    z = 2 * x + 2 + x  # Build a computational graph by combining the Value object in various operations
    q = z.relu() + z * x  # More computations: ReLU activation of z plus the product of z and x
    h = (z * z).relu()  # Square of z passed through ReLU
    y = h + q + q * x  # Further computations
    y.backward()  # Perform backpropagation to compute gradients
    xmg, ymg = x, y  # Store the final Value objects for later comparison (Micrograd versions)

    #! PyTorch
    x = torch.Tensor([-4.0]).double()  # Initialize a PyTorch Tensor with initial value -4.0, and set it to double precision
    x.requires_grad = True  # Indicate that we will need to compute gradients with respect to x
    z = 2 * x + 2 + x  # Same computational graph as with Micrograd
    q = z.relu() + z * x  # Similar computations as above
    h = (z * z).relu()  # Square of z, ReLU activated
    y = h + q + q * x  # Further computations
    y.backward()  # Perform backpropagation to compute gradients
    xpt, ypt = x, y  # Store the final Tensor objects for later comparison (PyTorch versions)

    # forward pass went well
    assert ymg.data == ypt.data.item()  # Check if the forward pass results are the same in Micrograd and PyTorch
    # backward pass went well
    assert xmg.grad == xpt.grad.item()  # Check if the gradients computed during backpropagation are the same
