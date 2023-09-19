## Ben's Mini Deep Learning Framework

### Description
I'm building this mini DL framework to get a better understanding of how things work under the hood. This is part of my ongoing efforts to keep my brain *sticky* (or neuroplastic). Your brain is the most important neural network, after all.

### setup.py
This adds a .pth file to the site-packages directory containing the project's root dir. This allows for absolute imports anywhere without having to install the package or use sys.path.append(). Using this right now because I know it works, but I plan to try `pip install --editable .` in the future.

## Completed
- [X] Add a Dense layer
- [X] Add auto differentiation
- [X] Add basic loss functions
- [X] Fix autodiff for scalars and shape mismatches
- [X] fix backward_pow func (need to handle edge cases)
- [X] Test backprop on dense layer
- [X] Add optimizers
- [X] Add checks for numerical instability
- [X] Finish replicating/reverse engineering Karpathy's [micrograd](https://github.com/karpathy/micrograd)
- [X] make a nn module that mirrors micrograd but with my logic
- [X] update my backward method to use Top sort?
- [X] make a test suite that mirrors micrograd/write sanity checks for Tensor class
  - [X] figure out why certain operation chains are failing and others aren't
- [X] Update nn to use new tensor class
- [X] Add activation functions with tests

## TODO
- [ ] Test log
- [ ] Test exp
- [ ] Test clip
- [ ] Get CE loss test passing
- [ ] Clean up unused code in lib
- [ ] Add a training loop
- [ ] Test the MLP on a simple dataset
- [ ] Add data handling
- [ ] Add evaluation
- [ ] Add dynamic lr optimizer
- [ ] Add regularization
- [ ] Add examples and docs
- [ ] Look into [this](https://vmartin.fr/automatic-jacobian-matrix-computation-with-sympy.html) jacobian matrix optimization

## Backlog