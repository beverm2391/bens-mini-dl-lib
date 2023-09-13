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
  
## TODO

I'm currently working on refactoring my base Tensor to handle more dtypes for both forward and backward (autograd) operations. This has been significantly more complex than I thought... but that means I'm learning a lot.


- [ ] Test backprop on dense layer
  - [ ] fix backward_pow func (need to handle edge cases)
- [ ] Find some way to write tests for the activation functions
- [ ] Add a network class
- [ ] Add a training loop
- [ ] Add data handling
- [ ] Add evaluation
- [ ] Add optimizers
- [ ] Add regularization
- [ ] Add examples and docs
- [ ] Look into [this](https://vmartin.fr/automatic-jacobian-matrix-computation-with-sympy.html) jacobian matrix optimization

## Backlog
- [ ] Finish Tensor tests
  - [X] Basic ops
  - [ ] Reduction ops
  - [ ] Shape
  - [ ] Indexing
  - [ ] Utility
  - [ ] Other
  - [ ] Auto diff