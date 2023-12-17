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
- [X] Test log
- [X] Test exp
- [X] Test clip
- [X] fix tensor backward __add__ method
- [X] clean out old methods in Tensor once fully implemented and tested
- [X] Test the MLP on a simple synthetic dataset (train/val), get it converging, add necessary methods
- [X] update no_grad context manager via thread local approach
- [X] fix the reshape method, and test
- [X] write a test for mean
- [X] make a sequential model class like pytorch

## TODO TASKS
- Just found this amazing book, [Understanding Deep Learning](https://udlbook.github.io/udlbook/) with detailed explanations and example implementations. Putting this here so I can use it as a guide moving forward.
- [softmax backprop stuff](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

- [ ] get MNISTv4 working with my lib
  - [X] finish adding subscriptable indexing to tensor class (__getitem__ backward)
  - [X] fix clip test
  - [X] write a test for dot product
  - [X] write a test for sum with axis and keepdims
  - [X] add keepdims to sum, max
  - [X] write a test for max with axis and keepdims
  - [X] write softmax and log softmax forward test
  - [X] fix BinaryCELoss test
  - [ ] write log softmax func and test
  - [ ] write negative log likelihood func and test
  - [ ] write CCELoss
  - [ ] get CatCELoss test passing

- [ ] add keepdims to rest of reduction ops**
- [ ] add a simple dataloader, with option to load all in memory at once
- [ ] write tests for the sequential model class
- [ ] figure out what basic layers I need to implement
- [ ] Clean up unused code in lib
- [ ] Add data handling
- [ ] Add evaluation
- [ ] Add dynamic lr optimizer
- [ ] Add regularization
- [ ] Add examples and docs
- [ ] write a test for the new no_grad context manager

## TODO IDEAS
- [ ] watch [this video](https://www.youtube.com/watch?v=VMj-3S1tku0) by Karpathy on backprop and learn from/copy his abstractions
- [ ] replicate some of [these models/demos](https://github.com/probml/pyprobml/tree/master/notebooks/book1/13) with this lib
- [ ] reverse engineer some of [these operator abstractions](https://github.com/wilson-labs/cola) to see how they work 
- [ ] Look into [this](https://vmartin.fr/automatic-jacobian-matrix-computation-with-sympy.html) jacobian matrix optimization

## Backlog