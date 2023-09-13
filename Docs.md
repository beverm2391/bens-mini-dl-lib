## Docs



## AutoGrad Implementation
### Scalars and 0-Dimensional Arrays/Tensors

From a 2017 Pytorch Issue, [introduce torch.Scalar to represent scalars in autograd](https://github.com/pytorch/pytorch/issues/1433):

>First, we want at least 0-dimensional Variables in PyTorch. Consider if we had only (immutable) Scalars and your loss is a .sum() over some variable. Now, if you treat this as an immutable scalar, either it has a grad and can be optimized (which is weird for an immutable value), or it doesn’t (and you have to hack around that in some way to still be able to do training).