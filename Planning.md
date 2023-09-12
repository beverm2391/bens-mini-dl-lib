## Principles and Planning

### Desired Functionality
1. Tensor class
   1. shape-agnostic operations (broadcasting)
   2. autograd
   4. indexing
   5. slicing
   6. dtype-agnostic
   7. device agnostc
2. Optimizers
3. Layers and models
4. Regularization
5. Loss functions
6. Utils

#### Ops List
**Forward**
1. Addition (__add__): Element-wise addition of two Tensors or a Tensor and a scalar.
2. Subtraction (__sub__): Element-wise subtraction of two Tensors or a Tensor and a scalar.
3. Multiplication (__mul__): Element-wise multiplication of two Tensors or a Tensor and a scalar.
4. Division (__truediv__): Element-wise division of two Tensors or a Tensor by a scalar.
5. Matrix Multiplication (__matmul__): Matrix multiplication of two Tensors.
6. Exponentiation (__pow__): Element-wise exponentiation of a Tensor.
7. Negation (__neg__): Negates the elements of a Tensor.
8. Transpose (transpose or T): Transpose of a matrix or higher-dimensional Tensor.
9. Reshape: Changes the shape of a Tensor.
10. Slicing and Indexing: Retrieve a subset or a single element from the Tensor.

**Backward**
1. Addition: The gradient is distributed equally to both parent Tensors.
2. Subtraction: The gradient is distributed to the first parent and the negative gradient to the second parent.
3. Multiplication: Distributes the gradient element-wise multiplied by the second parent to the first parent, and vice versa.
4. Division: More complicated; involves distributing the gradient while considering both the numerator and the denominator.
5. Matrix Multiplication: Involves transposing and multiplying the parent matrices with the gradient.
6. Exponentiation: Depends on the value of the exponent; usually involves logarithms and other parent data.
7. Negation: Passes through the negative of the gradient.
8. Transpose: The gradient is also transposed.
9. Reshape: The gradient is reshaped to match the original shape of the Tensor.
10. Slicing and Indexing: The gradient is placed in an array of zeros that match the original shape, at the indexed position.