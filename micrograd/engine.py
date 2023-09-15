class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data # value
        self.grad = 0 # zero out the gradient
        # internal vars
        self._backward = lambda: None # default backward pass (no op)
        self._prev = set(_children) # parents
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad # update gradient
            other.grad += out.grad # update gradient
        out._backward = _backward # override the default backward pass

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad # update gradient
            other.grad += self.data * out.grad # update gradient
        out._backward = _backward # override the default backward pass

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad # update gradient
        out._backward = _backward # override the default backward pass

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward # override the default backward pass  

        return out

    def backward(self):
        
        # topological order all of the children in the graph
        topo = [] # empty list
        visited = set() # empty set
        def build_topo(v): 
            if v not in visited: # if we haven't visited the node yet
                visited.add(v) # mark as visited
                for child in v._prev: # recursively build topological ordering
                    build_topo(child) # recursive call
                topo.append(v) # add to topological sort
        build_topo(self) # start from self

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1 # gradient of final node is 1
        for v in reversed(topo): # iterate in reverse topological order
            v._backward() # call the _backward method

    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"