class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data # value
        self.grad = 0 # zero out the gradient
        # internal vars
        self._backward = lambda: None # default backward pass (no op)
        self._prev = set(_children) # parents
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __pow__(self, other):
        pass

    def relu(self):
        pass

    def backward(self):
        pass

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