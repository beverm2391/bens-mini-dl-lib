class NeuralNetwork:
    def __init__(self, layers=[]):
        self.layers = layers
        self.loss = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self):
        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)