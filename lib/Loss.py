import numpy as np

class Loss:
    def check_shape(self, y_hat, y):
        if y_hat.shape != y.shape:
            raise ValueError(f"Shape mismatch: y_hat has shape {y_hat.shape} but y has shape {y.shape}")

    def forward(self, y_hat, y):
        raise NotImplementedError("The forward method is not implemented.")
    
    def backward(self, y_hat, y):
        raise NotImplementedError("The backward method is not implemented.")
    
    def __call__(self, y_hat, y, *args, **kwargs):
        return self.forward(y_hat, y, *args, **kwargs)

class MSE(Loss):
    def forward(self, y_hat, y):
        self.check_shape(y_hat, y)
        self.y_hat = y_hat  # store for backward pass
        self.y = y  # store for backward pass
        return ((y_hat - y) ** 2).mean()
    
    def backward(self):
        # derivative of MSE w.r.t y_hat
        return 2 * (self.y_hat - self.y) / self.y.size

class MAE(Loss):
    def forward(self, y_hat, y):
        self.check_shape(y_hat, y)
        self.y_hat = y_hat  # store for backward pass
        self.y = y  # store for backward pass
        return (abs(y_hat - y)).mean()
    
    def backward(self):
        # derivative of MAE w.r.t y_hat
        return np.sign(self.y_hat - self.y) / self.y.size