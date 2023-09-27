import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import argparse

from lib.Tensor import Tensor
from lib.NN import MLP, ReLU, MSELoss
from lib.Optimizers import SGD

def make_data(n=1000, plot=False):
    x = np.linspace(-10, 10, n) # 100 samples between -10 and 10

    # generate y = 2x + 1
    y = 2 * x + 1
    # add noise
    y += np.random.normal(5, 5, n)


    # reshape x and y to be column vectors
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x, y = shuffle(x, y, random_state=0)

    # split into train and test sets
    train_idx, val_idx, test_idx = int(0.6*n), int(0.8*n), int(1.0*n)

    train_x, val_x, test_x = x[:train_idx], x[train_idx:val_idx], x[val_idx:test_idx]
    train_y, val_y, test_y = y[:train_idx], y[train_idx:val_idx], y[val_idx:test_idx]

    # Plot
    if plot:
        plt.scatter(train_x, train_y, label="Training Data")
        plt.scatter(val_x, val_y, label="Validation Data", alpha=0.6)
        plt.scatter(test_x, test_y, label="Test Data", alpha=0.6)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title('Synthetic Linearly Correlated Data')
        plt.show()  

    # create tensors
    train_inputs = Tensor(train_x, requires_grad=True)
    train_targets = Tensor(train_y, requires_grad=True)

    val_inputs = Tensor(val_x, requires_grad=True)
    val_targets = Tensor(val_y, requires_grad=True)

    test_inputs = Tensor(test_x, requires_grad=True)
    test_targets = Tensor(test_y, requires_grad=True)

    return train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets

def main(args):
    train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = make_data()

    model = MLP([1, 10, 1], ReLU)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.001)

    train_loss_history = []
    val_loss_history = []

    epochs = args.epochs

    for epoch in range(epochs + 1):

        # train phase
        model.train() # set model to train mode
        optimizer.zero_grad() # reset gradients

        train_outputs = model(train_inputs) # forward pass
        train_loss = criterion(train_outputs, train_targets) # compute loss
        train_loss.backward() # compute gradients
        optimizer.step() # update parameters
        train_loss_history.append(train_loss.data) # save loss for plotting

        # validation phase
        model.eval() # set model to eval mode
        with Tensor.no_grad():
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            val_loss_history.append(val_loss.data)
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Train Loss: {train_loss.data:.04f} | Val Loss: {val_loss.data:.04f}")

    if args.plot:
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MLP to fit synthetic data.')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--plot', action='store_true', default=True, help='plot training and validation loss')
    args = parser.parse_args()

    main(args)