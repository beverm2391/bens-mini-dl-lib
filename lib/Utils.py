import numpy as np

class SynthecicData:
    @staticmethod
    def linear(n: int, range: tuple = (-10, 10), noise: float = 5, seed: int = 0):
        np.random.seed(seed)
        x = np.linspace(range[0], range[1], n) # 100 samples between -10 and 10
        # generate y = 2x + 1
        y = 2 * x + 1
        # add noise
        y += np.random.normal(0, noise, n)

        # reshape x and y to be column vectors
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return x, y