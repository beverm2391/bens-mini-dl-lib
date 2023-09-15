import numpy as np

class SynthecicData:
    @staticmethod
    def linear(n: int, range: tuple = (-10, 10), noise: float = 5, seed: int = 0):
        """
        Linear data with noise
        """
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
    
    @staticmethod
    def classification(n: int, classes: int = 2, dimensions: int = 2, seed: int = 0):
        """
        Binary classification data
        """
        np.random.seed(seed)
        X = np.random.randn(n, dimensions)
        y = np.random.randint(0, classes, size=(n, 1))
        return X, y

    @staticmethod
    def time_series(n: int, seasonality: bool = False, noise: float = 0.5, seed: int = 0):
        """
        Time series data
        """
        np.random.seed(seed)
        t = np.arange(0, n)
        y = np.sin(0.02 * t)
        if seasonality:
            y += np.sin(0.1 * t)
        y += noise * np.random.randn(n)
        return t.reshape(-1, 1), y.reshape(-1, 1)

    @staticmethod
    def high_dimensional(n: int, dimensions: int = 100, seed: int = 0):
        """
        High dimensional data used to test the curse of dimensionality
        """
        np.random.seed(seed)
        X = np.random.randn(n, dimensions)
        y = np.sum(X, axis=1) + np.random.randn(n)
        return X, y.reshape(-1, 1)

    @staticmethod
    def sparse_data(n: int, dimensions: int = 10, sparsity: float = 0.9, seed: int = 0):
        """
        Spare data
        """
        np.random.seed(seed)
        X = np.random.randn(n, dimensions)
        mask = np.random.rand(n, dimensions) < sparsity
        X[mask] = 0
        y = np.sum(X, axis=1) + np.random.randn(n)
        return X, y.reshape(-1, 1)

    @staticmethod
    def with_outliers(n: int, outliers: int = 5, seed: int = 0):
        """
        Data containing outliers
        """
        np.random.seed(seed)
        X, y = SynthecicData.linear(n)
        if outliers > 0:
            X[:outliers] += 20 * np.random.randn(outliers, 1)
            y[:outliers] += 50 * np.random.randn(outliers, 1)
        return X, y