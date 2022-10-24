import numpy as np

class Gradient_function:
    def __init__(self, n, X, y, Beta):
        self.n = n
        self.y = y
        self.X = X
        self.Beta = Beta

    def regular_gradient(self):
        return (2.0/self.n) * self.X.T @ (self.X @ self.Beta - self.y)
