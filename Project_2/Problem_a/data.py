import numpy as np
import matplotlib.pyplot as plt

class DataPoints:
    """
    Here the input is just n, the number of datapoints for a third degree polynomial with some noise.
    The output in output() function is just datapoints x, y and designMatrix X.

    Example of this class use:

    inst = DataPoints(100)
    x, y, X = inst.output()


    """
    def __init__(self, n):
        self.n = n


    def output(self):
        np.random.seed(10)
        self.x = 2 * np.random.randn(self.n,1)
        self.y = -0.5*self.x**3 + 1.5*self.x**2 + 1.5*self.x + 3*np.random.randn(self.n,1)

        X = np.c_[np.ones((self.n, 1)), self.x, self.x**2, self.x**3]

        #Hessian matrix:
        H = (2.0/self.n) * X.T @ X
        EigVal, EigVec = np.linalg.eig(H)
        print(f"The eigenvalues for the current Hessian matrix are {EigVal}")
        print(f"If they are positive, then our cost function is a convex function")

        return self.x, self.y, X




if __name__ == '__main__':
    inst = DataPoints(100)
    x, y, X = inst.output()
