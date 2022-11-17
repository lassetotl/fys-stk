def gradient(n, X, y, Beta, lmbd):
    return (2.0/n) * X.T @ (X @ Beta-y)+2*lmbd*Beta
