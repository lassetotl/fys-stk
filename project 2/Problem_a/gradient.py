def gradient(n, X, y, Beta):
    return (2.0/n) * X.T @ (X @ Beta - y)
