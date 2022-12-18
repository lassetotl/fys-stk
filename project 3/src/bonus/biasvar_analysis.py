import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn-darkgrid")

from sklearn.model_selection import train_test_split
from mlxtend.evaluate        import bias_variance_decomp
from sklearn.utils           import resample
from sklearn.linear_model    import LinearRegression, Ridge, Lasso
from sklearn.neural_network  import MLPRegressor
from sklearn.ensemble        import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


from tqdm import tqdm, trange
from random import random, seed
np.random.seed(2019) 


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# From lecture notes on Ridge/Lasso regression (4.8)
def create_X(x, y, n): 
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X

# Making meshgrid of random datapoints and computing Franke's function
N = 40
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

x_, y_ = np.meshgrid(x,y)

sigma = 0.1 # Variance of noise 
noise = np.random.normal(0, sigma, len(x_)*len(x_)) 
noise = noise.reshape(len(x_), len(x_))

z = FrankeFunction(x_, y_) + noise #Franke with added noise



def biasvar(method, solver, max_complexity): 
    """ Calculates the bias-variance and MSE for a chosen method and max 
        complexity 

        Input
            solver: The linear regression solver you want to use 
            n: complexity 
        Output: 
            error (mean squared error), bias and variance 
    """
    n_boostraps = 100

    # Initialize arrays 
    error    = np.zeros(max_complexity)
    bias     = np.zeros(max_complexity)
    variance = np.zeros(max_complexity)

    for n in trange(max_complexity):
        X_ = create_X(x_, y_, n+1)
        X_train_, X_test_, z_train_, z_test_ = train_test_split(X_, z.ravel(), test_size=0.2, random_state = 40)
        z_pred = np.empty((z_test_.shape[0], n_boostraps))

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train_)
        X_train_ = scaler.transform(X_train_)
        X_test_ = scaler.transform(X_test_)
        
        for k in range(n_boostraps):
            # Resample for bootsraps 
            X_, z_ = resample(X_train_, z_train_)

            if method == 'linreg': 
                z_pred[:, k] = solver.fit(X_, z_).predict(X_test_).ravel() 

            elif method == 'neuralnetwork': 
                # hidden_layers = (n+1, n+1, n+1)
                hidden_layers = (n+1, n+1)
                # hidden_layers = (n+1)
                lr = 0.01
                solver = MLPRegressor(hidden_layer_sizes=hidden_layers, activation='relu', solver='adam', max_iter=1000)
                z_pred[:, k] = solver.fit(X_, z_).predict(X_test_).ravel() 

            elif method == 'deeplearning': 
                # model = solver(max_depth=n+1)
                if solver == 'randomforest': 
                    model = RandomForestRegressor(max_depth=n+1)
                elif solver == 'decision': 
                    model = DecisionTreeRegressor(max_depth=n+1) 
                z_pred[:, k] = model.fit(X_, z_).predict(X_test_).ravel() 

        # Calculate MSE, bias and variance 
        error[n] = np.mean( np.mean((z_test_.reshape(-1, 1) - z_pred)**2, axis=1, keepdims=True) )
        bias[n] = np.mean( (z_test_.reshape(-1, 1) - np.mean(z_pred, axis=1, keepdims=True))**2 )
        # variance[n] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
        variance[n] = np.mean((z_pred - np.mean(z_pred, axis = 1, keepdims = True))**2)

    return error, bias, variance 

def plot_linreg(save=False): 
    """ Plots the bias-variance and MSE for OLS and Ridge for max 
        polynomial order n = 13 
    """
    n = 13
    order = np.array([b+1 for b in range(n)])

    ## OLS 
    error_ols, bias_ols, variance_ols       = biasvar(method='linreg', solver=LinearRegression(), max_complexity=n)
    fig = plt.figure(figsize = (6.5, 5)) 
    plt.plot(order, error_ols-sigma**2, color="k", label='Error',    linewidth = 2)
    plt.plot(order, bias_ols-sigma**2,  color="g", label='Bias',     linewidth = 1.5)
    plt.plot(order, variance_ols,       color="r", label='Variance', linewidth = 1.5)
    plt.xlabel('Model Complexity', fontsize = 15)
    plt.ylabel('Error', fontsize = 15)
    plt.legend(); plt.grid(1)
    if save: 
        plt.savefig('figures/bonus/biasvar_ols.pdf')
    else: 
        plt.show()

    ## Ridge 
    n = 25
    order = np.array([b+1 for b in range(n)])
    error_ridge, bias_ridge, variance_ridge = biasvar(method='linreg', solver=Ridge(alpha=1e-4), max_complexity=n)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 5)) 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (6.5, 10)) 
    ax1.plot(order, error_ridge-sigma**2, color="k", label='Error',    linewidth = 2)
    ax1.plot(order, bias_ridge-sigma**2,  color="g", label='Bias',     linewidth = 1.5)
    ax1.plot(order, variance_ridge,       color="r", label='Variance', linewidth = 1.5)
    ax1.set_xlabel('Model Complexity', fontsize = 15)
    ax1.set_ylabel(r'Error ($\lambda=10^{-4}$)', fontsize = 15)

    error_ridge, bias_ridge, variance_ridge = biasvar(method='linreg', solver=Ridge(alpha=1e-9), max_complexity=n)
    ax2.plot(order, error_ridge-sigma**2, color="k", label='Error',    linewidth = 2)
    ax2.plot(order, bias_ridge-sigma**2,  color="g", label='Bias',     linewidth = 1.5)
    ax2.plot(order, variance_ridge,       color="r", label='Variance', linewidth = 1.5)
    ax2.set_xlabel('Model Complexity', fontsize = 15)
    ax2.set_ylabel(r'Error ($\lambda=10^{-9}$)', fontsize = 15)
    plt.legend(); plt.grid(1); plt.tight_layout()
    if save: 
        plt.savefig('figures/bonus/biasvar_ridge.pdf')
    else: 
        plt.show()

def plot_deeplearning(save=False): 
    """ Plots the bias-variance and MSE for MLP for 
        two hidden layers with n = 30 nodes 
    """
    nodes = 25 # number of nodes 
    neurons = np.linspace(1, nodes+1, nodes)
    error, bias, variance = biasvar(method='neuralnetwork', solver=None, max_complexity=nodes)
    
    fig = plt.figure(figsize = (6.5, 5)) 
    plt.plot(neurons, error-sigma**2, color="k", label='Error',    linewidth = 2)
    plt.plot(neurons, bias-sigma**2,  color="g", label='Bias',     linewidth = 1.5)
    plt.plot(neurons, variance,       color="r", label='Variance', linewidth = 1.5)
    plt.xlabel('Model Complexity', fontsize = 15)
    plt.ylabel('Error', fontsize = 15)
    plt.legend(); plt.grid(1)
    if save: 
        plt.savefig('figures/bonus/biasvar_mlp.pdf')
    else: 
        plt.show()

def plot_ensemble(save=False): 
    """ Plots the bias-variance and MSE for decision trees and 
        the random forest regressor 
    """
    depth = 10 # max depth 
    hidden_neurons = np.linspace(1, depth+1, depth)

    error, bias, variance = biasvar(method='deeplearning', solver='decision', max_complexity=depth) 

    fig = plt.figure(figsize = (6.5, 5)) 
    plt.plot(hidden_neurons, error-sigma**2, color="k", label='Error',    linewidth = 2)
    plt.plot(hidden_neurons, bias-sigma**2,  color="g", label='Bias',     linewidth = 1.5)
    plt.plot(hidden_neurons, variance,       color="r", label='Variance', linewidth = 1.5)
    plt.xlabel('Model Complexity', fontsize = 15)
    plt.ylabel('Error', fontsize = 15)
    plt.legend(); plt.grid(1)
    if save: 
        plt.savefig('figures/bonus/biasvar_decisiontrees.pdf')
    else: 
        plt.show()

    error, bias, variance = biasvar(method='deeplearning', solver='randomforest', max_complexity=depth) 
    fig = plt.figure(figsize = (6.5, 5)) 
    plt.plot(hidden_neurons, error-sigma**2, color="k", label='Error',    linewidth = 2)
    plt.plot(hidden_neurons, bias-sigma**2,  color="g", label='Bias',     linewidth = 1.5)
    plt.plot(hidden_neurons, variance,       color="r", label='Variance', linewidth = 1.5)
    plt.xlabel('Model Complexity', fontsize = 15)
    plt.ylabel('Error', fontsize = 15)
    plt.legend(); plt.grid(1)
    if save: 
        plt.savefig('figures/bonus/biasvar_randomforests.pdf')
    else: 
        plt.show()



plot_linreg(save=False)
plot_deeplearning(save=False)
plot_ensemble(save=False)

