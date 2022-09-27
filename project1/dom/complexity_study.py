import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


def MSE(y_data,y_model): #mean square error
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#from lecture notes on Ridge/Lasso regression
def create_X(x, y, n): #design matrices for polynomials up to 10th degree
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

# Making meshgrid of datapoints and compute Franke's function
n = 5 #in task b; up to 5 (this affects the approximate terrain plot)
N = 100
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

x_, y_ = np.meshgrid(x,y)

#how does the scaling affect the experiment?
var = 0.1 #variance of noise (?)
noise = np.random.normal(0, var, len(x_)*len(x_))
noise = noise.reshape(len(x_), len(x_))

z = FrankeFunction(x_, y_) + noise #Franke with noise

#######################################################################


def model_complexity(degrees, trial):
    """
    It is assumed that you have defined func. for MSE(y_data, y_model) globaly.
    It is also assumed that you have defined design matrix function. create_X
    (specifically for Franke's function).

    Input:
        degrees: maximum degree range for complexity calc. Recommended: 30
        trial: number of resamples. Recommended: 100

    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample
    from sklearn.metrics import mean_squared_error

    #testerror og trainerror skal plottes mot polynomial
    testerror = np.zeros(degrees)
    trainerror = np.zeros(degrees)
    polynomial = np.zeros(degrees)


    for polydegree in range(degrees):
        polynomial[polydegree] = polydegree

        #Jeg vet egentlig ikke hvorfor man setter disse lik null, let me know hvis du finner det ut.
        testerror[polydegree] = 0.0
        trainerror[polydegree] = 0.0

        for samples in range(trial):
            #Dette er resampling. Jeg antar at train_test_split velger tilfeldige
            #verdier fra X? Hvis ikke, så er det vel ikke resampling, bare
            #unødvendig gjentagelse.
            X = create_X(x_, y_, polydegree)
            x_train, x_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size=0.2)
            #Den np.ravel(z) flater ut meshridden for z.

            #Lager array med fitted z's
            model = LinearRegression(fit_intercept=False).fit(x_train, z_train)
            z_predict_train = model.predict(x_train)
            z_predict_test = model.predict(x_test)

            testerror[polydegree] += MSE(z_test, z_predict_test)
            trainerror[polydegree] += MSE(z_train, z_predict_train)
            #Her tar vi MSE for alle samples, og deler det med antall samples.
            #På denne måten finner vi expectation value (<MSE>).

        testerror[polydegree] /= trial
        trainerror[polydegree] /= trial
        print(f"Loading: {(polydegree/degrees) * 100}%") #For høye verdier av degrees så tar det litt lang tid.
        # print("Degree of polynomial: %3d"% polynomial[polydegree])
        # print("Mean squared error on training data: %.8f" % trainerror[polydegree])
        # print("Mean squared error on test data: %.8f" % testerror[polydegree])

    plt.plot(polynomial, np.log10(trainerror), label='Training Error')
    plt.plot(polynomial, np.log10(testerror), label='Test Error')
    plt.title("Test and training error as a function of model complexity")
    plt.xlabel('Polynomial degree')
    plt.ylabel('log10[MSE]')
    plt.grid(True)
    plt.legend()
    plt.savefig("Complexity_study.png")
    plt.show()
model_complexity(35, 30)
