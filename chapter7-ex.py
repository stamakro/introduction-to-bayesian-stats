import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal,invwishart

# Makes things easier
def GetInv(m): return np.linalg.solve(m,np.eye(m.shape[1]))

data = pd.read_csv('../reading.csv', index_col=0, names=['pretest','posttest'], header=0)
Y = data.values

# Set prior parameters
np.random.seed(0)
mu0 = np.array([50,50])
L0 = np.array([[625,312.5],[312.5,625]])
nu0 = 4
S0 = np.array([[625,312.5],[312.5,625]])

n = Y.ndim
ybar, sigma = Y.mean(axis=0), data.cov().values # because numpy is stupid

NSim = 5000
YSim = np.zeros((NSim,n))
thetaSim = np.zeros((NSim,n))
sigmaSim = np.zeros((NSim,n,n))
for s in range(NSim):
    if s % 1000 == 0: print("Sampling {}".format(s))
    # Update theta
    Ln = GetInv(GetInv(L0) + n*GetInv(sigma))
    mun = np.dot(Ln, np.dot(GetInv(L0),mu0) + n*np.dot(GetInv(sigma),ybar))
    theta = multivariate_normal.rvs(mun,Ln,1)

    # Update sigma
    Sn = S0 + np.dot((Y-np.full(Y.shape,theta)).transpose(),(Y-np.full(Y.shape,theta)))
    sigma = GetInv(invwishart.rvs(nu0+n, GetInv(Sn),1))
    YSim[s] = multivariate_normal.rvs(theta,sigma,1)
    thetaSim[s],sigmaSim[s] = theta,sigma

