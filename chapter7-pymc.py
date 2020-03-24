#!/usr/bin/env python3
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from sys import argv
import pickle

np.random.seed(1)

if __name__ == "__main__":
    try:
        N = int(argv[1]) # sample size
    except IndexError:
        N = 1000

    # True parameters
    #muTrue = np.array([5, 0])
    #covTrue = np.array([[0.5, 0], [0, 2.0]])

    # Simulate data
    #X = np.random.multivariate_normal(muTrue, covTrue, size=N)

    X = np.zeros((22, 2), float)
    with open('reading.csv' ) as f:
        for i, line in enumerate(f):
            if i > 0:
                fields = line.split(',')
                X[i-1, 0] = float(fields[1])
                X[i-1, 1] = float(fields[2])

    print(np.mean(X,0))
    print(np.cov(X.T))

    # Define model
    model = pm.Model()
    with model:
        mu = pm.Normal('mu', mu=50, sd=25, shape=2) # NOW shape makes sense!!

        std_d = pm.HalfNormal.dist(sigma=20, shape=2) # google told me so
        #std_d = pm.HalfCauchy.dist(beta=2.5) #
        packed_chol = pm.LKJCholeskyCov('chol_cov', eta=1., n=2, sd_dist=std_d)
        chol = pm.expand_packed_triangular(2, packed_chol, lower=True)

        likelihood = pm.MvNormal('data', mu=mu, chol=chol, observed=X)
        trace = pm.sample(N)
    print(pm.summary(trace))

    with open('trace.pkl', 'wb') as f:
        pickle.dump(trace, f)
