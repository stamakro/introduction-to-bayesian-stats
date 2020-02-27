#!/usr/bin/env python3
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from sys import argv

np.random.seed(1)

if __name__ == "__main__":
    N = int(argv[1]) # sample size

    # True parameters 
    muTrue = np.array([5, 0])
    covTrue = np.array([[0.5, 0], [0, 2.0]])

    # Simulate data
    X = np.random.multivariate_normal(muTrue, covTrue, size=N)

    # Define model
    model = pm.Model()
    with model:
        mu = pm.Normal('mu', mu=0, sd=1, shape=2) # NOW shape makes sense!!
        cov = tt.diag(pm.HalfNormal('cov', sd=1, shape=2)) # google told me so

        likelihood = pm.MvNormal('data', mu=mu, cov=cov, observed=X)
        trace = pm.sample(1000)
    print(pm.summary(trace))

