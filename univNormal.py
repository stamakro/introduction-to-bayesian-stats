import numpy as np
import pymc3 as pm

np.random.seed(1991)

#sample size
N = 50

#random variable, normally distributed with mean and std.
m_true = 1.
sigma_true = 2


x = np.random.normal(m_true, sigma_true, N)

model = pm.Model()

with model:
	#parameters of the prior of the mean
	mu0 = 0.
	sigma0 = 3.
	mu = pm.Normal('mean', mu=mu0, sigma=sigma0)

	#assume variance is known

	#observed data / likelihood model
	xobs = pm.Normal('data', mu=mu, sigma=sigma_true, observed=x)

	trace = pm.sample(500)
	


