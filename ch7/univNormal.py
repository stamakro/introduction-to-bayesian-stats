import numpy as np
import pymc3 as pm
from bayesplots import plot_errordist_violin
from scipy.stats import pearsonr

np.random.seed(1991)

def autocorr(x, k):
	N = x.shape[0]
	assert k < N

	x1 = x[k:]
	x = x[:-k]

	assert x.shape == x1.shape

	return 	pearsonr(x, x1)[0]



#sample size
N = 20

#random variable, normally distributed with mean and std.
m_true = 10.
sigma_true = 1

k = 5
Nsamples = 500

x = np.random.normal(m_true, sigma_true, N)

print('observed mean %f' % np.mean(x))

model = pm.Model()

with model:
	#parameters of the prior of the mean
	mu0 = 0.
	sigma0 = 1.
	mu = pm.Normal('mean', mu=mu0, sigma=sigma0)

	#assume variance is known

	#observed data / likelihood model
	xobs = pm.Normal('data', mu=mu, sigma=sigma_true, observed=x)

	trace = pm.sample(Nsamples)
	trace2 = pm.sample(Nsamples * k)

m = np.array([t['mean'] for t in trace])
print(np.mean(m))
print(autocorr(m, 1))


mt = np.array([t['mean'] for t in trace2])

m2 = mt[np.arange(0, Nsamples*k, k)]
print(np.mean(m2))
print(autocorr(m2,1))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plot_errordist_violin(ax, [1], [trace['mean']])
fig.savefig('test_univ.png', dpi=600)
