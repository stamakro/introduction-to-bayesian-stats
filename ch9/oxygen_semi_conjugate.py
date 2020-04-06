#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm, multivariate_normal, gaussian_kde
from sys import argv
import pickle


def SSR(b):
    return np.sum((y - X.dot(b)) ** 2 )

def inverse(X):
    m, n = X.shape
    assert m == n

    return np.linalg.solve(X, np.eye(m))



np.random.seed(1)

X = np.array([[ 1.,  0., 23.,  0.],
       [ 1.,  0., 22.,  0.],
       [ 1.,  0., 22.,  0.],
       [ 1.,  0., 25.,  0.],
       [ 1.,  0., 27.,  0.],
       [ 1.,  0., 20.,  0.],
       [ 1.,  1., 31., 31.],
       [ 1.,  1., 23., 23.],
       [ 1.,  1., 27., 27.],
       [ 1.,  1., 28., 28.],
       [ 1.,  1., 22., 22.],
       [ 1.,  1., 24., 24.]])


y = np.array([ -0.87, -10.74,  -3.27,  -1.97,   7.5 ,  -7.25,  17.05,   4.96,
        10.4 ,  11.05,   0.26,   2.51])

n,p = X.shape

b_ols = inverse(X.T.dot(X)).dot(X.T).dot(y)
sigma2_ols = np.sum((y - X.dot(b_ols)) ** 2) / (n - p)
cov_ols = inverse(X.T.dot(X)) * sigma2_ols
ss = np.sqrt(np.diag(cov_ols))

N = 10000

#parameters of prior of sigma^2
# ν0 = 1
# σ0^2 = σ_ols
#for beta
#μ0 = 0
#Σ0 = σ_ols^2 * I
nu0 = 1
sigma0 = sigma2_ols
#a0 = 0.5
#b0 = sigma2_ols * 0.5


#the code below assumes that the prior mean is the 0 vector
mu0 = np.zeros(p)
ss = sigma2_ols
ss = 1000

b_post = np.zeros((N, p))
s2_post = np.zeros(N)

xTx = X.T.dot(X)

for i in range(N):
    if i == 0:
        s2_post[i] = sigma2_ols

    else:
        s2_post[i] = invgamma.rvs(0.5 * (nu0 + n), scale=0.5 * (nu0 * sigma0 + SSR(b_post[i-1])), size=1)

    Vb = inverse((xTx / s2_post[i]) + (np.eye(p) / ss))

    #Vb =  inverse((xTx + np.eye(p)) / s2_post[i])

    meanthing = X.T.dot(y) / s2_post[i]

    Eb = Vb.dot(meanthing)

    b_post[i] = np.random.multivariate_normal(Eb, Vb)



print (np.mean(b_post, 0))
print (np.std(b_post, axis=0, ddof=1))

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
d1 = gaussian_kde(b_post[:,1])

#xx = np.linspace(1.1* np.min(b_post[:,1]), 1.1*np.max(b_post[:,1]), 300)
xx = np.linspace(-60, 60, 500)


priorstd = np.ones(p) * np.sqrt(ss)
ax.hist(b_post[:,1], bins=40, color='k', alpha=0.3, density=True, edgecolor='k')
ax.plot(xx, d1(xx), color='C0', label='posterior')
ax.plot(xx, norm.pdf(xx, loc=0, scale=priorstd[1]), color='C1', label='prior')
ax.set_title('b2')
ax.legend()

ax = fig.add_subplot(1,2,2)
d2 = gaussian_kde(b_post[:,3])
#xx = np.linspace(1.1* np.min(b_post[:,3]), 1.1*np.max(b_post[:,3]), 300)
xx = np.linspace(-4, 4, 500)

ax.hist(b_post[:,3], bins=40, color='k', alpha=0.3, density=True, edgecolor='k')
ax.plot(xx, d2(xx), color='C0', label='posterior')
ax.plot(xx, norm.pdf(xx, loc=0, scale=priorstd[3]), color='C1', label='prior')
ax.set_title('b4')
ax.legend()

fig.savefig('9_2_oxygen_semiconjugate_coefficients.png')


ages = np.arange(20, 32)
deltas = np.zeros((N, ages.shape[0]))
for i, (b2, b4) in enumerate(zip(b_post[:,1], b_post[:,3])):
    deltas[i] = b2 + b4 * ages

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.boxplot(deltas, positions=ages, whis=[2.5, 97.5], showfliers=False)
ax.axhline(0, color='k', alpha=0.4)
ax.set_ylim(-11, 22)
ax.set_xlabel('age')
ax.set_ylabel('b2+ b4*age')

fig.savefig('9_2_oxygen_semiconjugate_predictions.png')
