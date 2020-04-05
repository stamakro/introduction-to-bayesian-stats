import numpy as np
from scipy.stats import invgamma, multivariate_normal, gaussian_kde, norm
import matplotlib.pyplot as plt


def inverse(X):
    m, n = X.shape
    assert m == n

    return np.linalg.solve(X, np.eye(m))

np.random.seed(18)

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


g = float(n)
nu0 = 1
sigma0 = sigma2_ols

E_b = (g / (g + 1)) * b_ols

N = 10000

s_post = np.zeros(N)
b_post = np.zeros((N, p))

gmd = np.eye(n) - (g * X.dot(inverse(X.T.dot(X))).dot(X.T)) / (g+1)
ssr = y.T.dot(gmd).dot(y)
#for i in xrange(N):
sigma2_post = invgamma.rvs(0.5 * (nu0 + n), scale=0.5 * (nu0 * sigma0 + ssr), size=N)
vv = g * inverse(X.T.dot(X)) / (g+1)

VV = np.zeros((N, p, p))
bb = np.zeros((N, p))
for i, ss in enumerate(sigma2_post):
    VV[i] = vv * ss
    bb[i] = np.random.multivariate_normal(E_b, VV[i])


print np.mean(bb, 0)
print np.std(bb, axis=0, ddof=1)

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
d1 = gaussian_kde(bb[:,1])

#xx = np.linspace(1.1* np.min(bb[:,1]), 1.1*np.max(bb[:,1]), 300)
xx = np.linspace(-110, 110, 500)

priorstd = np.sqrt(np.diag(inverse(X.T.dot(X)) * g * sigma2_ols))
ax.hist(bb[:,1], bins=40, color='k', alpha=0.3, density=True, edgecolor='k')
ax.plot(xx, d1(xx), color='C0', label='posterior')
ax.plot(xx, norm.pdf(xx, loc=0, scale=priorstd[1]), color='C1', label='prior')
ax.set_title('b2')
ax.legend()

ax = fig.add_subplot(1,2,2)
d2 = gaussian_kde(bb[:,3])
#xx = np.linspace(1.1* np.min(bb[:,3]), 1.1*np.max(bb[:,3]), 300)
xx = np.linspace(-4, 4, 500)

ax.hist(bb[:,3], bins=40, color='k', alpha=0.3, density=True, edgecolor='k')
ax.plot(xx, d2(xx), color='C0', label='posterior')
ax.plot(xx, norm.pdf(xx, loc=0, scale=priorstd[3]), color='C1', label='prior')
ax.set_title('b4')
ax.legend()


ages = np.arange(20, 32)
deltas = np.zeros((N, ages.shape[0]))
for i, (b2, b4) in enumerate(zip(bb[:,1], bb[:,3])):
    deltas[i] = b2 + b4 * ages
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.boxplot(deltas, positions=ages, whis=[2.5, 97.5], showfliers=False)
ax.axhline(0, color='k', alpha=0.4)
ax.set_ylim(-11, 17)
ax.set_xlabel('age')
ax.set_ylabel('b2+ b4*age')

plt.show()
