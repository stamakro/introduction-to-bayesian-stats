import pickle
import numpy as np
import matplotlib.pyplot as plt

def printStats(x, name):
    print(name)
    print('posterior mean: %f' % np.mean(x))
    print('posterior quantiles:')
    print('2.5\t50\t97.5')
    print('%.3f\t%.3f\t%.3f' % tuple(np.percentile(x, [2.5, 50, 97.5])))
    print('')



np.random.seed(17021991)

with open('trace.pkl', 'rb') as f:
    trace = pickle.load(f)


Nsamples = trace['mu'].shape[0]

SS = np.zeros((Nsamples, 2, 2))
rho = np.zeros((Nsamples,))

Yrand = np.zeros((Nsamples, 2))

for i, (m, v) in enumerate(zip(trace['mu'], trace['chol_cov'])):

    cc = np.array([[v[0], 0], [v[1], v[2]]])
    SS[i] = cc.dot(cc.T)

    rho[i] = SS[i, 0,1] / np.sqrt(SS[i, 0,0]) / np.sqrt(SS[i, 1, 1])

    Yrand[i] = np.random.multivariate_normal(m, SS[i], 1)

printStats(trace['mu'][:, 0], 'mu1')
printStats(trace['mu'][:, 1], 'mu2')
printStats(np.sqrt(SS[:, 0,0]), 'sigma1')
printStats(np.sqrt(SS[:, 1,1]), 'sigma2')
printStats(rho, 'correlation')


print('P(m1 < m2) = %.4f' % np.mean(trace['mu'][:, 0] < trace['mu'][:, 1]))
print('P(Y1 < Y2) = %.4f' % np.mean(Yrand[:, 0] < Yrand[:, 1]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Yrand[:,0], Yrand[:,1], color='C0', edgecolor='k')
m = np.min(Yrand)* 0.8
M = np.max(Yrand)* 1,2

xx = np.linspace(m, M, 100)
ax.plot(xx, xx, color='k', linestyle='--')
ax.set_xlabel('Test performance before')
ax.set_ylabel('Test performance after')
ax.set_title('Samples drawn from posterior')
#plt.show()
fig.savefig('posteriorsamples.png')
