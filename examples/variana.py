import numpy as np
from variana.dist_fit import Variana


def toy_score(x, center=None, K=1, beta=2, proper=True):
    if not center is None:
        x = x - center
    return np.log(K) - np.sum(((2 * proper - 1) ** np.arange(len(x))) * np.abs(x) ** beta, 0) / beta


               
dim = 10
vmax = 1e4
alpha = 0.1
beta = 1.2
stride = 3
niter = 100

#K = 1
K = np.random.rand()

#c = np.zeros(dim)
c = 5 * (np.random.rand(dim) - .5)

target = lambda x: toy_score(x, c, K, beta)

v = Variana(target, dim, alpha, vmax, stride)
fit = v.run(niter=niter)

print('True K = %f' % K)
print('Estimated K = %f' % fit.K)
print('Estimate / true = %f' % (fit.K / K))

print('Error on mean = %f' % np.max(np.abs(fit.m - c)))
###print('Error on variance = %f' % np.max(np.abs(fit.v - 1)))
