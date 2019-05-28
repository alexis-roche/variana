import numpy as np
from variana.dist_fit import VariationalSampling
from variana.gaussian import FactorGaussian

DIM = 10
BETA = 2
PROPER = True


def toy_dist(x, c=0, s=1, K=1, beta=BETA, proper=PROPER):
    """
    Function that takes an array with shape (dim, n) as input and
    returns an array with shape (n,) that contains the corresponding
    target log-distribution values.
    """
    return np.log(K) - np.sum(((2 * proper -1) ** np.arange(DIM)) * np.abs((x - c) / s) ** beta, 0) / beta


""" 
Tune the mean and variance of the cavity distribution. If we use
as vector as the variance, it will be understood as a diagonal matrix.
"""
K = np.random.rand()
c = np.random.rand(DIM) - .5
s = 1 + np.random.rand(DIM)
#K, c, s = 1, 0, 1
#c, s = 0, 1

kernel = FactorGaussian(np.zeros(DIM), np.ones(DIM))
log_target = lambda x: toy_dist(x, c, s, K=K) - kernel.log(x)
vs = VariationalSampling(log_target, kernel, ndraws=10*DIM)
q = vs.fit(vmax=1e6, optimizer='newton', hess_diag_approx=True)
###q = vs.fit(family='gaussian')
print(q)

print(vs._fit._info)

rel_err = lambda x, y: np.max(np.abs(y - x)) / np.maximum(1, np.max(np.abs(x)))
print('Order-0 error = %f' % rel_err(K, q.K))
print('Order-1 error = %f' % rel_err(c, q.m))
print('Order-2 error = %f' % rel_err(s ** 2, q.v))

q2 = vs.fit(vmax=1e6, optimizer='newton', hess_diag_approx=False)

print(q2)
print('Order-0 dev = %f' % rel_err(q2.K, q.K))
print('Order-1 dev = %f' % rel_err(q2.m, q.m))
print('Order-2 dev = %f' % rel_err(q2.v, q.v))

print(vs._fit._info)
