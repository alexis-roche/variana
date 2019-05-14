import numpy as np
from variana.dist_fit import VariationalSampling

DIM = 1
BETA = 2
PROPER = True


def toy_dist(x, c=None, K=1, beta=BETA, proper=PROPER):
    """
    Function that takes an array with shape (dim, n) as input and
    returns an array with shape (n,) that contains the corresponding
    target log-distribution values.
    """
    if c is None:
        c = np.zeros(DIM)
    return np.log(K) - np.sum(((2 * proper -1) ** np.arange(DIM)) * np.abs(x - c) ** beta, 0) / beta


""" 
Tune the mean and variance of the cavity distribution. If we use
as vector as the variance, it will be understood as a diagonal matrix.
"""
m = np.zeros(DIM)
v = np.ones(DIM)

K = np.random.rand()
###c = np.random.rand(DIM)
c = None
target = lambda x: toy_dist(x, c, K=K)

vs = VariationalSampling(target, (m, v))
g = vs.fit(vmax=1e6)
#g = vs.fit(family='gaussian')
print(g)


"""
print('Error on K = %f' % np.abs(1 - g.K))
print('Error on m = %f' % np.max(np.abs(c - g.m)))

from variana.gaussian import FactorGaussian
vs2 = VariationalSampling(target, FactorGaussian(m, v, K=2))
g2 = vs2.fit(vmax=1e6, global_fit=True)
"""
