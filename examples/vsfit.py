import numpy as np
from variana import VariationalSampler

DIM = 100


def target(x, beta=2):
    """
    Function that takes an array with shape (dim, n) as input and
    returns an array with shape (n,) that contains the corresponding
    target log-distribution values.
    """
    return -np.sum(np.abs(x) ** beta, 0) / beta
    ###return -np.sum(((-1) ** np.arange(DIM)) * np.abs(x) ** beta, 0) / beta


"""
Tune the mean and variance of the sampling kernel. If we use as vector
as the variance, it will be understood as a diagonal matrix.
"""
m = np.zeros(DIM)
v = np.ones(DIM)

"""
Create a variational sampler object.
"""
#vs = VariationalSampler(target, (m, v), ndraws = 100 * DIM)
vs = VariationalSampler(target, (m, v))

"""
Perform fitting.
"""
f = vs.fit(family='factor_gaussian', minimizer='lbfgs')
#f = vs.fit(family='factor_gaussian')
#f = vs.fit()

"""
Get the adjusted normalization constant, mean and variance.
"""
print('Estimated normalizing constant: %f' % f.gaussian.Z)
print('Estimated mean: %s' % f.gaussian.m)
print('Estimated variance (diagonal): %s' % f.gaussian.v)
