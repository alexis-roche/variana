import numpy as np
from variana.distfit import VariationalSampler

DIM = 10
BETA = 2


def target(x, beta=BETA):
    """
    Function that takes an array with shape (dim, n) as input and
    returns an array with shape (n,) that contains the corresponding
    target log-distribution values.
    """
    #return -np.sum(np.abs(x) ** beta, 0) / beta
    return -np.sum(((-1) ** np.arange(DIM)) * np.abs(x) ** beta, 0) / beta


def _make_bounds(stdev_max, cavity):
    dim = cavity.dim
    theta_max = np.full(dim, -.5 * stdev_max ** -2) - cavity.theta[(dim + 1):]
    bounds = [(None, None) for i in range(dim + 1)]
    bounds += [(None, theta_max[i]) for i in range(dim)]
    return bounds


def kl_fit(target, cavity, factorize=True, ndraws=None, global_fit=False, method='kullback', minimizer='lbfgs', stdev_max=None):
    vs = VariationalSampler(target, cavity, ndraws=ndraws)
    if factorize:
        family = 'factor_gaussian'
    else:
        family = 'gaussian'
    bounds = None
    if not stdev_max is None:
        if vs._cavity._family == 'factor_gaussian':
            bounds = _make_bounds(stdev_max, vs._cavity)
        else:
            print('Warning: ignoring bounds because cavity is not factorial')
    return vs.fit(family=family, global_fit=global_fit, method=method, minimizer=minimizer, bounds=bounds)



"""
Tune the mean and variance of the cavity distribution. If we use
as vector as the variance, it will be understood as a diagonal matrix.
"""
m = np.zeros(DIM)
v = np.ones(DIM)
cavity = (m, v)

#g = kl_fit(target, cavity, minimizer='lbfgs', stdev_max=1e2)


vs = VariationalSampler(target, cavity)
bounds = _make_bounds(1e2, vs._cavity)
g = vs.fit(family='factor_gaussian', minimizer='lbfgs', bounds=bounds)


"""
Get the adjusted normalization constant, mean and variance.
"""
print(g)
#print('Estimated factor: %f' % g.K)
#print('Estimated mean: %s' % g.m)
#print('Estimated variance (diagonal): %s' % g.v)
