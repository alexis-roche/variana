import numpy as np
from variana.dist_fit import LaplaceApproximation
from variana.gaussian import FactorGaussian


SQRT_TWO = np.sqrt(2)


def logZ(logK, v):
    dim = len(v)
    return logK + .5 * (dim * np.log(2 * np.pi) + np.sum(np.log(v)))


def mahalanobis(x, m, v):
    return np.sum((x - m) ** 2 / v)


class OnlineFit(object):

    def __init__(self, log_factor, m, v, vmax=1e5):
        self._log_factor = log_factor
        self._logK = log_target(m)
        self._m = np.asarray(m, dtype=float)
        self._v = np.asarray(v, dtype=float)
        self._logK = self._log_factor(self._m)
        self._m_cavity = self._m.copy()
        self._v_cavity = self._v.copy()
        self._vmax = float(vmax)
        self._dim = len(m)

    def ortho_basis(self, x):
        phi1 = (x - self._m) / np.sqrt(self._v)
        phi2 = (phi1 ** 2 - 1) / SQRT_TWO
        return np.append(1, np.concatenate((phi1, phi2)))

    def update(self, theta, tiny=1e-50):
        inv_var_ratio = np.maximum(1 - SQRT_TWO * theta[(self._dim + 1):], tiny)
        self._v /= inv_var_ratio
        np.minimum(self._v, self._vmax, out=self._v)
        mean_diff = (self._v / inv_var_ratio) * theta[1:(self._dim + 1)]
        self._m += mean_diff
        self._logK += theta[0] + .5 * (np.sum(inv_var_ratio + mean_diff ** 2 / self._v) - dim)

    def sample(self):
        return np.sqrt(self._v_cavity) * np.random.normal(size=self._dim) + self._m_cavity        

    def log(self, x):
        return self._logK - .5 * mahalanobis(x, self._m, self._v)
    
    def log_fitted_factor(self, x):
        return self.log(x) + .5 * mahalanobis(x, self._m_cavity, self._v_cavity)

    def get_force(self, x):
        rho = np.exp(logZ(0, self._v_cavity) - logZ(self._logK, self._v))
        delta = np.exp(self.log_fitted_factor(x)) - np.exp(self._log_factor(x))
        return rho * delta * self.ortho_basis(x)
        
    @property
    def logK(self):
        return self._logK

    @property
    def K(self):
        return np.exp(self._logK)

    @property
    def Z(self):
        return np.exp(logZ(self._logK, self._v))

    @property
    def m(self):
        return self._m

    @property
    def v(self):
        return self._v



class OnlineStarFit(OnlineFit):

    def __init__(self, log_target, m, v, vmax=1e5, alpha=0.1):
        self._logK = log_target(m)
        self._m = np.asarray(m, dtype=float)
        self._v = np.asarray(v, dtype=float)
        self._vmax = float(vmax)
        self._alpha = float(alpha)
        self._dim = len(m)
        self._rho_base = (1 - self._alpha) ** (-.5 * self._dim)
        self._log_factor = lambda x: alpha * log_target(x)

    def sample(self):
        return np.sqrt(self._v / (1 - self._alpha)) * np.random.normal(size=self._dim) + self._m        
    
    def log_fitted_factor(self, x):
        return self._alpha * self.log(x)

    def get_force(self, x):
        rho = np.exp(-self._alpha * self._logK) * self._rho_base
        delta = np.exp(self.log_fitted_factor(x)) - np.exp(self._log_factor(x))
        return rho * delta * self.ortho_basis(x)
        

    

class FuckOnlineStarFit(object):

    def __init__(self, log_target, m, v, vmax=1e5, alpha=0.1):
        self._logK = log_target(m)
        self._m = np.asarray(m, dtype=float)
        self._v = np.asarray(v, dtype=float)
        self._vmax = float(vmax)
        self._alpha = float(alpha)
        self._dim = len(m)
        self._rho_base = (1 - self._alpha) ** (-.5 * self._dim)
        self._log_factor = lambda x: alpha * log_target(x)

    def ortho_basis(self, x):
        phi1 = (x - self._m) / np.sqrt(self._v)
        phi2 = (phi1 ** 2 - 1) / SQRT_TWO
        return np.append(1, np.concatenate((phi1, phi2)))

    def update(self, theta, tiny=1e-50):
        inv_var_ratio = np.maximum(1 - SQRT_TWO * theta[(self._dim + 1):], tiny)
        self._v /= inv_var_ratio
        np.minimum(self._v, self._vmax, out=self._v)
        mean_diff = (self._v / inv_var_ratio) * theta[1:(self._dim + 1)]
        self._m += mean_diff
        self._logK += theta[0] + .5 * (np.sum(inv_var_ratio + mean_diff ** 2 / self._v) - dim)

    def sample(self):
        return np.sqrt(self._v / (1 - self._alpha)) * np.random.normal(size=self._dim) + self._m        

    def log(self, x):
        return self._logK - .5 * np.sum((x - self._m) ** 2 / self._v)
    
    def log_fitted_factor(self, x):
        return self._alpha * self.log(x)

    def get_force(self, x):
        rho = np.exp(-self._alpha * self._logK) * self._rho_base
        delta = np.exp(self.log_fitted_factor(x)) - np.exp(self._log_factor(x))
        return rho * delta * self.ortho_basis(x)
        
    @property
    def logK(self):
        return self._logK

    @property
    def K(self):
        return np.exp(self._logK)

    @property
    def Z(self):
        return np.exp(self._logK + .5 * (self._dim * np.log(2 * np.pi) + np.sum(np.log(self._v))))

    @property
    def m(self):
        return self._m

    @property
    def v(self):
        return self._v



    

def toy_score(x, m=0, v=1, K=1, power=2, proper=True):
    x = (x - m) / np.sqrt(v)
    return np.log(K) - np.sum(((2 * proper - 1) ** np.arange(len(x))) * np.abs(x) ** power, 0) / power


def error(x, xt, tiny=1e-10):
    ###return np.max(np.abs(x - xt) / np.abs(xt))
    return np.max(np.abs(x - xt))
    

dim = 25
power = 2


"""
K = 1
m = np.zeros(dim)
v = np.ones(dim)
"""
K = np.random.rand()
m = 5 * (np.random.rand(dim) - .5)
v = 5 * (np.random.rand(dim) + 1)
##v = 1

# Laplace approximation
log_target = lambda x: toy_score(x, m, v, K, power)
l = LaplaceApproximation(log_target, np.zeros(dim))
ql = l.fit()

# Online parameters
alpha = 0.1
beta = 1 / (100 * dim)
vmax = 1e4
#niter = 10000
#nsubiter = 1
#gamma =  .01
m0 = np.zeros(dim)
v0 = np.full(dim, 1)

tniter = 1000 * dim
nsubiter = 1
niter = tniter // nsubiter
gamma = min(.1 * nsubiter / dim, 1)

"""
q = OnlineFit(log_target, m0, v0, vmax=vmax)
f = np.zeros(2 * dim + 1)
rec = []
for j in range(niter):
    print('Iteration: %d' % j)
    # Loop to minimize D(wf||wg)
    for i in range(nsubiter):
        x = q.sample()
        f = (1 - beta) * f + beta * q.get_force(x)
    rec.append(q.logK)
    #rec.append((error(q.K, K), error(q.m, m), error(q.v, v)))
    q.update(-gamma * f)
rec = np.array(rec)

### HACK for now
from variana.gaussian import FactorGaussian
g = FactorGaussian(q.m, q.v, logK=q.logK) / FactorGaussian(m0, v0, logK=0)
"""

q = OnlineStarFit(log_target, m0, v0, vmax=vmax)
f = np.zeros(2 * dim + 1)
rec = []
for j in range(niter):
    print('Iteration: %d' % j)
    # Loop to minimize D(wf||wg)
    for i in range(nsubiter):
        x = q.sample()
        f = (1 - beta) * f + beta * q.get_force(x)
    #rec.append(q.logK)
    rec.append((error(q.K, K), error(q.m, m), error(q.v, v)))
    q.update(-gamma * f)
rec = np.array(rec)

import pylab as pl
pl.plot(rec)
pl.legend(('K', 'm', 'v'))
pl.show()


