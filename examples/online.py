"""TODO

Relier cet algorithme à la descente stochastique 2e ordre que nous
avons implémetentée pour ReidNet (SEP).

Comprendre les conditions de fonctionnement en grande
dimension. A-t-on besoin de travailler par blocs de paramètres?

A priori sur les paramètres de 2nd ordre de la forme: log pi(theta) =
lda * theta?

Optimal parameters. 

For proxy == 'discrete_kl', we seem to be doing well with stepsize=0.1 and niter=1000*dim.
For proxy == 'likelihood', it seems more appropriate to use stepsize=0.001 and niter=1e5*dim.

"""

import numpy as np
from variana.dist_fit import LaplaceApproximation
from variana.gaussian import FactorGaussian

import pylab as pl


SQRT_TWO = np.sqrt(2)
HUGE_LOG = np.log(np.finfo(float).max)


def logZ(logK, v):
    dim = len(v)
    return logK + .5 * (dim * np.log(2 * np.pi) + np.sum(np.log(v)))


def mahalanobis(x, m, v):
    return np.sum((x - m) ** 2 / v)


def safe_exp(x, s):
    """
    Compute s * exp(x)
    Assume s > 0
    """
    return np.exp(np.minimum(x + np.log(s), HUGE_LOG))


def safe_diff_exp(x, y, s):
    """
    Compute s * (exp(x) - exp(y)) in a wise manner
    Assume s > 0
    """
    m = np.maximum(x, y)
    d = np.exp(x - m) - np.exp(y - m)
    ###return np.exp(np.minimum(m + np.log(s), HUGE_LOG)) * d
    return safe_exp(m, s) * d


class OnlineContextFit(object):

    def __init__(self, log_factor, m, v, gamma, vmax=1e5, proxy='discrete_kl'):
        self._gen_init(len(m), gamma, vmax, proxy)
        self._m_cavity = np.asarray(m, dtype=float)
        self._v_cavity = np.asarray(v, dtype=float)
        self._log_factor = log_factor
        self._logK = self._log_factor(self._m)

    def _gen_init(self, dim, gamma, vmax, proxy):
        self._dim = dim
        self._gamma = float(gamma)
        self._m = np.zeros(dim)
        ### TODO: scale initial variance automatically
        self._v = np.ones(dim)
        self._vmax = float(vmax)
        self._proxy = str(proxy)
        if proxy == 'discrete_kl':
            self.force = self._force
        elif proxy == 'likelihood':
            self.force = self._force_likelihood
        else:
            raise ValueError('Unknown KL divergence proxy')

    def update(self, dtheta):
        prec_ratio = np.maximum(1 - SQRT_TWO * dtheta[(self._dim + 1):], self._v / self._vmax)        
        """
        prec_ratio = 1 - SQRT_TWO * dtheta[(self._dim + 1):]
        if np.min(prec_ratio - self._v / self._vmax) < 0:
            return False
        """
        self._v /= prec_ratio
        mean_diff = (self._v / prec_ratio) * dtheta[1:(self._dim + 1)]
        self._m += mean_diff
        self._logK += dtheta[0] + .5 * (np.sum(prec_ratio + mean_diff ** 2 / self._v) - dim)

    def ortho_basis(self, x):
        phi1 = (x - self._m) / np.sqrt(self._v)
        phi2 = (phi1 ** 2 - 1) / SQRT_TWO
        return np.append(1, np.concatenate((phi1, phi2)))

    """
    def update(self, dtheta):
        th0 = self._logK - .5 * np.sum(self._m ** 2 / self._v) + dtheta[0]
        th1 = self._m / self._v + dtheta[1:(dim + 1)]
        th2 = np.minimum(-.5 / self._v + dtheta[(self._dim + 1):], -.5 / self._vmax)
        self._v = - .5 / th2
        self._m = self._v * th1
        self._logK = th0 + .5 * np.sum(self._m ** 2 / self._v)
        
    def ortho_basis(self, x):
        xc = (x - self._m) / np.sqrt(self._v)
        th2 = (xc ** 2 - 1) / (2 * self._v)
        th1 = xc / np.sqrt(self._v) - 2 * self._m * th2
        th0 = 1 - np.sum(self._m * th1 + (self._m ** 2 + self._v) * th2)
        return np.append(th0, np.concatenate((th1, th2)))
    """
    
    def sample(self):
        return np.sqrt(self._v_cavity) * np.random.normal(size=self._dim) + self._m_cavity        

    def log(self, x):
        return self._logK - .5 * mahalanobis(x, self._m, self._v)
    
    def log_fitted_factor(self, x):
        return self.log(x) + .5 * mahalanobis(x, self._m_cavity, self._v_cavity)

    def rho(self):
        return np.exp(logZ(0, self._v_cavity) - logZ(self._logK, self._v))    

    def delta(self, x):
        return safe_diff_exp(self._log_factor(x), self.log_fitted_factor(x), self.rho())
        
    def _force(self, x):
        # Get the optimal descent direction in fixed coordinates
        return self.delta(x) * self.ortho_basis(x)

    def _force_likelihood(self, x):
        f = safe_exp(self._log_factor(x), self.rho()) * self.ortho_basis(x)
        f[0] -= 1
        return f    

    def _record(self):
        if not hasattr(self, '_rec'):
            self._rec = []
        self._rec.append(self._logK)
    
    def run(self, niter, nsubiter=1):
        for j in range(niter):
            print('Iteration: %d' % j)
            x = self.sample()
            f = self.force(x)
            for i in range(1, nsubiter):
                beta = 1 / (1 + i)
                f = (1 - beta) * f + beta * self.force(self.sample())
            self.update(self._gamma * f)
            self._record()

    def factor_fit(self):
        from variana.gaussian import FactorGaussian
        g = FactorGaussian(self._m, self._v, logK=self._logK) / FactorGaussian(self._m_cavity, self._v_cavity, logK=0)
        return g
        
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


    
class OnlineStarFit(OnlineContextFit):

    def __init__(self, log_target, dim, alpha, gamma, vmax=1e5, proxy='discrete_kl'):
        self._gen_init(dim, gamma, vmax, proxy)
        self._alpha = float(alpha)
        self._rho_base = (1 - self._alpha) ** (-self._dim / 2)
        self._log_factor = lambda x: alpha * log_target(x)
        self._logK = log_target(m)
        
    def sample(self):
        return np.sqrt(self._v / (1 - self._alpha)) * np.random.normal(size=self._dim) + self._m
    
    def log_fitted_factor(self, x):
        return self._alpha * self.log(x)

    def rho(self):
        return np.exp(-self._alpha * self._logK) * self._rho_base

    def _record(self):
        if not hasattr(self, '_rec'):
            self._rec = []
        self._rec.append((self._logK, rms(self._m), rms(self._v)))

        
        
class OnlineFit(OnlineContextFit):

    def __init__(self, log_target, dim, gamma, vmax=1e5):
        self._gen_init(dim, gamma, vmax, 'discrete_kl')
        self._log_target = log_target
        self._logK = log_target(m)

    def sample(self):
        return np.sqrt(self._v) * np.random.normal(size=self._dim) + self._m

    def delta(self, x):
        return self._log_target(x) - self.log(x)

    def _record(self, *args):
        if not hasattr(self, '_rec'):
            self._rec = []
        ###self._rec.append((error(self._logK, args[0]), error(self._m, args[1]), error(self._v, args[2])))
        self._rec.append((self._logK, rms(self._m), rms(self._v)))

                         
def rms(x):
    return np.sqrt(np.sum(x ** 2))
                         
        
def toy_score(x, m=0, v=1, K=1, power=2, proper=True):
    x = (x - m) / np.sqrt(v)
    return np.log(K) - np.sum(((2 * proper - 1) ** np.arange(len(x))) * np.abs(x) ** power, 0) / power


def error(x, xt, tiny=1e-10):
    ###return np.max(np.abs(x - xt) / np.abs(xt))
    return np.max(np.abs(x - xt))
   



dim = 10
power = 2
K = np.random.rand()
m = 5 * (np.random.rand(dim) - .5)
v = 5 * (np.random.rand(dim) + 1)
###K, m, v = 1, 0, 1

# Laplace approximation
log_target = lambda x: toy_score(x, m, v, K, power)
l = LaplaceApproximation(log_target, np.zeros(dim))
ql = l.fit()

# Online parameters
proxy = 'discrete_kl' ###proxy = 'likelihood'
vmax = 1e2
gamma0 = 0.01 / np.sqrt(dim)
alpha = .7 ###/ dim 
gamma = gamma0 * ((1 - alpha) ** (dim / 2)) / alpha
niter = int(10 / gamma0)

if proxy == 'likelihood':
    niter *= 100
    gamma /= 100    

print('alpha = %f, gamma = %f' % (alpha, gamma))


"""
q = OnlineContextFit(log_target, np.zeros(dim), np.ones(dim), gamma, vmax=vmax, proxy=proxy)
q.run(niter)
g = q.factor_fit()
pl.figure()
pl.plot(q._rec)
pl.show()
"""

q = OnlineStarFit(log_target, dim, alpha, gamma, vmax=vmax, proxy=proxy)
q.run(niter)
pl.figure()
pl.plot(q._rec)
pl.legend(('K', 'm', 'v'))
pl.show()

qc = OnlineFit(log_target, dim, gamma0, vmax=vmax)
### Rule of thumb: set the simple online fit step size as apha times
### the star fit step size
qc.run(niter)
pl.figure()
pl.plot(qc._rec)
pl.legend(('K', 'm', 'v'))
pl.show()

