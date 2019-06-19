"""TODO

Relier cet algorithme à la descente stochastique 2e ordre que nous
avons implémetentée pour ReidNet (SEP).

Comprendre les conditions de fonctionnement en grande
dimension. A-t-on besoin de travailler par blocs de paramètres?

A priori sur les paramètres de 2nd ordre de la forme: log pi(theta) =
lda * theta?

Scale initial variance automatically. Should we start from small or
large variance??? Clearly, the best is to start from a good guess. For
CNN, we can use standard tensor flow random initializer as a starting
point.

Optimal parameters? 

For proxy == 'likelihood', we seem to need niter *= 100 and gamma /=
100 for similar convergence as for proxy == 'discrete_kl'.
"""
import numpy as np
import pylab as pl

from variana.dist_fit import LaplaceApproximation
from variana.gaussian import FactorGaussian
from variana.toy_dist import ExponentialPowerLaw
from variana.utils import approx_gradient, approx_hessian_diag


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
    return safe_exp(m, s) * d



class OnlineFit(object):

    def __init__(self, log_target, dim, gamma, vmax=1e5):
        self._gen_init(log_target, dim, gamma, vmax)

    def _gen_init(self, log_target, dim, gamma, vmax):
        self._dim = dim
        self._gamma = float(gamma)
        self._vmax = float(vmax)
        self._log_target = log_target
        self._logK = log_target(m)
        self._m = np.zeros(dim)
        self._v = np.ones(dim)
        self._force = self.force
        # Init ground truth parameters
        self._logZt = 0
        self._mt = 0
        self._vt = 0

    def update_fit(self, dtheta):
        prec_ratio = np.maximum(1 - SQRT_TWO * dtheta[(self._dim + 1):], self._v / self._vmax)
        self._v /= prec_ratio
        mean_diff = (self._v / prec_ratio) * dtheta[1:(self._dim + 1)]
        self._m += mean_diff
        self._logK += dtheta[0] + .5 * (np.sum(prec_ratio + mean_diff ** 2 / self._v) - dim)

    def ortho_basis(self, x):
        phi1 = (x - self._m) / np.sqrt(self._v)
        phi2 = (phi1 ** 2 - 1) / SQRT_TWO
        return np.append(1, np.concatenate((phi1, phi2)))
   
    def sample(self):
        return np.sqrt(self._v) * np.random.normal(size=self._dim) + self._m

    def log(self, x):
        return self._logK - .5 * mahalanobis(x, self._m, self._v)

    def log_fitted_factor(self, x):
        return self.log(x) + .5 * mahalanobis(x, self._m_cavity, self._v_cavity)

    def epsilon(self, x):
        return self._log_target(x) - self.log(x)
        
    def force(self, x):
        return self.epsilon(x) * self.ortho_basis(x)

    def _record(self):
        if not hasattr(self, '_rec'):
            self._rec = []
        self._rec.append(self.error())

    def ground_truth(self, logZt, mt, vt):
        self._logZt = logZt
        self._mt = mt
        self._vt = vt

    def error(self):
        return (rms(self.logZ - self._logZt),
                rms(self._m - self._mt),
                rms(self._v - self._vt))

    def update(self, nsubiter=1):
        f = self._force(self.sample())
        for i in range(1, nsubiter):
            beta = 1 / (1 + i)
            f = (1 - beta) * f + beta * self._force(self.sample())
        self.update_fit(self._gamma * f)

    def run(self, niter, record=False, **kwargs):
        for i in range(1, niter + 1):
            if not i % (niter // 10): 
                print('\rComplete: %2.0f%%' % (100 * i / niter), end='', flush=True)
            self.update(**kwargs)
            if record:
                self._record()
        print('')

    @property
    def logK(self):
        return self._logK

    @property
    def K(self):
        return np.exp(self._logK)

    @property
    def logZ(self):
        return logZ(self._logK, self._v)
    
    @property
    def Z(self):
        return np.exp(self.logZ)

    @property
    def m(self):
        return self._m

    @property
    def v(self):
        return self._v

    def disp(self, title=''):
        pl.figure()
        pl.title(title)
        pl.plot(self._rec)
        pl.legend(('Z', 'm', 'v'))
        pl.show()
    


class OnlineContextFit(OnlineFit):

    def __init__(self, log_factor, m, v, gamma, vmax=1e5, proxy='discrete_kl'):
        self._gen_init(log_factor, len(m), gamma, vmax)
        self._m_cavity = np.asarray(m, dtype=float)
        self._v_cavity = np.asarray(v, dtype=float)
        self._log_factor = log_factor
        self._log_target = None
        self._init_force(proxy)

    def _init_force(self, proxy):        
        if proxy == 'likelihood':
            self._force = self.force_likelihood
        elif proxy != 'discrete_kl':
            raise ValueError('Unknown proxy')
   
    def sample(self):
        return np.sqrt(self._v_cavity) * np.random.normal(size=self._dim) + self._m_cavity        

    def log_fitted_factor(self, x):
        return self.log(x) + .5 * mahalanobis(x, self._m_cavity, self._v_cavity)

    def rho(self):
        return np.exp(logZ(0, self._v_cavity) - logZ(self._logK, self._v))    

    def epsilon(self, x):
        return safe_diff_exp(self._log_factor(x), self.log_fitted_factor(x), self.rho())
        
    def force_likelihood(self, x):
        f = safe_exp(self._log_factor(x), self.rho()) * self.ortho_basis(x)
        f[0] -= 1
        return f

    def factor_fit(self):
        g = FactorGaussian(self._m, self._v, logK=self._logK) / FactorGaussian(self._m_cavity, self._v_cavity, logK=0)
        return g
 

        
class OnlineStarFit(OnlineFit):

    def __init__(self, log_target, dim, alpha, gamma, vmax=1e5, proxy='discrete_kl'):
        self._gen_init(log_target, dim, gamma, vmax)
        self._alpha = float(alpha)
        self._rho_base = (1 - self._alpha) ** (-self._dim / 2)
        self._log_factor = lambda x: alpha * log_target(x)
        if proxy == 'likelihood':
            self._force = self.force_likelihood
        elif proxy != 'discrete_kl':
            raise ValueError('Unknown proxy')
        
    def sample(self):
        return np.sqrt(self._v / (1 - self._alpha)) * np.random.normal(size=self._dim) + self._m
    
    def log_fitted_factor(self, x):
        return self._alpha * self.log(x)

    def rho(self):
        return np.exp(-self._alpha * self._logK) * self._rho_base
    
    def epsilon(self, x):
        return safe_diff_exp(self._log_factor(x), self.log_fitted_factor(x), self.rho())

    def force_likelihood(self, x):
        f = safe_exp(self._log_factor(x), self.rho()) * self.ortho_basis(x)
        f[0] -= 1
        return f



class OnlineStarTaylorFit(OnlineFit):

    def __init__(self, log_target, dim, alpha, vmax=1e5, epsilon=1e-5, grad=None, hess_diag=None):
        self._gen_init(log_target, dim, 1, vmax)
        self._alpha = float(alpha)
        self._epsilon = float(epsilon)
        self._log_factor = lambda x: self._alpha * log_target(x)
        if grad is None:
            self._grad_log_factor = lambda x: approx_gradient(self._log_factor, x, self._epsilon)
        else:
            self._grad_log_factor = lambda x: self._alpha * grad(x)
        if hess_diag is None:
            self._hess_diag_log_factor = lambda x: approx_hessian_diag(self._log_factor, x, self._epsilon)
        else:
            self._hess_diag_log_factor = lambda x: self._alpha * hess_diag(x)
        
    def update(self):
        a = self._log_factor(self._m)
        g = self._grad_log_factor(self._m)
        h = self._hess_diag_log_factor(self._m)
        prec_ratio = (1 - self._alpha) / self._v - h
        self._v = np.minimum(1 / prec_ratio, self._vmax)
        self._m += self._v * g
        self._logK = (1 - self._alpha) * self._logK + a + .5 * np.sum(self._v * g ** 2)
        

                         
def rms(x):
    return np.sqrt(np.sum(x ** 2))
                         

def error(x, xt, tiny=1e-10):
    ###return np.max(np.abs(x - xt) / np.abs(xt))
    return np.max(np.abs(x - xt))
   

dim = 10
beta = 1.8
K = np.random.rand()
m = 5 * (np.random.rand(dim) - .5)
s2 = 5 * (np.random.rand(dim) + 1)
###K, m, v = 1, 0, 1

# Target definition
target = ExponentialPowerLaw(m, s2, logK=np.log(K), beta=beta)

# Laplace approximation
l = LaplaceApproximation(target.log, np.zeros(dim), grad=target.grad_log, hess_diag=target.hess_diag_log)
ql = l.fit()

# Online parameters
proxy = 'discrete_kl'
###proxy = 'likelihood'
vmax = 1e2
alpha = .1 / np.sqrt(dim)
gamma_s = .01 / np.sqrt(dim)
gamma = gamma_s * ((1 - alpha) ** (dim / 2)) / alpha
niter = int(10 / gamma_s)

print('Dimension = %d, beta = %f' % (dim, beta))

"""
q0 = OnlineContextFit(target.log, np.zeros(dim), np.ones(dim), gamma_s, vmax=vmax, proxy=proxy)
q0.run(niter, record=True)
g = q0.factor_fit()
q0.disp('context')
"""

print('Simple fit: gamma = %f' % gamma_s)
q = OnlineFit(target.log, dim, gamma_s, vmax=vmax)
q.ground_truth(target.logZ, target.m, target.v)
q.run(niter, record=True)
print('Error = %3.2f %3.2f %3.2f' % q.error())
q.disp('simple')

print('Star fit: alpha = %f, gamma = %f' % (alpha, gamma))
qs = OnlineStarFit(target.log, dim, alpha, gamma, vmax=vmax, proxy=proxy)
qs.ground_truth(target.logZ, target.m, target.v)
qs.run(niter, record=True)
print('Error = %3.2f %3.2f %3.2f' % qs.error())
qs.disp('star')

print('Star-Taylor fit: alpha = %f' % alpha)
qst = OnlineStarTaylorFit(target.log, dim, alpha, vmax=vmax, grad=target.grad_log, hess_diag=target.hess_diag_log)
qst.ground_truth(target.logZ, target.m, target.v)
qst.run(niter, record=True)
print('Error = %3.2f %3.2f %3.2f' % qst.error())
qst.disp('star-taylor')

