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


def split_max(a, b):
    """
    Compute m = max(a, b) and returns a tuple (m, a - m, b - m)
    """
    if a > b:
        return a, 0.0, b - a
    else:
        return b, a - b, 0.0


def safe_exp(x, log_s):
    """
    Compute s * exp(x)
    """
    return np.exp(np.minimum(x + log_s, HUGE_LOG))


def safe_diff_exp(x, y, log_s):
    """
    Compute s * (exp(x) - exp(y))
    """
    """
    m = np.maximum(x, y)
    d = np.exp(x - m) - np.exp(y - m)
    """
    m, xc, yc = split_max(x, y)
    d = np.exp(xc) - np.exp(yc)
    return safe_exp(m, log_s) * d



class OnlineFit(object):

    def __init__(self, log_target, dim, gamma, vmax=1e5, logK=None, m=None, v=None):
        self._gen_init(log_target, dim, vmax, logK, m, v)
        self.reset(gamma)

    def _gen_init(self, log_target, dim, vmax, logK, m, v):
        self._dim = dim
        self._vmax = float(vmax)
        self._log_target = log_target
        if m is None:
            self._m = np.zeros(dim)
        else:
            self._m = np.asarray(m)
        if v is None:
            self._v = np.ones(dim)
        else:
            self._v = np.asarray(v)
        if logK is None:
            self._logK = log_target(self._m)
        else:
            self._logK = float(logK)        
        self._force = self.force
        # Init ground truth parameters
        self._logZt = 0
        self._mt = 0
        self._vt = 0
        
    def reset(self, gamma):
        self._gamma = float(gamma)
        
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

    def __init__(self, log_factor, m, v, gamma, vmax=1e5, proxy='discrete_kl', logK=None):
        self._gen_init(log_factor, len(m), vmax, logK, m, v)
        self._m_cavity = np.asarray(m, dtype=float).copy()
        self._v_cavity = np.asarray(v, dtype=float).copy()
        self._log_factor = log_factor
        self._log_target = None
        if proxy == 'likelihood':
            self._force = self.force_likelihood
        elif proxy != 'discrete_kl':
            raise ValueError('Unknown proxy')
        self.reset(gamma)

    def sample(self):
        return np.sqrt(self._v_cavity) * np.random.normal(size=self._dim) + self._m_cavity        

    def log_fitted_factor(self, x):
        return self.log(x) + .5 * mahalanobis(x, self._m_cavity, self._v_cavity)

    def log_rho(self):
        return logZ(0, self._v_cavity) - logZ(self._logK, self._v)

    def epsilon(self, x):
        return safe_diff_exp(self._log_factor(x), self.log_fitted_factor(x), self.log_rho())
        
    def force_likelihood(self, x):
        f = safe_exp(self._log_factor(x), self.log_rho()) * self.ortho_basis(x)
        f[0] -= 1
        return f

    def factor_fit(self):
        g = FactorGaussian(self._m, self._v, logK=self._logK) / FactorGaussian(self._m_cavity, self._v_cavity, logK=0)
        return g

    def stepsisze(self):
        return self._gamma


    
class OnlineStarFit(OnlineFit):

    def __init__(self, log_target, dim, alpha, gamma, vmax=1e5, proxy='discrete_kl', logK=None, m=None, v=None):
        self._gen_init(log_target, dim, vmax, logK, m, v)
        if proxy == 'likelihood':
            self._force = self.force_likelihood
        elif proxy != 'discrete_kl':
            raise ValueError('Unknown proxy')
        self.reset(alpha=alpha, gamma=gamma)
        
    def reset(self, alpha, gamma=None):
        self._alpha = float(alpha)
        self._log_factor = lambda x: self._alpha * self._log_target(x)
        if not gamma is None:
            self._gamma = float(gamma)

    def sample(self):
        return np.sqrt(self._v / (1 - self._alpha)) * np.random.normal(size=self._dim) + self._m
    
    def log_fitted_factor(self, x):
        return self._alpha * self.log(x)

    def epsilon(self, x):
        return safe_diff_exp(self._log_factor(x), self.log_fitted_factor(x), -self._alpha * self._logK)

    def force_likelihood(self, x):
        f = safe_exp(self._log_factor(x), -self._alpha * self._logK) * self.ortho_basis(x)
        f[0] -= 1
        return f

    def stepsize(self):
        return self._gamma * (1 - self._alpha) ** (self._dim / 2)


    
class OnlineKLFit(OnlineFit):

    def __init__(self, log_target, dim, gamma, lda=1, vmax=1e5, proxy='discrete_kl', logK=None, m=None, v=None):
        self._gen_init(log_target, dim, vmax, logK, m, v)
        if proxy == 'likelihood':
            self._force = self.force_likelihood
        elif proxy != 'discrete_kl':
            raise ValueError('Unknown proxy')
        self.reset(lda, gamma)
        
    def reset(self, lda, gamma=None):
        self._lda = float(lda)
        if not gamma is None:
            self._gamma = float(gamma)
        
    def sample(self):
        return np.sqrt(self._v / self._lda) * np.random.normal(size=self._dim) + self._m
    
    def epsilon(self, x):
        log_p = self._log_target(x)
        log_q = self.log(x)
        zob1 = -self._lda * log_q
        zob2 = log_q + zob1
        return safe_diff_exp(log_p + zob1, zob2, 0)
    
    def force_likelihood(self, x):
        raise NotImplementedError('This is a test piece of code.')



class OnlineKLFit0(OnlineFit):

    def __init__(self, log_target, dim, gamma, lda=0, vmax=1e5, proxy='discrete_kl', logK=None, m=None, v=None):
        self._gen_init(log_target, dim, vmax, logK, m, v)
        if proxy == 'likelihood':
            self._force = self.force_likelihood
        elif proxy != 'discrete_kl':
            raise ValueError('Unknown proxy')
        self.reset(lda, gamma)
        
    def log_pi(self, x):
        return -.5 * (self._dim * np.log(2 * np.pi * self._vmax) + mahalanobis(x, 0, self._vmax))

    def reset(self, lda, gamma=None):
        self._lda = float(lda)
        if not gamma is None:
            self._gamma = float(gamma)
        
    def sample(self):
        if np.random.rand() > (1 - self._lda):
            return np.sqrt(self._vmax) * np.random.normal(size=self._dim)
        return np.sqrt(self._v) * np.random.normal(size=self._dim) + self._m
    
    def epsilon(self, x):
        log_p = self._log_target(x)
        log_q = self.log(x)
        log_q0 = self.logZ + self.log_pi(x)
        max1, log_pc, log_qc = split_max(log_p, log_q)
        num = np.exp(log_pc) - np.exp(log_qc)
        max2, log_qc, log_q0c = split_max(log_q, log_q0)
        den = (1 - self._lda) * np.exp(log_qc) + self._lda * np.exp(log_q0c)
        return np.exp(max1 - max2) * (num / den)

    def force_likelihood(self, x):
        raise NotImplentedError('This is a test piece of code.')

    

class OnlineStarTaylorFit(OnlineFit):

    def __init__(self, log_target, dim, alpha, vmax=1e5, epsilon=1e-5, grad=None, hess_diag=None,
                 logK=None, m=None, v=None):
        self._gen_init(log_target, dim, vmax, logK, m, v)
        self._epsilon = float(epsilon)
        self.reset(alpha, grad=grad, hess_diag=hess_diag)

    def reset(self, alpha, grad=None, hess_diag=None):
        self._alpha = float(alpha)
        self._log_factor = lambda x: self._alpha * self._log_target(x)
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
beta = 1
vmax = 1e2
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
gamma_s = .01 / np.sqrt(dim)
###alpha = .1 / np.sqrt(dim)
###gamma = gamma_s / alpha
alpha, gamma = .1, .1
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

"""
print('Star fit: alpha = %f, gamma = %f' % (alpha, gamma))
qs = OnlineStarFit(target.log, dim, alpha, gamma, vmax=vmax, proxy=proxy)
qs.ground_truth(target.logZ, target.m, target.v)
qs.run(niter, record=True)
###qs.reset(alpha=.5)
###print('Star fit: alpha = %f, gamma = %f' % (qs._alpha, qs._gamma))
###qs.run(niter, record=True)
print('Error = %3.2f %3.2f %3.2f' % qs.error())
qs.disp('star')

print('Star-Taylor fit: alpha = %f' % alpha)
qst = OnlineStarTaylorFit(target.log, dim, alpha, vmax=vmax, grad=target.grad_log, hess_diag=target.hess_diag_log)
qst.ground_truth(target.logZ, target.m, target.v)
qst.run(niter, record=True)
print('Error = %3.2f %3.2f %3.2f' % qst.error())
qst.disp('star-taylor')
"""

"""
gamma1 = 1e-3 / dim
log_target = lambda x: target.log(x) - .5 * np.sum(x ** 2) / vmax
q1 = OnlineKLFit(log_target, dim, gamma1, lda=.5, logK=q.logK, m=q.m, v=q.v)
"""

q1 = OnlineKLFit(target.log, dim, gamma_s, lda=.75, logK=q.logK, m=q.m, v=q.v)
q1.ground_truth(target.logZ, target.m, target.v)
q1.run(10 * niter, nsubiter=1, record=True)
print('Error = %3.2f %3.2f %3.2f' % q1.error())
q1.disp('online KL')
###pl.plot(pl.axis()[-1] * np.array(q1._zob) / np.max(q1._zob), ':')


r"""
q = FactorGaussian(target.m, target.v, logZ=target.logZ)
N = 100000
X = q.random(N)
log_p = np.array([target.log(x) for x in X.T])
log_q = q.log(X)
log_r = log_p - log_q
z = np.exp(log_r) - 1

pl.figure()
pl.hist(z)
pl.show()
"""
