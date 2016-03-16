"""
Variational sampling
"""
from time import time
import numpy as np

from .utils import safe_exp
from .gaussian import Gaussian, FactorGaussian
from .fit import KLFit, LFit


def reflect_sample(xs, m):
    return np.reshape(np.array([xs.T, m - xs.T]).T,
                      (xs.shape[0], 2 * xs.shape[1]))


def as_normalized_gaussian(g):
    """
    renormalize input to unit integral
    """
    if isinstance(g, Gaussian):
        return Gaussian(g.m, g.V)
    elif isinstance(g, FactorGaussian):
        return FactorGaussian(g.m, g.v)
    if len(g) == 2:
        m, V = np.asarray(g[0]), np.asarray(g[1])
    else:
        raise ValueError('input not understood')
    if V.ndim < 2:
        G = FactorGaussian(m, V)
    elif V.ndim == 2:
        G = Gaussian(m, V)
    else:
        raise ValueError('input variance not understood')
    return G


def sample_fun(f, x):
    try:
        return f(x).squeeze(), f
    except:
        ff = lambda x: np.array([f(xi) for xi in x.T])
        return ff(x).squeeze(), ff


class Variana(object):

    def __init__(self, target, kernel, ndraws, reflect=False):
        """
        Variational sampler class.

        Fit a target distribution with a Gaussian distribution by
        maximizing an approximate KL divergence based on independent
        random sampling.

        Parameters
        ----------
        target: callable
          returns the log of the target distribution

        kernel: tuple
          a tuple `(m, V)` where `m` is a vector representing the mean
          of the sampling distribution and `V` is a matrix or vector
          representing the variance. If a vector, a diagonal variance
          is assumed.

        ndraws: int
          sample size

        reflect: bool
          if True, reflect the sample about the sampling kernel mean
        """
        self.kernel = as_normalized_gaussian(kernel)
        self.target = target
        self.ndraws = ndraws
        self.reflect = reflect

        # Sample random points
        t0 = time()
        self._sample()
        self.sampling_time = time() - t0

    def _sample(self, x=None, w=None):
        """
        Sample independent points from the specified kernel and
        compute associated distribution values.
        """
        self.x = self.kernel.sample(ndraws=self.ndraws)
        if self.reflect:
            self.x = reflect_sample(self.x, self.kernel.m)
        # Compute pn, the vector of sampled probability values
        # normalized by the maximum probability within the sample
        self._log_pn, self.target = sample_fun(self.target, self.x)
        self._pn, self._logscale = safe_exp(self._log_pn)
        self._log_pn -= self._logscale

    def _get_scale(self):
        return np.exp(self._logscale)

    def fit(self, objective='kl', **args):
        """
        Perform fitting.

        Parameters
        ----------
        objective: str
          one of 'kl' or 'l' standing for discrete Kullback-Leibler
          divergence minimization or weighted likelihood maximization,
          respectively.
        """
        if objective == 'kl':
            return KLFit(self, **args)
        elif objective == 'l':
            return LFit(self, **args)
        else:
            raise ValueError('unknown objective')

    def _get_p(self):
        return self._pn * self._scale

    def _get_log_p(self):
        return self._log_pn + self._logscale

    p = property(_get_p)
    log_p = property(_get_log_p)
    _scale = property(_get_scale)


def vsfit(target, kernel, ndraws, guess=None, reflect=False, objective='kl'):
    """
    Given a target distribution p(x) and a Gaussian kernel w(x), this function returns a 
    Gaussian fit q(x) to p(x) that approximately solves the KL minimization problem:

    q = argmin D(wp/g||wq/g),

    where g(x) is some initial guess Gaussian fit. If None, a flat distribution is assumed.

    The KL divergence is approximated by sampling points indepedently from w(x), and optionally
    reflecting the sample around the kernel mean.

    Note that, if w=g, then the output approximately minimizes the global KL divergence D(p||q).
    """
    if guess is None:
        t = target
    else:
        t = lambda x: target(x) - guess.log(x)
    v = Variana(t, kernel, ndraws, reflect=reflect)
    if guess is None:
        return v.fit(objective=objective).fit
    else:
        return v.fit(objective=objective).fit * guess


  
def gnewton(target, kernel, ndraws, maxiter=100, reflect=False, objective='kl', tol=1e-5):
    from scipy.optimize import brent
    guess = None
    for i in range(maxiter):
        print('Iteration %d' % (i+1))
        # Perform local quadratic approximation
        print('x0 = %s' % kernel.m)
        print('... Variational sampling')
        guess = vsfit(target, kernel, ndraws, reflect=reflect, objective=objective)
        print('tentative x1 = %s' % guess.m)
        # Perform line search
        print(' Brent line search')
        x0 = kernel.m
        xt = lambda a: x0 + a * (guess.m - x0)
        f = lambda a: -target(xt(a))
        amin, fmin, _, _ = brent(f, brack=(0, 1), tol=tol, full_output=True)
        x1 = xt(amin)
        print('corrected x1 = %s (target value = %f)' % (x1, -fmin))
        # Update kernel
        kernel = Gaussian(x1, kernel.V)
        # Stopping criterion
        err = np.max(np.abs(x1 - x0))
        print('err = %f' % err)
        if err < tol:
            break
    return x1, -fmin


