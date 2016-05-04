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

    def __init__(self, target, cavity, gamma2=None, ndraws=None, reflect=False):
        """Variational sampler class.

        Fit a target factor with a Gaussian distribution by maximizing
        an approximate KL divergence based on Gaussian quadrature or
        independent random sampling.

        Parameters
        ----------
        target: callable
          returns the log of the target factor (utility)

        cavity: tuple
          a tuple `(m, V)` where `m` is a vector representing the mean
          of the sampling distribution and `V` is a matrix or vector
          representing the variance. If a vector, a diagonal variance
          is assumed.

        ndraws: None or int
          if None, a precision 3 quadrature rule is used. If int,
          random sample size

        reflect: bool
          if True, reflect the sample about the cavity mean
        """
        self.cavity = as_normalized_gaussian(cavity)
        self.target = target
        self.gamma2 = gamma2
        self.ndraws = ndraws
        self.reflect = reflect

        # Sample random points
        t0 = time()
        self._sample()
        self.sampling_time = time() - t0

    def _sample(self):
        """
        Sample independent points from the specified cavity and
        compute associated distribution values.
        """
        if self.ndraws is None:
            if self.gamma2 is None:
                self.gamma2 = self.cavity.dim + .5
            self.x, self.w = self.cavity.quad3(self.gamma2)
            self.reflect = True
        else:
            self.x = self.cavity.random(ndraws=self.ndraws)
            if self.reflect:
                self.x = reflect_sample(self.x, self.cavity.m)
            self.w = np.zeros(self.x.shape[1])
            self.w[:] = 1 / float(self.x.shape[1])
        # Compute fn, the vector of sampled factor values normalized
        # by the maximum factor value within the sample
        self._log_fn, self.target = sample_fun(self.target, self.x)
        self._fn, self._logscale = safe_exp(self._log_fn)
        self._log_fn -= self._logscale

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
        return self._fn * self._scale

    def _get_log_p(self):
        return self._log_fn + self._logscale

    p = property(_get_p)
    log_p = property(_get_log_p)
    _scale = property(_get_scale)


def vsfit(target, cavity, ndraws, guess=None, reflect=False, objective='kl'):
    """
    Given a target distribution p(x) and a Gaussian cavity w(x), this function returns a 
    Gaussian fit q(x) to p(x) that approximately solves the KL minimization problem:

    q = argmin D(wp/g||wq/g),

    where g(x) is some initial guess Gaussian fit. If None, a flat distribution is assumed.

    The KL divergence is approximated by sampling points indepedently from w(x), and optionally
    reflecting the sample around the cavity mean.

    Note that, if w=g, then the output approximately minimizes the global KL divergence D(p||q).
    """
    if guess is None:
        t = target
    else:
        t = lambda x: target(x) - guess.log(x)
    v = Variana(t, cavity, ndraws, reflect=reflect)
    if guess is None:
        return v.fit(objective=objective).fit
    else:
        return v.fit(objective=objective).fit * guess


  
def gnewton(target, cavity, ndraws, maxiter=100, reflect=False, objective='kl', tol=1e-5):
    from scipy.optimize import brent
    guess = None
    for i in range(maxiter):
        print('Iteration %d' % (i+1))
        # Perform local quadratic approximation
        print('x0 = %s' % cavity.m)
        print('... Variational sampling')
        guess = vsfit(target, cavity, ndraws, reflect=reflect, objective=objective)
        print('tentative x1 = %s' % guess.m)
        # Perform line search
        print(' Brent line search')
        x0 = cavity.m
        xt = lambda a: x0 + a * (guess.m - x0)
        f = lambda a: -target(xt(a))
        amin, fmin, _, _ = brent(f, brack=(0, 1), tol=tol, full_output=True)
        x1 = xt(amin)
        print('corrected x1 = %s (target value = %f)' % (x1, -fmin))
        # Update cavity
        cavity = Gaussian(x1, cavity.V)
        # Stopping criterion
        err = np.max(np.abs(x1 - x0))
        print('err = %f' % err)
        if err < tol:
            break
    return x1, -fmin


