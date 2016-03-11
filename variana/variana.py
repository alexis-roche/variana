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
        self.log_pn, self.target = sample_fun(self.target, self.x)
        self.pn, self.logscale = safe_exp(self.log_pn)
        self.log_pn -= self.logscale


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


  

def gnewton(target, kernel, ndraws, niters, alpha=0.5, beta=0.5, reflect=False, objective='kl'):
    guess = None
    scale = lambda V: np.max(np.linalg.eigh(V)[0])

    for i in range(niters):
        print('Iteration %d' % (i+1))
        print('kernel: m = %s, scale = %s' % (kernel.m, scale(kernel.V)))
        guess = vsfit(target, kernel, ndraws, guess=guess, reflect=reflect, objective=objective)
        print('guess: m = %s, scale = %f' % (guess.m, scale(guess.V)))
        # new kernel
        ##kernel = Gaussian(guess.m, alpha * guess.V)
        m = alpha * kernel.m + (1 - alpha) * guess.m
        V = beta * kernel.V + (1 - beta) * guess.V
        kernel = Gaussian(m, V)
    return guess




