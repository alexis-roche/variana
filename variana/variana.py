"""
Variational sampling
"""
from time import time
import numpy as np
from scipy.optimize import fmin_ncg

from .utils import (HUGE, safe_exp, approx_gradient, approx_hessian_diag, approx_hessian)
from .gaussian import (as_normalized_gaussian, Gaussian, FactorGaussian)
from .fit import (VariationalFit, QuadratureFit)


def reflect_sample(xs, m):
    return np.reshape(np.array([xs.T, m - xs.T]).T,
                      (xs.shape[0], 2 * xs.shape[1]))


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

    def fit(self, method='variational', **args):
        """
        Perform fitting.

        Parameters
        ----------
        method: str
          one of 'laplace', 'quick_laplace', 'quadrature' or 'variational'.
        """
        if method == 'variational':
            return VariationalFit(self, **args)
        elif method == 'quadrature':
            return QuadratureFit(self, **args)
        else:
            raise ValueError('unknown method')

    def _get_f(self):
        return self._fn * self._scale

    def _get_log_f(self):
        return self._log_fn + self._logscale

    f = property(_get_f)
    log_f = property(_get_log_f)
    _scale = property(_get_scale)


def prod_factors(f):
    out = f[0]
    for i in range(1, len(f)):
        out *= f[i]
    return out


def laplace_approx(u, g, h, cavity, optimize=True):
    """
    u: loss function -> factor f = exp(-u)
    g: gradient of u 
    h: Hessian of u
    cavity: cavity distribution
    """
    m = cavity.m
    if optimize:
        f = lambda x: u(x) - cavity.log(x.reshape((-1, 1)))
        fprime = lambda x: g(x) + (x - cavity.m) / cavity.v
        fhess_p = lambda x, p: (h(x) + (1 / cavity.v)) * p
        m = fmin_ncg(f, m, fprime, fhess_p=fhess_p, disp=0) 
    dim = len(m)
    u0 = u(m)
    g0 = g(m)
    a0 = .5 * h(m)
    am0 = a0 * m
    theta = np.zeros(2 * dim + 1)
    theta[0] = u0 -np.dot(g0, m) + np.dot(m.T, am0)
    theta[1:1+dim] = g0 - 2 * am0
    theta[1+dim:] = a0
    if cavity.family == 'factor_gaussian':
        return FactorGaussian(theta=theta)
    elif cavity.family == 'gaussian':
        raise ValueError('not implemented yet, brother')
    else:
        raise ValueError('unknown family')


class NumEP(object):

    def __init__(self, utility, batches, prior, guess=None, niters=1, 
                 gamma2=None, ndraws=None, reflect=None, method='variational',
                 gradient=None, hessian=None, step=1e-5, minimizer='newton'):
        """
        Assume: utility = fn(x, i)
        """
        self.utility = utility
        self.batches = batches
        self.nfactors = len(batches)
        self.prior = as_normalized_gaussian(prior)
        self.dim = self.prior.dim
        if guess is None:
            tmp = FactorGaussian(np.zeros(self.dim), HUGE * np.ones(self.dim))
            self.approx_factors = [tmp for a in batches]
        else:
            self.approx_factors = [as_normalized_gaussian(guess) for a in batches]
        self.niters = niters
        self.gamma2 = gamma2
        self.ndraws = ndraws
        self.reflect = reflect
        self.method = method
        self.gradient = gradient
        self.hessian = hessian
        self.step = float(step)  # for finite-difference Laplace
        self.args = {}
        if self.method == 'variational':
            self.args['minimizer'] = minimizer
        self._gaussian = prod_factors([self.prior] + self.approx_factors)
        
    def _get_gaussian(self):
        #return prod_factors([self.prior] + self.approx_factors)
        return self._gaussian

    gaussian = property(_get_gaussian)

    def cavity(self, a):
        #return prod_factors([self.prior] + [self.approx_factors[b] for b in [b for b in self.batches if b != a]])
        return self._gaussian / self.approx_factors[a]
	
    def update_factor(self, a):

        # compute a candidate for the factor approximation 
        target = lambda x: self.utility(x, a)
        cavity = self.cavity(a)
        if self.method in ('quadrature', 'variational'):
            v = Variana(target, cavity, gamma2=self.gamma2, ndraws=self.ndraws, reflect=self.reflect)
            prop = v.fit(method=self.method, family=cavity.family, **self.args).gaussian
        elif self.method in ('laplace', 'quick_laplace'):
            if self.gradient is None:
                g = lambda x: approx_gradient(target, x, self.step)
            else:
                g = lambda x: self.gradient(x, a)
            if self.hessian is None:
                if cavity.family == 'factor_gaussian':
                    h = lambda x: approx_hessian_diag(target, x, self.step)
                else:
                    h = lambda x: approx_hessian(target, x, self.step)
            else:
                h = lambda x: self.hessian(x, a)
            if self.method == 'laplace':
                prop = laplace_approx(target, g, h, cavity, optimize=True)
            else:
                prop = laplace_approx(target, g, h, cavity, optimize=False)
        else:
            raise ValueError('not a method I am aware of, sorry')

        # update factor only if the candidate fit is numerically defined
        if np.max(np.isinf(prop.theta)):
            return

        # if candidate accepted, update the overall fit
        self.approx_factors[a] = prop
        self._gaussian = cavity * prop
            
    def run(self): 
        for a in self.batches:
	    self.update_factor(a)

    def __call__(self):
        for i in range(self.niters):
            print('Iteration n. %d/%d' % (i + 1, niters))
            self.run()



class IncrementalEP(object):

    def __init__(self, prior, guess=None, niters=1, 
                 gamma2=None, ndraws=None, reflect=None, method='variational',
                 step=1e-5, minimizer='newton'):
        """
        Assume: utility = fn(x, i)
        """
        self.prior = as_normalized_gaussian(prior)
        self.dim = self.prior.dim
        self.niters = niters
        self.gamma2 = gamma2
        self.ndraws = ndraws
        self.reflect = reflect
        self.method = method
        self.step = float(step)  # for finite-difference Laplace
        self.args = {}
        if self.method == 'variational':
            self.args['minimizer'] = minimizer
        self._gaussian = self.prior
        
    def _get_gaussian(self):
        return self._gaussian

    gaussian = property(_get_gaussian)

    def update_factor(self, utility, gradient=None, hessian=None):

        # compute a candidate for the factor approximation 
        if self.method in ('quadrature', 'variational'):
            v = Variana(utility, self._gaussian, gamma2=self.gamma2, ndraws=self.ndraws, reflect=self.reflect)
            prop = v.fit(method=self.method, family=self._gaussian.family, **self.args).gaussian
        elif self.method in ('laplace', 'quick_laplace'):
            if gradient is None:
                gradient = lambda x: approx_gradient(utility, x, self.step)
            if hessian is None:
                if self._gaussian.family == 'factor_gaussian':
                    hessian = lambda x: approx_hessian_diag(utility, x, self.step)
            if self.method == 'laplace':
                prop = laplace_approx(utility, gradient, hessian, self._gaussian, optimize=True)
            else:
                prop = laplace_approx(utility, gradient, hessian, self._gaussian, optimize=False)
        else:
            raise ValueError('not a method I am aware of, sorry')

        # update factor only if the candidate fit is numerically defined
        if np.max(np.isinf(prop.theta)):
            return

        # if candidate accepted, update the overall fit
        self._gaussian = self._gaussian * prop
            
    def __call__(self):
        for i in range(self.niters):
            print('Iteration n. %d/%d' % (i + 1, niters))
            self.run()


