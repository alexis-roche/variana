"""
Variational sampling
"""
import numpy as np
from scipy.optimize import fmin_ncg

from .utils import (probe_time, minimizer, inv_sym_matrix, HUGE, approx_gradient, approx_hessian_diag, approx_hessian)
from .gaussian import (instantiate_family, as_gaussian, Gaussian, FactorGaussian, laplace_approximation)



def safe_exp(x):
    """
    Returns a tuple (exp(x-xmax), xmax).
    """
    xmax = x.max()
    return np.exp(x - xmax), xmax


def reflect_sample(xs, m):
    return np.reshape(np.array([xs.T, m - xs.T]).T,
                      (xs.shape[0], 2 * xs.shape[1]))


def lambdify_target(f, x):
    return lambda x: np.array([f(xi) for xi in x.T])



class VariationalSampler(object):

    def __init__(self, target, cavity, rule='balanced', ndraws=None, reflect=False):
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

        rule: str
          Defines the underlying one-dimensional quadrature precision-3 rule
          One of 'balanced' or 'optimal_d4', 'exact_d3_uniform', 'exact_d3_positive'
        """
        self._cavity = as_gaussian(cavity)
        self._target = target
        self._rule = rule
        self._ndraws = ndraws
        self._reflect = reflect

        # Sample random points
        self._sampling_time = self._sample()
        
    @probe_time
    def _sample(self):
        """
        Sample independent points from the specified cavity and
        compute associated distribution values.
        """
        if self._ndraws is None:
            self._x, self._w = self._cavity.quad3(self._rule)
            self._reflect = True
        else:
            self._x = self._cavity.random(ndraws=self._ndraws)
            if self._reflect:
                self._x = reflect_sample(self._x, self._cavity.m)
            self._w = np.zeros(self._x.shape[1])
            self._w[:] = 1 / float(self._x.shape[1])
        # Compute fn, the vector of sampled factor values normalized
        # by the maximum factor value within the sample
        try:
            self._log_fn = self._target(self._x)
        except:
            self._target = lambdify_target(self._target, self._x)
            self._log_fn = self._target(self._x).squeeze()
        self._fn, self._logscale = safe_exp(self._log_fn)
        self._log_fn -= self._logscale
    
    def fit(self, method='kullback', family='gaussian', global_fit=False,
            minimizer='lbfgs', bounds=None,  tol=1e-5, maxiter=None):
        """
        Perform fitting.

        Parameters
        ----------
        method: str
          one of 'laplace', 'quick_laplace', 'moment' or 'kullback'.
        """
        if method == 'kullback':
            self._fit = KullbackFit(self, family=family, global_fit=global_fit,
                                    minimizer=minimizer, bounds=bounds, tol=tol, maxiter=maxiter)
        elif method == 'moment':
            self._fit = MomentFit(self, family=family, global_fit=global_fit)
        else:
            raise ValueError('unknown method')
        return self._fit.gaussian()
        
    @property
    def x(self):
        return self._x

    @property
    def w(self):
        return self._w


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
    return laplace_approximation(m, u(m), g(m), h(m))




#########################################################
# Fitting objects
#########################################################

class MomentFit(object):

    def __init__(self, sample, family='gaussian', global_fit=False):
        """
        Importance weighted likelihood fitting method.
        """
        self._sample = sample
        self._dim = sample._x.shape[0]
        self._npts = sample._x.shape[1]
        self._family = instantiate_family(family, self._dim)
        self._global_fit = global_fit
        
        # Pre-compute some stuff and cache it
        self._F = self._family.design_matrix(sample._x)

        # Perform fit
        self._fit()

    def _fit(self):
        self._integral = np.dot(self._F, self._sample._w * self._sample._fn)
        self._integral *= np.exp(self._sample._logscale)
        wq = self._family.from_integral(self._integral)
        self._gaussian = wq
        
    @property
    def theta(self):
        return self.gaussian().theta

    def gaussian(self):
        if self._global_fit:
            return self._sample._cavity.Z * self._gaussian
        else:
            return self._gaussian / self._sample._cavity.normalize()


class KullbackFit(object):

    def __init__(self, sample, family='gaussian', global_fit=False,
                 minimizer='lbfgs', bounds=None, tol=1e-5, maxiter=None):
        """
        Sampling-based KL divergence minimization.

        Parameters
        ----------
        tol : float
          Tolerance on optimized parameter

        maxiter : int
          Maximum number of iterations in optimization

        minimizer : string
          One of 'newton', 'steepest', 'conjugate'
        """
        self._sample = sample
        self._dim = sample._x.shape[0]
        self._npts = sample._x.shape[1]
        self._family = instantiate_family(family, self._dim)
        self._global_fit = global_fit

        # Pre-compute some stuff and cache it
        self._theta = None
        self._F = self._family.design_matrix(sample._x)
        self._gn = None
        self._log_gn = None

        # Initial parameter guess: fit the sampled point with largest probability
        self._theta_init = np.zeros(self._F.shape[0])
        self._theta_init[0] = self._sample._logscale

        # Perform fit
        self._minimizer = minimizer
        self._bounds = bounds
        self._tol = tol
        self._maxiter = maxiter
        self._info = self._fit()

    def _update_fit(self, theta):
        """
        Compute fit
        """
        if not self._minimizer in ('lbfgs',):
            if theta is self._theta:
                return True
        self._log_gn = np.dot(self._F.T, theta) - self._sample._logscale
        self._gn = np.exp(self._log_gn)
        self._theta = theta
        fail = np.isinf(self._log_gn).max() or np.isinf(self._gn).max()
        return not fail

    def _loss(self, theta):
        """
        Compute the empirical divergence:

          sum wn [pn * log pn/qn + qn - pn],

        where:
          wn are the weights
          pn is the target factor
          qn is the parametric fit
        """
        if not self._update_fit(theta):
            return np.inf
        tmp = self._sample._fn * (self._sample._log_fn - self._log_gn) + self._gn - self._sample._fn
        return np.sum(self._sample._w * tmp)

    def _gradient(self, theta):
        """
        Compute the gradient of the loss.
        """
        self._update_fit(theta)
        return np.dot(self._F, self._sample._w * (self._gn - self._sample._fn))

    def _hessian(self, theta):
        """
        Compute the hessian of the loss.
        """
        self._update_fit(theta)
        return np.dot(self._F * (self._sample._w * self._gn), self._F.T)

    def _pseudo_hessian(self):
        """
        Approximate the Hessian at the minimum by substituting the
        fitted distribution with the target distribution.
        """
        return np.dot(self._F * (self._sample._w * self._sample._fn), self._F.T)

    def _fit(self):
        """
        Perform Gaussian approximation.
        """
        theta = self._theta_init
        meth = self._minimizer
        if meth == 'steepest':
            hessian = self._pseudo_hessian()
        else:
            hessian = self._hessian
        m = minimizer(meth, theta, self._loss, self._gradient,
                      hessian,
                      maxiter=self._maxiter, tol=self._tol,
                      bounds=self._bounds)
        self._theta = m.argmin()
        return m.info()

    @property
    def theta(self):
        if self._global_fit:
            return self._sample._cavity.theta + self._theta 
        else:
            return self._theta

    def gaussian(self):
        return self._family.from_theta(self.theta)

