from time import time
import numpy as np

from ..utils import (minimizer, inv_sym_matrix)
from ..gaussian import instantiate_family

VERBOSE = False


class QuadratureFit(object):

    def __init__(self, sample, family='gaussian'):
        """
        Importance weighted likelihood fitting method.
        """
        t0 = time()
        self.sample = sample
        self.dim = sample.x.shape[0]
        self.npts = sample.x.shape[1]
        self.family = instantiate_family(family, self.dim)

        # Pre-compute some stuff and cache it
        self._F = self.family.design_matrix(sample.x)

        # Perform fit
        self._fit()
        self.time = time() - t0

    def _fit(self):
        self._integral = np.dot(self._F, self.sample.w * self.sample._fn)
        self._integral *= self.sample._scale
        wq = self.family.from_integral(self._integral)
        self._gaussian = wq / self.sample.cavity

    def _get_integral(self):
        return self._integral

    def _get_gaussian(self):
        return self._gaussian

    def _get_theta(self):
        return self._gaussian.theta

    theta = property(_get_theta)
    gaussian = property(_get_gaussian)
    integral = property(_get_integral)



class VariationalFit(object):

    def __init__(self, sample, family='gaussian', tol=1e-5, maxiter=None,
                 minimizer='newton'):
        """
        Sampling-based KL divergence minimization.

        Parameters
        ----------
        tol : float
          Tolerance on optimized parameter

        maxiter : int
          Maximum number of iterations in optimization

        minimizer : string
          One of 'newton', 'quasi_newton', steepest', 'conjugate'
        """
        t0 = time()
        self.sample = sample
        self.dim = sample.x.shape[0]
        self.npts = sample.x.shape[1]
        self.family = instantiate_family(family, self.dim)

        # Pre-compute some stuff and cache it
        self._theta = None
        self._F = self.family.design_matrix(sample.x)
        self._gn = None
        self._log_gn = None

        # Initial parameter guess: fit the sampled point with largest probability
        self._theta_init = np.zeros(self._F.shape[0])
        self._theta_init[0] = self.sample._logscale

        # Perform fit
        self.minimizer = minimizer
        self.tol = tol
        self.maxiter = maxiter
        self._fit()
        self.time = time() - t0

    def _update_fit(self, theta):
        """
        Compute fit
        """
        if not theta is self._theta:
            self._log_gn = np.dot(self._F.T, theta) - self.sample._logscale
            self._gn = np.exp(self._log_gn)
            self._theta = theta
            fail = np.isinf(self._log_gn).max() or np.isinf(self._gn).max()
        else:
            fail = False
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
        tmp = self.sample._fn * (self.sample._log_fn - self._log_gn) + self._gn - self.sample._fn
        return np.sum(self.sample.w * tmp)

    def _gradient(self, theta):
        """
        Compute the gradient of the loss.
        """
        self._update_fit(theta)
        return np.dot(self._F, self.sample.w * (self._gn - self.sample._fn))

    def _hessian(self, theta):
        """
        Compute the hessian of the loss.
        """
        self._update_fit(theta)
        return np.dot(self._F * (self.sample.w * self._gn), self._F.T)

    def _pseudo_hessian(self):
        """
        Approximate the Hessian at the minimum by substituting the
        fitted distribution with the target distribution.
        """
        return np.dot(self._F * (self.sample.w * self.sample._fn), self._F.T)

    def _fit(self):
        """
        Perform Gaussian approximation.
        """
        theta = self._theta_init
        meth = self.minimizer
        if meth == 'quasi_newton':
            hessian = self._pseudo_hessian()
        else:
            hessian = self._hessian
        m = minimizer(meth, theta, self._loss, self._gradient,
                      hessian,
                      maxiter=self.maxiter, tol=self.tol,
                      verbose=VERBOSE)
        if VERBOSE:
            print(m)
        self._theta = m.argmin()
        self.minimizer = m

    def _get_theta(self):
        theta = self._theta.copy()
        return theta

    def _get_gaussian(self):
        return self.family.from_theta(self.theta)

    def _get_integral(self):
        return self._get_gaussian().integral()

    theta = property(_get_theta)
    gaussian = property(_get_gaussian)
    integral = property(_get_integral)

