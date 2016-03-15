from time import time
import numpy as np

from .utils import (inv_sym_matrix, min_methods)
from .gaussian import (Gaussian, GaussianFamily, FactorGaussianFamily)

VERBOSE = False

families = {'gaussian': GaussianFamily,
            'factor_gaussian': FactorGaussianFamily}


class LFit(object):

    def __init__(self, sample, family='gaussian'):
        """
        Importance weighted likelihood fitting method.
        """
        t0 = time()
        self.sample = sample
        self.dim = sample.x.shape[0]
        self.npts = sample.x.shape[1]

        # Instantiate fitting family
        if family not in families.keys():
            raise ValueError('unknown family')
        self.family = families[family](self.dim)

        # Pre-compute some stuff and cache it
        self._F = self.family.design_matrix(sample.x)

        # Perform fit
        self._do_fitting()
        self.time = time() - t0

    def _do_fitting(self):
        self._integral = np.dot(self._F, self.sample._pn) / self.npts
        self._integral *= np.exp(self.sample._logscale)
        wq = self.family.from_integral(self._integral)
        self._fit = wq / self.sample.kernel

    def _get_integral(self):
        return self._integral

    def _get_var_integral(self):
        """
        Estimate variance on integral estimate
        """
        n = self._integral.size
        var = np.dot(self._F * (self.sample._pn ** 2), self._F.T) / self.npts \
            - np.dot(self._integral.reshape(n, 1), self._integral.reshape(1, n))
        var /= self.npts
        return var

    def _get_fit(self):
        return self._fit

    def _get_theta(self):
        return self._fit.theta

    def _get_sensitivity_matrix(self):
        # compute the normalized probabilities
        log_qn = np.dot(self._F.T, self._fit.theta)
        qn = np.exp(log_qn - self.sample._logscale)
        return np.dot(self._F * qn, self._F.T) *\
            (np.exp(self.sample._logscale) / self.npts)

    def _get_var_theta(self):
        inv_sensitivity_matrix = inv_sym_matrix(self.sensitivity_matrix)
        return np.dot(np.dot(inv_sensitivity_matrix, self.var_integral),
                      inv_sensitivity_matrix)

    def _get_kl_error(self):
        return .5 * np.trace(np.dot(self.var_integral,
                                    inv_sym_matrix(self.sensitivity_matrix)))

    theta = property(_get_theta)
    fit = property(_get_fit)
    integral = property(_get_integral)
    var_integral = property(_get_var_integral)
    var_theta = property(_get_var_theta)
    sensitivity_matrix = property(_get_sensitivity_matrix)
    kl_error = property(_get_kl_error)


class KLFit(object):

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

        # Instantiate fitting family
        if family not in families.keys():
            raise ValueError('unknown family')
        self.family = families[family](self.dim)

        # Pre-compute some stuff and cache it
        self._theta = None
        self._F = self.family.design_matrix(sample.x)
        self._qn = None
        self._log_qn = None

        # Initial parameter guess: fit the sampled point with largest probability
        self._theta_init = np.zeros(self._F.shape[0])
        self._theta_init[0] = self.sample._logscale

        # Perform fit
        self.minimizer = minimizer
        self.tol = tol
        self.maxiter = maxiter
        self._do_fitting()
        self.time = time() - t0

    def _update_fit(self, theta):
        """
        Compute fit
        """
        if not theta is self._theta:
            self._log_qn = np.dot(self._F.T, theta) - self.sample._logscale
            self._qn = np.exp(self._log_qn)
            self._theta = theta
            fail = np.isinf(self._log_qn).max() or np.isinf(self._qn).max()
        else:
            fail = False
        return not fail

    def _loss(self, theta):
        """
        Compute the empirical divergence:

          sum[pn * log pn/qn + qn - pn],

        where:
          pn is the target distribution
          qn is the parametric fit
        """
        if not self._update_fit(theta):
            return np.inf
        return np.sum(self.sample._pn * (self.sample._log_pn - self._log_qn)
                      + self._qn - self.sample._pn)

    def _gradient(self, theta):
        """
        Compute the gradient of the loss.
        """
        self._update_fit(theta)
        return np.dot(self._F, self._qn - self.sample._pn)

    def _hessian(self, theta):
        """
        Compute the hessian of the loss.
        """
        self._update_fit(theta)
        return np.dot(self._F * self._qn, self._F.T)

    def _pseudo_hessian(self):
        """
        Approximate the Hessian at the minimum by substituting the
        fitted distribution with the target distribution.
        """
        return np.dot(self._F * self.sample.pe, self._F.T)

    def _do_fitting(self):
        """
        Perform Gaussian approximation.
        """
        theta = self._theta_init
        meth = self.minimizer
        if meth not in min_methods.keys():
            raise ValueError('unknown minimizer')
        if meth in ('newton', 'ncg'):
            m = min_methods[meth](theta, self._loss, self._gradient,
                                  self._hessian,
                                  maxiter=self.maxiter, tol=self.tol,
                                  verbose=VERBOSE)
        elif meth in ('quasi_newton',):
            m = min_methods[meth](theta, self._loss, self._gradient,
                                  self._pseudo_hessian(),
                                  maxiter=self.maxiter, tol=self.tol,
                                  verbose=VERBOSE)
        else:
            m = min_methods[meth](theta, self._loss, self._gradient,
                                  maxiter=self.maxiter, tol=self.tol,
                                  verbose=VERBOSE)
        if VERBOSE:
            m.message()
        self._theta = m.argmin()
        self.minimizer = m

    def _var_integral(self, theta):
        self._update_fit(theta)
        return np.dot(self._F * ((self.sample._pn - self._qn) ** 2), self._F.T)\
            * (np.exp(2 * self.sample._logscale) / (self.npts ** 2))

    def _sensitivity_matrix(self, theta):
        return self._hessian(self._theta) *\
            (np.exp(self.sample._logscale) / self.npts)

    def _get_theta(self):
        theta = self._theta.copy()
        return theta

    def _get_fit(self):
        return self.family.from_theta(self.theta)

    def _get_integral(self):
        return self._get_fit().integral()

    def _get_var_integral(self):
        return self._var_integral(self._theta)

    def _get_sensitivity_matrix(self):
        return self._sensitivity_matrix(self._theta)

    def _get_var_theta(self):
        inv_sensitivity_matrix = inv_sym_matrix(self.sensitivity_matrix)
        return np.dot(np.dot(inv_sensitivity_matrix,
                             self._var_integral(self._theta)),
                      inv_sensitivity_matrix)

    def _get_kl_error(self):
        """
        Estimate the expected excess KL divergence.
        """
        return .5 * np.trace(np.dot(self.var_integral,
                                    inv_sym_matrix(self.sensitivity_matrix)))

    theta = property(_get_theta)
    fit = property(_get_fit)
    integral = property(_get_integral)
    var_integral = property(_get_var_integral)
    var_theta = property(_get_var_theta)
    sensitivity_matrix = property(_get_sensitivity_matrix)
    kl_error = property(_get_kl_error)


