"""
Variational sampling
"""
import numpy as np
from scipy.optimize import fmin_ncg

from .utils import probe_time, minimizer
from .gaussian import instantiate_family, as_gaussian, Gaussian, FactorGaussian, laplace_approximation



def safe_exp(x):
    """
    Returns a tuple (exp(x-xmax), xmax).
    """
    xmax = x.max()
    return np.exp(x - xmax), xmax


def reflect_sample(xs, m):
    return np.reshape(np.array([xs.T, m - xs.T]).T, (xs.shape[0], 2 * xs.shape[1]))


def vectorize(f, x):
    return lambda x: np.array([f(xi) for xi in x.T])


class VariationalSampling(object):

    def __init__(self, log_factor, cavity, rule='balanced', ndraws=None, reflect=False):
        """Variational sampling class.

        Fit a target factor with a Gaussian distribution by maximizing
        an approximate KL divergence based on Gaussian quadrature or
        independent random sampling.

        Parameters
        ----------
        log_factor: callable
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
        self._log_factor = log_factor
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
            self._log_fn = self._log_factor(self._x)
        except:
            self._log_factor = vectorize(self._log_factor, self._x)
            self._log_fn = self._log_factor(self._x).squeeze()
        self._fn, self._logscale = safe_exp(self._log_fn)
        self._log_fn -= self._logscale
    
    def fit(self, method='kullback', family='factor_gaussian', global_fit=False,
            minimizer='lbfgs', vmax=None,  tol=1e-5, maxiter=None):
        """
        Perform fitting.

        Parameters
        ----------
        method: str
          one of 'laplace', 'quick_laplace', 'moment' or 'kullback'.
        """
        if method == 'kullback':
            self._fit = KullbackFit(self, family, vmax=vmax, global_fit=global_fit, 
                                    minimizer=minimizer, tol=tol, maxiter=maxiter)
        elif method == 'moment':
            self._fit = MomentFit(self, family, vmax=vmax, global_fit=global_fit)
        else:
            raise ValueError('unknown method')
        return self._fit.gaussian()
        
    @property
    def x(self):
        return self._x

    @property
    def w(self):
        return self._w

    @property
    def log_fx(self):
        return self._log_fn + self._logscale

    @property
    def fx(self):
        return np.exp(self._logscale) * self._fn
    
    
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

    def __init__(self, sample, family, vmax=None, global_fit=False):
        """
        Importance weighted likelihood fitting method.
        """
        self._sample = sample
        self._dim = sample._x.shape[0]
        self._npts = sample._x.shape[1]
        self._family = instantiate_family(family, self._dim)
        self._global_fit = global_fit
        self._vmax = vmax
        if not vmax is None and 'family' == 'gaussian':
            raise NotImplementedError('Second-order constraints not implemented for full Gaussian fitting.')
        
        # Pre-compute some stuff and cache it
        self._F = self._family.design_matrix(sample._x)

        # Perform fit
        self._fit()

    def _fit(self):
        self._integral = np.dot(self._F, self._sample._w * self._sample._fn)
        self._integral *= np.exp(self._sample._logscale)
        wq = self._family.from_integral(self._integral)
        self._factor_fit = wq
        if not self._vmax is None:
            self._factor_fit.theta[(self._dim + 1):] = \
                                np.minimum(self._factor_fit.theta[(self._dim + 1):], \
                                -.5 / self._vmax - self._sample._cavity.theta[(self._dim + 1):])
        
    @property
    def theta(self):
        return self.gaussian().theta

    def gaussian(self):
        if self._global_fit:
            return self._sample._cavity.Z * self._factor_fit
        else:
            return self._factor_fit / self._sample._cavity.normalize()


def make_bounds(vmax, dim, theta0):
    theta_max = np.full(dim, -.5 / vmax) - theta0[(dim + 1):]
    bounds = [(None, None) for i in range(dim + 1)]
    bounds += [(None, theta_max[i]) for i in range(dim)]
    return bounds


class KullbackFit(object):

    def __init__(self, sample, family, vmax=None, global_fit=False,
                 minimizer='lbfgs', tol=1e-5, maxiter=None):
        """
        Sampling-based KL divergence minimization.

        Parameters
        ----------
        tol : float
          Tolerance on optimized parameter

        maxiter : None or int
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
        self._minimizer = str(minimizer)
        self._vmax = vmax
        if not vmax is None and family == 'gaussian':
            raise NotImplementedError('Second-order constraints not implemented for full Gaussian fitting.')
        self._tol = float(tol)
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
        if self._vmax is None:
            bounds = None
        else:
            bounds = make_bounds(self._vmax, self._dim, self._sample._cavity.theta)
        m = minimizer(meth, theta, self._loss, self._gradient,
                      hessian,
                      maxiter=self._maxiter, tol=self._tol,
                      bounds=bounds)
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


# Helper function
def dist_fit(log_factor, cavity, factorize=True, ndraws=None, global_fit=False, method='kullback', minimizer='lbfgs', vmax=None):
    vs = VariationalSampling(log_factor, cavity, ndraws=ndraws)
    if factorize:
        family = 'factor_gaussian'
    else:
        family = 'gaussian'
    return vs.fit(family=family, global_fit=global_fit, method=method, minimizer=minimizer, vmax=vmax)



# Iterative M-projection

class Variana(object):

    def __init__(self, target, dim, alpha, vmax, stride, m0=None, v0=None):
        self._target = target
        self._dim = int(dim)
        self._alpha = float(alpha)
        self._vmax = vmax
        if stride > dim:
            raise ValueError('stride (%d) should be less or equal to dim (%d)' % (stride, dim)) 
        self._stride = int(stride)
        if m0 is None:
            m0 = np.zeros(self._dim)
        if v0 is None:
            v0 = self._vmax
        self._fit = FactorGaussian(m=m0, v=np.full(self._dim, v0), K=np.exp(target(m0)))
        aux = np.arange(self._dim // self._stride, dtype=int) * self._stride
        self._slices = np.append(aux, self._dim)
        
    def update(self, i0, i1):
        """
        Improve the fit for slice(i0, i1) in the state space
        """
        # Auxiliary variables
        s = slice(i0, i1)
        s1 = slice(1 + i0, 1 + i1)
        s2 = slice(1 + i0 + self._fit.dim, 1 + i1 + self._fit.dim)
        loc_dim = i1 - i0

        # Define local cavity from the current fit
        loc_theta = np.concatenate((np.array((self._fit.theta[0],)), \
                                    np.array(self._fit.theta[s1]), \
                                    np.array(self._fit.theta[s2])))
        init_loc_fit = FactorGaussian(theta=loc_theta)
        log_gamma = self._alpha * (init_loc_fit.logK - self._fit.logK)

        # Pick current fit center (will be modified in place)                
        x = self._fit.m
        self._fit.cleanup()

        # Define log target
        def log_factor(xa):
            x[s] = xa
            return self._alpha * self._target(x)

        # Perform local fit by variational sampling 
        vs = VariationalSampling(log_factor, init_loc_fit ** (1 - self._alpha))
        if hasattr(self._vmax, '__getitem__'):
            vmax = self._vmax[s]
        else:
            vmax = self._vmax
        loc_fit = vs.fit(vmax=vmax, global_fit=True)

        # Update overall fit
        self._fit.set_theta(loc_fit.theta[0] + log_gamma, indices=0)
        self._fit.set_theta(loc_fit.theta[slice(1, 1 + loc_dim)], indices=s1)
        self._fit.set_theta(loc_fit.theta[slice(1 + loc_dim, 1 + 2 * loc_dim)], indices=s2)

    def run(self, niter):
        for i in range(niter):
            for j in range(len(self._slices) - 1):
                self.update(self._slices[j], self._slices[j + 1])
        return self._fit

    
