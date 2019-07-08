"""
Variational sampling

TODO:

Convergence test in star approximation
Test iteration improvement in star approximation (???)
"""
import numpy as np
from scipy.optimize import fmin_ncg

from .utils import probe_time, is_sequence, minimizer, approx_gradient, approx_hessian, approx_hessian_diag, rms
from .gaussian import gaussian_family, as_gaussian, Gaussian, FactorGaussian, laplace_approximation



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
        self._fit = None
        
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
        # For numerical stability, we normalize the factor values by
        # the maximum factor value within the sample, and keep track
        # of the logarithm of that maximum value.
        try:
            self._log_fn = self._log_factor(self._x)
        except:
            self._log_factor = vectorize(self._log_factor, self._x)
            self._log_fn = self._log_factor(self._x).squeeze()
        self._log_fmax = self._log_fn.max()
        self._log_fn -= self._log_fmax
        self._fn = np.exp(self._log_fn)
    
    def fit(self, proxy='discrete_kl', family='factor_gaussian',
            output_factor=False, vmax=None,
            optimizer='lbfgs', tol=1e-5, maxiter=None,
            hess_diag_approx=False, output_info=False):
        """
        Perform fitting.

        Parameters
        ----------
        proxy: str
          'likelihood' or 'discrete_kl'.

        family: str
          'factor_gaussian' or 'gaussian'.

        output_factor: bool
          True if the factor approximation only is to be output.

        vmax: None or float
          If float, applies a maximum variance constraint to the fit.

        optimizer: str
          Only applicable to 'discrete_kl' fitting proxy.

        tol: float
          Only applicable to 'discrete_kl' fitting proxy.

        maxiter: int
          Only applicable to 'discrete_kl' fitting proxy.

        hess_diag_approx: bool
          Only applicable to 'newton' optimizer.        
        """
        if proxy == 'discrete_kl':
            self._fit = DiscreteKLFit(self, family, vmax=vmax, output_factor=output_factor, 
                                      optimizer=optimizer, tol=tol, maxiter=maxiter,
                                      hess_diag_approx=hess_diag_approx)
        elif proxy == 'likelihood':
            self._fit = LikelihoodFit(self, family, vmax=vmax, output_factor=output_factor)
        else:
            raise ValueError('unknown proxy')

        gauss = self._fit.gaussian()
        if output_info:
            if proxy == 'discrete_kl':
                return gauss, self._fit._info
            else:
                return gauss, None
        else:
            return gauss
        
    @property
    def x(self):
        return self._x

    @property
    def w(self):
        return self._w

    @property
    def log_fx(self):
        return self._log_fn + self._log_fmax

    @property
    def fx(self):
        return np.exp(self._log_fmax) * self._fn


    
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

class LikelihoodFit(object):

    def __init__(self, sample, family, vmax=None, output_factor=False):
        """
        Importance weighted likelihood fitting proxy.
        """
        self._sample = sample
        self._dim = sample._x.shape[0]
        self._npts = sample._x.shape[1]
        self._family = str(family)
        self._vmax = vmax
        if not vmax is None and 'family' == 'gaussian':
            raise NotImplementedError('Second-order constraints not implemented for full Gaussian fitting.')
        self._output_factor = output_factor
        
        # Pre-compute some stuff and cache it
        self._family_obj = gaussian_family(self._family, self._dim)
        self._F = self._family_obj.design_matrix(sample._x)

        # Perform fit
        self._run()

    def _run(self):
        self._integral = np.dot(self._F, self._sample._w * self._sample._fn)
        self._integral *= np.exp(self._sample._log_fmax)
        self._full_fit = self._family_obj.from_integral(self._integral)
        if not self._vmax is None:
            self._full_fit.gate_variance(self._vmax)
            
    def gaussian(self):
        if self._output_factor:
            return self._full_fit / self._sample._cavity.normalize()
        else:
            return self._sample._cavity.Z * self._full_fit
            



class DiscreteKLFit(object):

    def __init__(self, sample, family, vmax=None, output_factor=False,
                 optimizer='lbfgs', tol=1e-5, maxiter=None,
                 hess_diag_approx=False):
        """
        Sampling-based KL divergence minimization.

        Parameters
        ----------
        tol : float
          Tolerance on optimized parameter

        maxiter : None or int
          Maximum number of iterations in optimization

        optimizer : string
          One of 'newton', 'steepest', 'conjugate'
        """
        self._sample = sample
        self._dim = sample._x.shape[0]
        self._npts = sample._x.shape[1]
        self._family = str(family)
        self._output_factor = output_factor

        # Pre-compute some stuff and cache it
        self._family_obj = gaussian_family(self._family, self._dim)
        self._F = self._family_obj.design_matrix(sample._x)
        self._gn = None
        self._log_gn = None

        # Perform fit
        self._optimizer = str(optimizer)
        self._vmax = vmax
        if not vmax is None and family == 'gaussian':
            raise NotImplementedError('Second-order constraints not implemented for full Gaussian fitting.')
        self._tol = float(tol)
        self._maxiter = maxiter
        self._hess_diag_approx = bool(hess_diag_approx)
        self._info = self._run()

    def _update_fit(self, theta):
        """
        Compute fit
        """
        if not self._optimizer in ('lbfgs',):
            if theta is self._theta:
                return True
        self._log_gn = np.dot(self._F.T, theta) - self._sample._log_fmax
        self._gn = np.exp(self._log_gn)
        self._theta = theta
        fail = np.isinf(self._log_gn).max() or np.isinf(self._gn).max()
        return not fail

    def _loss(self, theta):
        """
        Compute the empirical divergence:

          sum wn [fn * log fn/gn + fn - gn],

        where:
          wn are the weights
          fn is the normalized target factor
          gn is the normalized parametric fit
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

    def _hessian_diag(self, theta):
        """
        Compute the hessian of the loss.
        """
        self._update_fit(theta)
        return np.diag(np.sum((self._sample._w * self._gn) * (self._F ** 2), axis=1))

    def _pseudo_hessian(self):
        """
        Approximate the Hessian at the minimum by substituting the
        fitted distribution with the target distribution.
        """
        return np.dot(self._F * (self._sample._w * self._sample._fn), self._F.T)

    def _run(self):
        """
        Perform Gaussian approximation via discrete KL minimization.

        This class relies on minimizing the KL divergence between the
        **normalized** target factor (via division by its maximum) and
        the consistently normalized parametric fit:

        gn_theta = (1 / fmax) exp[theta' phi]

        This is equal to the actual KL divergence divided by the
        factor maximum, which is constant, therefore minimizing the
        normalized KL divergence is equivalent to minimizing the
        actual one.
        """
        if self._optimizer == 'steepest':
            hessian = self._pseudo_hessian()
        else:
            if self._hess_diag_approx:
                hessian = self._hessian_diag
            else:
                hessian = self._hessian
        bounds = None
        if not self._vmax is None:
            if self._family == 'factor_gaussian':
                theta2_max = -.5 / self._vmax - self._sample._cavity.theta[(self._dim + 1):]
                bounds = [(None, None) for i in range(self._dim + 1)]
                bounds += [(None, theta2_max[i]) for i in range(self._dim)]

        # Initial parameter guess: fit the sampled point with largest probability
        self._theta_init = np.zeros(self._F.shape[0])
        self._theta_init[0] = self._sample._log_fmax
        self._theta = self._theta_init
        m = minimizer(self._loss, self._theta, self._optimizer,
                      self._gradient, hessian,
                      maxiter=self._maxiter, tol=self._tol,
                      bounds=bounds)
        theta = m.argmin()
        self._factor_fit = self._family_obj.from_theta(theta)
        return m.info()

    def gaussian(self):
        if self._output_factor:
            return self._factor_fit
        else:
            return self._sample._cavity * self._factor_fit
    
       
    
# Helper function
def dist_fit(log_factor, cavity, factorize=True, ndraws=None, output_factor=False, proxy='discrete_kl', optimizer='lbfgs', vmax=None):
    vs = VariationalSampling(log_factor, cavity, ndraws=ndraws)
    if factorize:
        family = 'factor_gaussian'
    else:
        family = 'gaussian'
    return vs.fit(family=family, output_factor=output_factor, proxy=proxy, vmax=vmax, optimizer=optimizer)



# Star approximation (iterative M-projection)

class StarApproximation(object):

    def __init__(self, log_target, init_fit, alpha, vmax, learning_rate=1, block_size=None, proxy='discrete_kl'):
        self._log_target = log_target
        self._fit = as_gaussian(init_fit)
        self._alpha = float(alpha)
        self._vmax = vmax
        self._learning_rate = float(learning_rate)
        if block_size is None:
            self._slices = np.array((0, self._fit.dim))
        elif is_sequence(block_size):
            self._slices = np.append(0, np.cumsum(block_size))
        else:
            stride = min(block_size, self._fit.dim)
            self._slices = np.append(np.arange(self._fit.dim // stride, dtype=int) * stride,
                                     self._fit.dim)
        self._proxy = str(proxy)
        
    def update(self, block_idx, **kwargs):
        """
        Improve the fit for slice(i0, i1) in the state space
        """
        # Auxiliary variables
        i0, i1 = self._slices[block_idx:(block_idx + 2)]
        s = slice(i0, i1)
        s1 = slice(1 + i0, 1 + i1)
        s2 = slice(1 + i0 + self._fit.dim, 1 + i1 + self._fit.dim)
        loc_dim = i1 - i0
        full_dim = (i0 == 0) and (i1 == self._fit.dim)

        # Define local cavity from the current fit
        if full_dim:
            loc_theta = self._fit.theta
            init_loc_fit = self._fit
        else:
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
            return self._alpha * self._log_target(x)

        # Perform local fit by variational sampling
        if self._proxy in ('discrete_kl', 'likelihood'):
            vs = VariationalSampling(log_factor, init_loc_fit ** (1 - self._alpha))
            if is_sequence(self._vmax):
                vmax = self._vmax[s]
            else:
                vmax = self._vmax
            loc_fit = vs.fit(proxy=self._proxy, vmax=vmax, **kwargs)
        else:
            la = LaplaceApproximation(log_factor, init_loc_fit.m)
            loc_fit = (init_loc_fit ** (1 - self._alpha)) * la.fit()

        # Update overall fit
        if self._learning_rate < 1:
            loc_theta = (1 - self._learning_rate) * loc_theta + self._learning_rate * loc_fit.theta
        else:
            loc_theta = loc_fit.theta
        if full_dim:
            self._fit.set_theta(loc_theta)
        else:
            self._fit.set_theta(loc_theta[0] + self._learning_rate * log_gamma, indices=0)
            self._fit.set_theta(loc_theta[slice(1, 1 + loc_dim)], indices=s1)
            self._fit.set_theta(loc_theta[slice(1 + loc_dim, 1 + 2 * loc_dim)], indices=s2)

    def fit(self, niter, **kwargs):
        for i in range(niter):
            for j in range(len(self._slices) - 1):
                self.update(j, **kwargs)
        return self._fit

    
# Laplace approximation object

class LaplaceApproximation(object):

    def __init__(self, log_target, x0, track_mode=False,
                 grad=None, hess=None, hess_diag=None, epsilon=1e-5,
                 optimizer='lbfgs', **args):
        self._log_target = log_target
        self._x0 = np.asarray(x0)
        self._epsilon = float(epsilon)
        if grad is None:
            self._grad = lambda x: approx_gradient(log_target, x, self._epsilon)
        else:
            self._grad = grad
        if hess is None:
            self._hess = lambda x: approx_hessian(log_target, x, self._epsilon)
        else:
            self._hess = hess
        if hess_diag is None:
            if len(self._x0) == 1:
                self._hess_diag = self._hess
            else:
                self._hess_diag = lambda x: np.diag(self._hess(x))
        else:
            self._hess_diag = hess_diag
        if track_mode:
            loss = lambda x: -self._log_target(x)
            grad = lambda x: -self._grad(x)
            hess = lambda x: -self._hess(x)
            m = minimizer(loss, self._x0, optimizer, grad, hess, **args)
            self._x0 = m.argmin()
        
    def fit(self, family='factor_gaussian'):
        if family == 'factor_gaussian':
            hess = self._hess_diag
        else:
            hess = self._hess
        return laplace_approximation(self._x0, self._log_target(self._x0), self._grad(self._x0), hess(self._x0))
    


##########################################################
# ONLINE METHODS
##########################################################

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



class OnlineIProj(object):

    def __init__(self, log_target, init_fit, gamma, vmax=1e5, vmin=1e-10):
        self._gen_init(log_target, init_fit, vmax, vmin)
        self.reset(gamma)

    def _gen_init(self, log_target, init_fit, vmax, vmin):
        self._log_target = log_target
        if isinstance(init_fit, OnlineIProj):
            self._logK = init_fit.logK
        else:
            init_fit = as_gaussian(init_fit)
            self._logK = log_target(init_fit.m)
        self._m = init_fit.m
        self._v = np.minimum(np.maximum(init_fit.v, vmin), vmax)
        self._dim = len(self._m)
        self._vmax = float(vmax)
        self._vmin = float(vmin)
        self._force = self.force
        # Init ground truth parameters
        self._logZt = 0
        self._mt = 0
        self._vt = 0
        
    def reset(self, gamma):
        self._gamma = float(gamma)
        
    def update_fit(self, dtheta):
        prec_ratio = np.minimum(np.maximum(1 - SQRT_TWO * dtheta[(self._dim + 1):], self._v / self._vmax), self._v / self._vmin)
        self._v /= prec_ratio
        mean_diff = (np.sqrt(self._v) / prec_ratio) * dtheta[1:(self._dim + 1)]
        self._m += mean_diff
        self._logK += dtheta[0] + .5 * (np.sum(prec_ratio + mean_diff ** 2 / self._v) - self._dim)

    def ortho_basis(self, x):
        phi1 = (x - self._m) / np.sqrt(self._v)
        phi2 = (phi1 ** 2 - 1) / SQRT_TWO
        return np.append(1, np.concatenate((phi1, phi2)))
   
    def sample(self):
        return np.sqrt(self._v) * np.random.normal(size=self._dim) + self._m

    def log(self, x):
        return self._logK - .5 * mahalanobis(x, self._m, self._v)

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
        import pylab as pl
        pl.figure()
        pl.title(title)
        pl.plot(self._rec)
        pl.legend(('Z', 'm', 'v'))
        pl.show()
    


class OnlineContextFit(OnlineIProj):

    def __init__(self, log_factor, cavity, gamma, vmax=1e5, vmin=1e-10, proxy='discrete_kl'):
        self._cavity = as_gaussian(cavity)
        self._gen_init(log_factor, self._cavity.copy(), vmax, vmin)
        self._log_factor = log_factor
        self._log_target = None
        if proxy == 'likelihood':
            self._force = self.force_likelihood
        elif proxy != 'discrete_kl':
            raise ValueError('Unknown proxy')
        self.reset(gamma)

    def sample(self):
        return np.sqrt(self._cavity.v) * np.random.normal(size=self._dim) + self._cavity.m        

    def log_fitted_factor(self, x):
        return self.log(x) + .5 * mahalanobis(x, self._cavity.m, self._cavity.v)

    def log_rho(self):
        return logZ(0, self._cavity.v) - logZ(self._logK, self._v)

    def epsilon(self, x):
        return safe_diff_exp(self._log_factor(x), self.log_fitted_factor(x), self.log_rho())
        
    def force_likelihood(self, x):
        f = safe_exp(self._log_factor(x), self.log_rho()) * self.ortho_basis(x)
        f[0] -= 1
        return f

    def factor_fit(self):
        g = FactorGaussian(self._m, self._v, logK=self._logK) / FactorGaussian(self._cavity.m, self._cavity.v, logK=0)
        return g

    def stepsisze(self):
        return self._gamma


    
class OnlineStarFit(OnlineIProj):

    def __init__(self, log_target, init_fit, alpha, gamma, vmax=1e5, vmin=1e-10, proxy='discrete_kl'):
        self._gen_init(log_target, init_fit, vmax, vmin)
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


    
class OnlineMProj(OnlineIProj):

    def __init__(self, log_target, init_fit, gamma, lda=1, vmax=1e5, vmin=1e-10, proxy='discrete_kl'):
        self._gen_init(log_target, init_fit, vmax, vmin)
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
        aux1 = -self._lda * log_q
        aux2 = log_q + aux1
        return safe_diff_exp(log_p + aux1, aux2, 0)
    
    def force_likelihood(self, x):
        raise NotImplementedError('This is a test piece of code.')



class OnlineMProj0(OnlineIProj):

    def __init__(self, log_target, init_fit, gamma, lda=0, vmax=1e5, vmin=1e-10, proxy='discrete_kl'):
        self._gen_init(log_target, init_fit, vmax, vmin)
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

    

class OnlineStarTaylorFit(OnlineIProj):

    def __init__(self, log_target, init_fit, alpha, vmax=1e5, epsilon=1e-5, grad=None, hess_diag=None):
        self._gen_init(log_target, init_fit, vmax, 0)
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


        

  
