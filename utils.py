"""
Constants used in several modules.
Basic implementation of Gauss-Newton gradient descent scheme.
"""
from ._utils import _sdot

from time import time
from warnings import warn
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import fmin_cg, fmin_ncg, fmin_bfgs

_TINY = 1e-10
_HUGE = 1e50

def probe_time(func):
    def wrapper(x):
        t0 = time()
        res = func(x)
        dt = time() - t0
        if res is None:
            return dt
        else:
            return dt, res
    return wrapper


def force_tiny(x):
    return np.maximum(x, _TINY)


def force_finite(x):
    return np.clip(x, -_HUGE, _HUGE)

    
def hdot(x, A):
    return np.dot(x, np.dot(A, x))


def safe_exp(x):
    """
    Returns a tuple (exp(x-xmax), xmax).
    """
    xmax = x.max()
    return np.exp(x - xmax), xmax


def decomp_sym_matrix(A):
    s, P = eigh(A)
    sign_s = 2. * (s >= 0) - 1
    abs_s = force_tiny(np.abs(s))
    return abs_s, sign_s, P


def inv_sym_matrix(A):
    s, P = eigh(A)
    return np.dot(P * (1 / force_tiny(s)), P.T)


def norminf(x):
    return np.max(np.abs(x))


def to_float(x):
    if x is None:
        return None
    else:
        return float(x)


def to_int(x):
    if x is None:
        return None
    else:
        return int(x)


class SteepestDescent(object):

    def __init__(self, x, f, grad_f, hess_f=None,
                 maxiter=None, tol=1e-7,
                 stepsize=1., adaptive=True,
                 proj=None,
                 verbose=False):

        self._generic_init('steepest',
                           x, f, grad_f, None,
                           maxiter, tol,
                           stepsize, adaptive, proj,
                           verbose)
        self.run()

    def _generic_init(self, name,
                      x, f, grad_f, hess_f,
                      maxiter, tol,
                      stepsize, adaptive, proj,
                      verbose):
        self._name = name
        self._x = np.asarray(x).ravel()
        self._f = f
        self._grad_f = grad_f
        self._hess_f = hess_f
        self._fval = self._f(self._x)
        self._init_fval = self._fval
        self._evals = np.array((1, 0, 0))
        self._iterations = 0
        self._maxiter = to_int(maxiter)
        self._tol = to_float(tol)
        self._stepsize = to_float(stepsize)
        self._adaptive = bool(adaptive)
        if proj is None:
            self._proj = lambda x: x
        else:
            self._proj = proj
        self._verbose = verbose

    def direction(self):
        dx = np.nan_to_num(-self._grad_f(self._x))
        self._evals[1] += 1
        return dx

    def run(self):
        self._time = self._run()
        if self._verbose:
            print(self)
    
    @probe_time
    def _run(self):

        stuck = False
        while not stuck:

            x0 = self._x
            fval0 = self._fval
            dx = self.direction()
            norm_dx = norminf(dx)

            # Line search
            a = self._stepsize
            done, lucky = False, False
            while not done:
                stuck = (a * norm_dx) < self._tol
                x = np.nan_to_num(self._proj(x0 + a * dx))
                fval = self._f(x)
                self._evals[0] += 1
                if fval < self._fval:
                    lucky = True
                    self._x = x
                    self._fval = fval
                    if self._adaptive:
                        self._stepsize = a
                    a *= 2
                else:  # no more lucky
                    if lucky:
                        done = True
                    a /= 2
                if stuck:
                    break
                
            # Increase iteration number
            self._iterations += 1
            if not self._maxiter is None:
                if self._iterations > self._maxiter:
                    break

    def argmin(self):
        return self._x

    def __str__(self):
        return 'Optimization complete (%s method).\n' % self._name \
            + ' Initial function value: %f\n' % self._init_fval\
            + ' Final function value: %f\n' % self._fval\
            + ' Iterations: %d\n' % self._iterations\
            + ' Function evaluations: %d\n' % self._evals[0]\
            + ' Gradient evaluations: %d\n' % self._evals[1]\
            + ' Hessian evaluations: %d\n' % self._evals[2]\
            + ' Optimization time: %f sec\n' % self._time\
            + ' Final step size: %f\n' % self._stepsize


class ConjugateDescent(SteepestDescent):

    def __init__(self, x, f, grad_f, hess_f=None,
                 maxiter=None, tol=1e-7,
                 stepsize=1., adaptive=True,
                 proj=None,
                 verbose=False):
        self._generic_init('conjugate',
                           x, f, grad_f, None,
                           maxiter, tol,
                           stepsize, adaptive, proj,
                           verbose)
        self._prev_dx = None
        self._prev_g = None
        self.run()
        
    def direction(self):
        """
        Polak-Ribiere rule. Reset direction if beta < 0 or if
        objective increases along proposed direction.
        """
        g = self._grad_f(self._x)
        self._evals[1] += 1
        if self._prev_dx is None:
            dx = -g
        else:
            b = max(0, np.dot(g, g - self._prev_g) / np.sum(self._prev_g ** 2))
            dx = -g + b * self._prev_dx
            if np.dot(dx, g) > 0:
                dx = -g
        self._prev_g = g
        self._prev_dx = dx
        return np.nan_to_num(dx)


class NewtonDescent(SteepestDescent):

    def __init__(self, x, f, grad_f, hess_f,
                 maxiter=None, tol=1e-7,
                 stepsize=1., adaptive=True,
                 damping=1e-10,
                 proj=None,
                 verbose=False):
        self._generic_init('newton',
                           x, f, grad_f, hess_f,
                           maxiter, tol,
                           stepsize, adaptive, proj,
                           verbose)
        self._damping = to_float(damping)
        self.run()
        
    def direction(self):
        """Compute the gradient g and Hessian H, then solve H dx = -g using
        the Cholesky decomposition: H = L L.T

        Upon failure, add a scalar matrix to the Hessian until the
        Cholesky decomposition works.
        """
        g = self._grad_f(self._x)
        H = self._hess_f(self._x)
        self._evals[1:3] += 1
        Hr = H
        damping = self._damping
        while True:
            try:        
                L, _ = cho_factor(Hr, lower=0)
                dx = -cho_solve((L, 0), g)
                break
            except:
                warn('Ooops... singular Hessian, regularizing')
                Hr = H + damping * np.eye(H.shape[0])
                damping *= 10
        return np.nan_to_num(dx)


class QuasiNewtonDescent(SteepestDescent):

    def __init__(self, x, f, grad_f, hess_f,
                 maxiter=None, tol=1e-7,
                 stepsize=1., adaptive=True,
                 proj=None,
                 verbose=False):
        """
        Assume fix hessian
        """
        self._generic_init('quasi_newton',
                           x, f, grad_f, None,
                           maxiter, tol,
                           stepsize, adaptive, proj,
                           verbose)
        self._hess_inv = inv_sym_matrix(hess_f)
        self.run()
        
    def direction(self):
        g = self._grad_f(self._x)
        self._evals[1] += 1
        dx = -np.dot(self._hess_inv, g)
        return np.nan_to_num(dx)


class ScipyCG(SteepestDescent):

    def __init__(self, x, f, grad_f, hess_f=None,
                 maxiter=None, tol=1e-7, verbose=False):
        self._generic_init('cg', x, f, grad_f, None, maxiter, tol, None, None, None, verbose)
        self.run()

    @probe_time
    def _run(self):
        self._x, self._fval = fmin_cg(self._f, self._x, fprime=self._grad_f,
                                      gtol=self._tol, maxiter=self._maxiter, 
                                      full_output=True, disp=self._verbose)[0:2]

    def __str__(self):
        return '\t Initial function value: %f\n' % self._init_fval\
            + '\t Optimization time: %f sec\n' % self._time
    
        
class ScipyNCG(ScipyCG):
        
    def __init__(self, x, f, grad_f, hess_f=None,
                 maxiter=None, tol=1e-7, verbose=False):
        self._generic_init('ncg', x, f, grad_f, hess_f, maxiter, tol, None, None, None, verbose)
        self._time = self.run()

    @probe_time
    def _run(self):
        self._x, self._fval = fmin_ncg(self._f, self._x, fprime=self._grad_f,
                                       fhess=self._hess_f,
                                       avextol=self._tol, maxiter=self._maxiter, 
                                       full_output=True, disp=self._verbose)[0:2]

        
class ScipyBFGS(ScipyCG):
        
    def __init__(self, x, f, grad_f, hess_f=None,
                 maxiter=None, tol=1e-7, verbose=False):
        self._generic_init('bfgs', x, f, grad_f, None, maxiter, tol, None, None, None, verbose)
        self._time = self.run()

    @probe_time
    def _run(self):
        self._x, self._fval = fmin_bfgs(self._f, self._x, fprime=self._grad_f,
                                        gtol=self._tol, maxiter=self._maxiter, 
                                        full_output=True, disp=self._verbose)[0:2]


def minimizer(name, x, f, grad_f, hess_f=None,
              maxiter=None, tol=1e-7,
              stepsize=1., adaptive=True, proj=None,
              verbose=False):
    """
    name must be one of 'steepest', 'conjugate', 'newton', 'quasi_newton', 'cg', 'ncg', 'bfgs'
    """
    min_obj = {'steepest': SteepestDescent,
               'conjugate': ConjugateDescent,
               'newton': NewtonDescent,
               'quasi_newton': QuasiNewtonDescent,
               'cg': ScipyCG,
               'ncg': ScipyNCG,
               'bfgs': ScipyBFGS}

    if name not in min_obj.keys():
        raise ValueError('unknown minimizer')
    elif name in ('cg', 'ncg', 'bfgs'):
        return min_obj[name](x, f, grad_f, hess_f=hess_f, maxiter=maxiter, tol=tol, verbose=verbose)
    else:
        return min_obj[name](x, f, grad_f, hess_f=hess_f, maxiter=maxiter, tol=tol,
                             stepsize=stepsize, adaptive=adaptive, proj=proj, verbose=verbose)


def approx_gradient(f, x, epsilon):
    """
    Approximate the gradient of a function using central finite
    differences

    Parameters
    ----------
    f: callable
      The function to differentiate
    x: ndarray
      Point where the function gradient is to be evaluated
    epsilon: float
      Stepsize for finite differences

    Returns
    -------
    g: ndarray
      Function gradient at `x`
    """
    npts = 1
    n = x.shape[0]
    if len(x.shape) > 1:
        npts = x.shape[1]
    g = np.zeros((n, npts))
    ei = np.zeros(n)
    for i in range(n):
        ei[i] = .5 * epsilon
        g[i, :] = (f((x.T + ei).T) - f((x.T - ei).T)) / epsilon
        ei[i] = 0
    return g.squeeze()


def approx_hessian_diag(f, x, epsilon):
    """
    Approximate the Hessian diagonal of a function using central
    finite differences

    Parameters
    ----------
    f: callable
      The function to differentiate
    x: ndarray
      Point where the Hessian is to be evaluated
    epsilon: float
      Stepsize for finite differences

    Returns
    -------
    h: ndarray
      Diagonal of the Hessian at `x`
    """
    npts = 1
    n = x.shape[0]
    if len(x.shape) > 1:
        npts = x.shape[1]
    h = np.zeros((n, npts))
    ei = np.zeros(n)
    fx = f(x)
    for i in range(n):
        ei[i] = epsilon
        h[i, :] = (f((x.T + ei).T) + f((x.T - ei).T) - 2 * fx) / (epsilon ** 2)
        ei[i] = 0
    return h.squeeze()


def approx_hessian(f, x, epsilon):
    """
    Approximate the full Hessian matrix of a function using central
    finite differences

    Parameters
    ----------
    f: callable
      The function to differentiate
    x: ndarray
      Point where the Hessian is to be evaluated
    epsilon: float
      Stepsize for finite differences

    Returns
    -------
    H: ndarray
      Hessian matrix at `x`
    """
    npts = 1
    n = x.shape[0]
    if len(x.shape) > 1:
        npts = x.shape[1]
    H = np.zeros((n, n, npts))
    ei = np.zeros(n)
    for i in range(n):
        ei[i] = .5 * epsilon
        g1 = approx_gradient(f, (x.T + ei).T, epsilon)
        g2 = approx_gradient(f, (x.T - ei).T, epsilon)
        H[i, ...] = (g1 - g2) / epsilon
        ei[i] = 0
    return H.squeeze()


class CachedFunction(object):

    def __init__(self, fun):
        self._x = None
        self._f = None
        self._fun = fun

    def __call__(self, x):
        if not self._x is x:
            self._x = x
            self._f = self._fun(x)
        return self._f


def sdot(A, B):
    return _sdot(A.astype(np.double), B.astype(np.double))

