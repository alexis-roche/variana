"""
Constants used in several modules.
Basic implementation of Gauss-Newton gradient descent scheme.
"""
from time import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, pinvh
from scipy.optimize import fmin_cg, fmin_ncg, fmin_bfgs, fmin_l_bfgs_b


TINY = 1e-100


def probe_time(func):
    def wrapper(x, *args, **kwargs):
        t0 = time()
        res = func(x, *args, **kwargs)
        dt = time() - t0
        if res is None:
            return dt
        elif isinstance(res, tuple):
            return (dt,) + res
        else:
            return dt, res
    return wrapper



class SteepestDescent(object):

    def __init__(self, f, x, grad_f, hess_f=None,
                 args=(),
                 maxiter=None, tol=1e-5,
                 stepsize=1, adaptive=True,
                 proj=None):

        self._name = self.__class__.__name__
        self._x = np.asarray(x).ravel()
        self._args = args
        self._calls = np.array((0, 0, 0))
        self._iter = 0
        if maxiter is None:
            self._maxiter = None
        else:
            self._maxiter = int(maxiter)
        self._tol = float(tol)
        self._stepsize = float(stepsize)
        self._adaptive = bool(adaptive)
        self._f = f
        self._grad_f = grad_f
        if callable(hess_f):
            self._hess_f = hess_f
            self._hess_f_inv = None
        else:
            if hess_f is None:
                self._hess_f = None
                self._hess_f_inv = None
            else:
                self._hess_f = np.asarray(hess_f)
                self._hess_f_inv = pinvh(hess_f)
        if proj is None:
            self._proj = lambda x: x
        else:
            self._proj = proj
            self._x = proj(self._x)
        self._fval = self._f(self._x, *self._args)
        self._calls[0] += 1
        self._init_fval = self._fval
        self._warnflag = 0
        self._finit()
        self._time = self._run()

    def _finit(self):
        return
        
    def direction(self):
        dx = -self._grad_f(self._x, *self._args)
        self._calls[1] += 1
        if not self._hess_f_inv is None:
            dx = np.dot(self._hess_f_inv, dx)
        return np.nan_to_num(dx)
    
    @probe_time
    def _run(self):

        stuck = False
        while not stuck:
            x0 = self._x
            fval0 = self._fval
            dx = self.direction()
            norm_dx = np.max(np.abs(dx))

            # Line search
            a = self._stepsize
            done, lucky = False, False
            while not done:
                stuck = (a * norm_dx) < self._tol
                x = np.nan_to_num(self._proj(x0 + a * dx))
                fval = self._f(x, *self._args)
                self._calls[0] += 1
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
            self._iter += 1
            if not self._maxiter is None:
                if self._iter > self._maxiter:
                    self._warnflag = 1
                    break

    def argmin(self):
        return self._x

    def info(self):
        return {'method': self._name,
                'initial function value': self._init_fval,
                'final function value': self._fval,
                'iterations': self._iter,
                'function calls': self._calls[0],
                'gradient calls': self._calls[1],
                'hessian calls': self._calls[2],
                'time': self._time,
                'warn flag': self._warnflag,
                'final stepsize': self._stepsize}


class ConjugateDescent(SteepestDescent):

    def _finit(self):
        self._prev_dx = None
        self._prev_g = None
    
    def direction(self):
        """
        Polak-Ribiere rule. Reset direction if beta < 0 or if
        objective increases along proposed direction.
        """
        g = self._grad_f(self._x, *self._args)
        self._calls[1] += 1
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

    def direction(self):
        """
        Compute the gradient g and Hessian H, then solve H dx = -g
        using the Cholesky decomposition: H = L L.T

        If the Hessian is not positive definite, try to get the
        direction from a regularized version of the cost function
        using damping.
        """
        g = self._grad_f(self._x, *self._args)
        H = self._hess_f(self._x, *self._args)
        self._calls[1:3] += 1
        gr = g
        Hr = g
        damping = 0
        while True:
            try:
                L, _ = cho_factor(Hr, lower=0)
                dx = -cho_solve((L, 0), gr)
                break
            except:
                #print('Ooops... singular Hessian')
                damping = 10 * max(TINY, damping)
                gr = g + damping * self._x
                Hr = H + damping * np.eye(len(self._x))
        return np.nan_to_num(dx)


class ScipyCG(object):

    def __init__(self, f, x, grad_f, hess_f=None, args=(),
                 maxiter=None, tol=1e-5, bounds=None, disp=False):
        self._x = np.asarray(x)
        self._f = f
        self._grad_f = grad_f
        self._hess_f = hess_f
        self._args = args
        if maxiter is None:
            self._maxiter = None
        else:
            self._maxiter = int(maxiter)
        self._tol = float(tol)
        self._bounds = bounds
        self._disp = bool(disp)
        self._time = self._run()
        
    @probe_time
    def _run(self):
        aux = fmin_cg(self._f, self._x, fprime=self._grad_f,
                      args=self._args,
                      gtol=self._tol, maxiter=self._maxiter, 
                      full_output=True, disp=self._disp)
        self._x, self._fval, self._fcalls, self._gcalls, self._warnflag = aux
        self._hcalls = 0
        self._name = 'fmin_cg'

    def _info(self):
        return {'method': self._name,
                'final function value': self._fval,
                'function calls': self._fcalls,
                'gradient calls': self._gcalls,
                'hessian calls': self._hcalls,
                'warn flag': self._warnflag,
                'time': self._time}

    def info(self):
        return self._info()
    
    def argmin(self):
        return self._x

        
class ScipyNCG(ScipyCG):
        
    @probe_time
    def _run(self):
        aux = fmin_ncg(self._f, self._x, fprime=self._grad_f,
                       args=self._args,
                       fhess=self._hess_f,
                       avextol=self._tol, maxiter=self._maxiter, 
                       full_output=True, disp=self._disp)
        self._x, self._fval, self._fcalls, self._gcalls, self._hcalls, self._warnflag = aux
        self._name = 'fmin_ncg'


class ScipyBFGS(ScipyCG):
        
    @probe_time
    def _run(self):
        aux = fmin_bfgs(self._f, self._x, fprime=self._grad_f,
                        args=self._args,
                        gtol=self._tol, maxiter=self._maxiter, 
                        full_output=True, disp=self._disp)
        self._x, self._fval, self._gopt, self._Bopt, self._fcalls, self._gcalls, self._warnflag = aux
        self._hcalls = 0
        self._name = 'fmin_bfgs'

    def info(self):
        d = self._info()
        d.update({'gopt': self._gopt, 'Bopt': self._Bopt})
        return d


class ScipyLBFGS(ScipyCG):
        
    @probe_time
    def _run(self):
        if self._maxiter is None:
            self._maxiter = 100000
        aux = fmin_l_bfgs_b(self._f, self._x, fprime=self._grad_f,
                            args=self._args,
                            pgtol=self._tol, maxiter=self._maxiter,
                            bounds=self._bounds,
                            disp=self._disp)
        self._x, self._fval, self._run_info = aux
        self._name = 'fmin_l_bfgs_b'

    def _info(self):
        d = self._run_info
        d.update({'method': self._name, 'time': self._time})
        return d


def squash(x):
    aux = np.unique(x)
    if len(aux) == 1:
        return aux
    return x


def bounds_to_proj(bounds):
    """
    Convert sequence of min-max bounds to a projection function.
    """
    replace_none = lambda x, a: a if x is None else x   
    b0 = squash(np.array([replace_none(b[0], -np.inf) for b in bounds]))
    b1 = squash(np.array([replace_none(b[1], np.inf) for b in bounds]))
    b0_is_inf = (np.max(b0) == -np.inf)
    b1_is_inf = (np.min(b1) == np.inf)
    if b0_is_inf:
        if b1_is_inf:
            proj = lambda x: x
        else:
            proj = lambda x: (x <= b1) * x
    elif b1_is_inf:
        proj = lambda x: (x >= b0) * x
    else:
        proj = lambda x: (x >= b0) * (x <= b1) * x
    return proj


def minimizer(f, x, optimizer, grad_f, hess_f=None,
              args=(),
              maxiter=None, tol=1e-5,
              stepsize=1, adaptive=True,
              bounds=None,
              disp=False):
    """
    optimizer must be one of 'steepest', 'conjugate', 'newton', 'cg', 'ncg', 'bfgs', 'lbfgs'
    """
    min_obj = {'steepest': SteepestDescent,
               'conjugate': ConjugateDescent,
               'newton': NewtonDescent,
               'cg': ScipyCG,
               'ncg': ScipyNCG,
               'bfgs': ScipyBFGS,
               'lbfgs': ScipyLBFGS}

    if optimizer not in min_obj.keys():
        raise ValueError('unknown optimizer')
    local_meth = optimizer in ('steepest', 'conjugate', 'newton')

    if local_meth:
        proj = None
        if not bounds is None:
            if callable(bounds):
                proj = bounds
            else:
                proj = bounds_to_proj(bounds)

        return min_obj[optimizer](f, x, grad_f, hess_f=hess_f,
                             maxiter=maxiter, tol=tol,
                             stepsize=stepsize, adaptive=adaptive,
                             proj=proj)

    if not bounds is None and optimizer != 'lbfgs':
            raise NotImplementedError('%s optimization method does not accept constraints' % optimizer)
        
    return min_obj[optimizer](f, x, grad_f, hess_f=hess_f,
                              maxiter=maxiter, tol=tol,
                              bounds=bounds, disp=disp)



def approx_gradient(f, x, epsilon, args=()):
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
    n = x.shape[0]
    npts = 1
    if len(x.shape) > 1:
        npts = x.shape[1]
    g = np.zeros((n, npts))
    ei = np.zeros(n)
    for i in range(n):
        ei[i] = .5 * epsilon
        g[i, :] = (f((x.T + ei).T, *args) - f((x.T - ei).T, *args)) / epsilon
        ei[i] = 0
    return g.squeeze()


def approx_hessian_diag(f, x, epsilon, args=()):
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
    n = x.shape[0]
    npts = 1
    if len(x.shape) > 1:
        npts = x.shape[1]
    h = np.zeros((n, npts))
    ei = np.zeros(n)
    fx = f(x, *args)
    for i in range(n):
        ei[i] = epsilon
        h[i, :] = (f((x.T + ei).T, *args) + f((x.T - ei).T, *args) - 2 * fx) / (epsilon ** 2)
        ei[i] = 0
    return h.squeeze()


def approx_hessian(f, x, epsilon, args=()):
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
    n = x.shape[0]
    npts = 1
    if len(x.shape) > 1:
        npts = x.shape[1]
    H = np.zeros((n, n, npts))
    ei = np.zeros(n)
    for i in range(n):
        ei[i] = .5 * epsilon
        g1 = approx_gradient(f, (x.T + ei).T, epsilon, args=args)
        g2 = approx_gradient(f, (x.T - ei).T, epsilon, args=args)
        H[i, ...] = np.reshape((g1 - g2) / epsilon, (n, npts))
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
    from ._utils import _sdot
    return _sdot(A.astype(np.double), B.astype(np.double))


# This function returns the memory block address of an array.
def aid(x):
    return x.__array_interface__['data'][0]



def is_sequence(x):
    return hasattr(x, '__len__')


