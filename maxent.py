import numpy as np
import scipy.optimize as spo

from .utils import sdot, min_methods, CachedFunction

VERBOSE = True

"""
Use the generalized KL divergence.

D(p||pi) = int [p log(p/pi) + pi - p]

L(p, w) = D(p||pi) - w (int pf - F)

=> log(p/pi) + 1 - 1 - w f = 0
=> log(p/pi) = w f
=> p_w = pi exp(w f)

Note: f0 = 1 and F0 = 1 by convention.

DUAL FUNCTION

psi(w) = w int p_w f + int pi - int p_w - w int p_w f + w F
=> psi(w) = w F - int p_w + int pi
=> grad_psi(w) = F - int p_w f
=> hess_psi(w) = - int p_w ff'

"""

class MaxentModelGKL(object):

    def __init__(self, dim, functions, values):
        """
        dim is an integer or an array-like reperesenting the prior
        functions is a sequence of functions
        values is an array-like 
        """
        if isinstance(dim, int):
            self._prior = np.ones(dim)
        else:
            self._prior = np.asarray(prior)
        self._prior /= np.sum(self._prior)
        # (n_labels, n_params)
        dim = len(self._prior)
        functions = [lambda x: 1] + [f for f in functions]
        self._fx = np.array([[f(x) for f in functions] for x in range(dim)])
        self._values = np.concatenate(([1.], values))
        self._w = np.zeros(self._fx.shape[-1])
        self._dist = CachedFunction(self.__dist)
        self._dist_fx = CachedFunction(self.__dist_fx)

    def __dist(self, w):
        return self._prior * np.exp(np.dot(self._fx, w))
       
    def __dist_fx(self, w):
        return self._dist(w)[:, None] * self._fx

    def dist(self):
        return self._dist(self._w)
    
    def dual(self, w):
        return np.dot(w, self._values) - np.sum(self._dist(w)) + 1

    def gradient_dual(self, w):
        return self._values - np.sum(self._dist_fx(w), 0)

    def hessian_dual(self, w):
        return -np.dot(self._dist_fx(w).T, self._fx)

    def fit(self, method='newton', maxiter=None, tol=1e-5):
        cache = {'w': None, 'f': None, 'g': None, 'h': None}

        def cost(w):
            return -self.dual(w)
            
        def gradient_cost(w):
            return -self.gradient_dual(w)
        
        def hessian_cost(w):
            return -self.hessian_dual(w)

        m = min_methods[method](self._w, cost, gradient_cost, hessian_cost,
                                maxiter=maxiter, tol=tol,
                                verbose=VERBOSE)
        self._w = m.argmin()


        
class MaxentModel(object):

    def __init__(self, dim, functions, values):
        """
        dim is an integer or an array-like reperesenting the prior
        functions is a sequence of functions
        values is an array-like 
        """
        if isinstance(dim, int):
            self._prior = np.ones(dim)
        else:
            self._prior = np.asarray(prior)
        self._prior /= np.sum(self._prior)
        self._functions = functions
        # (n_labels, n_params)
        dim = len(self._prior)
        self._fx = np.array([[f(x) for f in functions] for x in range(dim)])
        self._values = values
        self._w = np.zeros(len(functions))
        self._udist = CachedFunction(self.__udist)
        self._udist_fx = CachedFunction(self.__udist_fx)
        self._z = CachedFunction(self.__z)
        self._gradient_z = CachedFunction(self.__gradient_z)

    def __udist(self, w):
        return self._prior * np.exp(np.dot(self._fx, w))
       
    def __udist_fx(self, w):
        return self._udist(w)[:, None] * self._fx

    def __z(self, w):
        return np.sum(self._udist(w))

    def __gradient_z(self, w):
        return np.sum(self._udist_fx(w), 0)

    def _hessian_z(self, w):
        return np.dot(self._udist_fx(w).T, self._fx)

    def dual(self, w):
        return np.dot(w, self._values) - np.log(self._z(w))

    def gradient_dual(self, w):
        return self._values - self._gradient_z(w) / self._z(w)

    def hessian_dual(self, w):
        z = self._z(w)
        g = self._gradient_z(w)[:, None]
        return -self._hessian_z(w) / z + np.dot(g, g.T) / (z  ** 2)

    def fit(self, method='newton', maxiter=None, tol=1e-5):
        cache = {'w': None, 'f': None, 'g': None, 'h': None}

        def cost(w):
            return -self.dual(w)
            
        def gradient_cost(w):
            return -self.gradient_dual(w)
        
        def hessian_cost(w):
            return -self.hessian_dual(w)

        m = min_methods[method](self._w, cost, gradient_cost, hessian_cost,
                                maxiter=maxiter, tol=tol,
                                verbose=VERBOSE)
        self._w = m.argmin()
        
    def dist(self):
        return self._udist(self._w) / self._z(self._w)




class ConditionalMaxentModel(object):

    def __init__(self, dim, functions, values, Y):
        """
        dim is an integer or an array-like reperesenting the prior
        functions is a sequence of functions
        values is an array-like
        """
        if isinstance(dim, int):
            self._prior = np.ones(dim)
        else:
            self._prior = np.asarray(prior)
        self._prior /= np.sum(self._prior)
        self._functions = functions
        # (n_y, n_x, n_params)
        dim = len(self._prior)
        self._fxy = np.array([[[f(x, y) for f in functions] for x in range(dim)] for y in Y])
        self._values = values
        self._w = np.zeros(len(functions))
        self._udist = CachedFunction(self.__udist)
        self._udist_fxy = CachedFunction(self.__udist_fxy)
        self._z = CachedFunction(self.__z)
        self._gradient_z = CachedFunction(self.__gradient_z)

    def __udist(self, w):
        return self._prior * np.exp(np.dot(self._fxy, w))
       
    def __udist_fxy(self, w):
        return self._udist(w)[..., None] * self._fxy

    def __z(self, w):
        return np.sum(self._udist(w), 1)

    def __gradient_z(self, w):
        return np.sum(self._udist_fxy(w), 1)

    def _hessian_z(self, w):
        ###return np.array([np.dot(d.T, f) for d, f in zip(self._udist_fxy(w), self._fxy)])
        return sdot(np.swapaxes(self._udist_fxy(w), 1, 2), self._fxy)   

    def dual(self, w):
        return np.dot(w, self._values) - np.mean(np.log(self._z(w)))

    def gradient_dual(self, w):
        z = self._z(w)
        g = self._gradient_z(w)
        return self._values - np.mean(g / z[:, None], 0)

    def hessian_dual(self, w):
        z = self._z(w)
        gn = self._gradient_z(w) / z[:, None]
        ###Gn2 = np.array([np.dot(x, x.T) for x in gn[..., None]])
        Gn2 = sdot(gn[:, :, None], gn[:, None, :])       
        Hn = self._hessian_z(w) / z[:, None, None]
        return np.mean(-Hn + Gn2, 0)
        
    def fit(self, method='newton', maxiter=None, tol=1e-5):
        cache = {'w': None, 'f': None, 'g': None, 'h': None}

        def cost(w):
            return -self.dual(w)
            
        def gradient_cost(w):
            return -self.gradient_dual(w)
        
        def hessian_cost(w):
            return -self.hessian_dual(w)

        m = min_methods[method](self._w, cost, gradient_cost, hessian_cost,
                                maxiter=maxiter, tol=tol,
                                verbose=VERBOSE)
        self._w = m.argmin()
        
    def _dist(self):
        return self._udist(self._w) / self._z(self._w)[..., None]

    def dist(self, y, w=None):
        if w is None:
            w = self._w
        fx = np.array([[f(x, y) for f in self._functions] for x in range(len(self._prior))])
        p = self._prior * np.exp(np.dot(fx, w))
        return p / np.sum(p)
    
