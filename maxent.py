import numpy as np
import scipy.optimize as spo

##from .utils import safe_exp
from .utils import min_methods, CachedFunction

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
        # (n_labels, n_params)
        dim = len(self._prior)
        tmp = np.array([[f(x) for f in functions] for x in range(dim)])
        self._fx = np.concatenate((np.ones((dim, 1)), tmp), axis=1)
        self._values = np.concatenate(([1.], values))
        self._w = np.zeros(self._fx.shape[1])
        self._dist = CachedFunction(self.__dist)
        self._dist_fx = CachedFunction(self.__dist_fx)

    def __dist(self, w):
        return self._prior * np.exp(np.dot(self._fx, w))
       
    def __dist_fx(self, w):
        return np.expand_dims(self._dist(w), 1) * self._fx

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
        


