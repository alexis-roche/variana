import numpy as np
import scipy.optimize as spo

##from .utils import safe_exp
from .utils import min_methods

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
        
    def _dist(self, w):
        return self._prior * np.exp(np.dot(self._fx, w))

    def dist(self):
        return self._dist(self._w)

    def dual_and_derivatives(self, w):
        d = self._dist(w)        
        z = np.sum(d)
        aux = np.expand_dims(d, 1) * self._fx
        return np.dot(w, self._values) - z + 1, \
            self._values - np.sum(aux, 0), \
            -np.dot(aux.T, self._fx)

    def fit(self, method='newton', maxiter=None, tol=1e-5):
        cache = {'w': None, 'f': None, 'g': None, 'h': None}

        def cost(w):
            if w is cache['w']:
                return cache['f']
            f, g, h = self.dual_and_derivatives(w)
            f, g, h = -f, -g, -h
            cache['w'] = w
            cache['f'] = f
            cache['g'] = g
            cache['h'] = h
            return f
            
        def grad_cost(w):
            if not w is cache['w']:
                f = cost(w)
            return cache['g']

        def grad_hess(w):
            if not w is cache['w']:
                f = cost(w)
            return cache['h']

        m = min_methods[method](self._w, cost, grad_cost, grad_hess,
                                maxiter=maxiter, tol=tol,
                                verbose=VERBOSE)
        self._w = m.argmin()
        
