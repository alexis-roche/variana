import numpy as np
import scipy.optimize as spo

from .utils import sdot, min_methods, CachedFunction, TINY, safe_exp

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

    def __init__(self, dim, basis, moments):
        """
        dim is an integer or an array-like reperesenting the prior
        basis is a function of label, data, feature index
        moments is a sequence of moments corresponding to basis
        """
        if isinstance(dim, int):
            self._prior = np.ones(dim)
        else:
            self._prior = np.asarray(prior)
        self._prior /= np.sum(self._prior)
        self._fx = np.array([[1] + [basis(x, i) for i in range(len(moments))] for x in range(len(self._prior))])
        self._moments = np.concatenate(([1.], moments))
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
        return np.dot(w, self._moments) - np.sum(self._dist(w)) + 1

    def gradient_dual(self, w):
        return self._moments - np.sum(self._dist_fx(w), 0)

    def hessian_dual(self, w):
        return -np.dot(self._dist_fx(w).T, self._fx)

    def fit(self, w=None, method='newton', maxiter=None, tol=1e-5):

        def cost(w):
            return -self.dual(w)
            
        def gradient_cost(w):
            return -self.gradient_dual(w)
        
        def hessian_cost(w):
            return -self.hessian_dual(w)
                    
        if not w is None:
            self._w = w
        m = min_methods[method](self._w, cost, gradient_cost, hessian_cost,
                                maxiter=maxiter, tol=tol,
                                verbose=VERBOSE)
        self._w = m.argmin()

    @property
    def weights(self):
        return self._w

        
class MaxentModel(object):

    def __init__(self, dim, basis, moments):
        """
        dim is an integer or an array-like reperesenting the prior
                basis is a function of label, data, feature index
        moments is a sequence of moments corresponding to basis 
        """
        if isinstance(dim, int):
            self._prior = np.ones(dim)
        else:
            self._prior = np.asarray(prior)
        self._prior /= np.sum(self._prior)
        self._fx = np.array([[basis(x, i) for i in range(len(moments))] for x in range(len(self._prior))])
        self._moments = moments
        self._w = np.zeros(len(moments))
        self._udist = CachedFunction(self.__udist)
        self._udist_fx = CachedFunction(self.__udist_fx)
        self._z = CachedFunction(self.__z)
        self._gradient_z = CachedFunction(self.__gradient_z)

    def __udist(self, w):
        udist, norma = safe_exp(np.dot(self._fx, w))
        return self._prior * udist, norma
       
    def __udist_fx(self, w):
        udist, _ = self._udist(w)
        return udist[:, None] * self._fx

    def __z(self, w):
        udist, norma = self._udist(w)
        return np.maximum(np.sum(udist), TINY), norma

    def __gradient_z(self, w):
        return np.sum(self._udist_fx(w), 0)

    def _hessian_z(self, w):
        return np.dot(self._udist_fx(w).T, self._fx)

    def dual(self, w):
        z, norma = self._z(w)
        return np.dot(w, self._moments) - np.log(z) - norma

    def gradient_dual(self, w):
        z, _ = self._z(w)
        return self._moments - self._gradient_z(w) / z

    def hessian_dual(self, w):
        z, _ = self._z(w)
        g = self._gradient_z(w)[:, None]
        return -self._hessian_z(w) / z + np.dot(g, g.T) / (z  ** 2)

    def fit(self, w=None, method='newton', maxiter=None, tol=1e-5):

        def cost(w):
            return -self.dual(w)
            
        def gradient_cost(w):
            return -self.gradient_dual(w)
        
        def hessian_cost(w):
            return -self.hessian_dual(w)

        if not w is None:
            self._w = np.asarray(w)
        m = min_methods[method](self._w, cost, gradient_cost, hessian_cost,
                                maxiter=maxiter, tol=tol,
                                verbose=VERBOSE)
        self._w = m.argmin()
        
    def dist(self):
        udist, _ = self._udist(self._w)
        return udist / np.sum(udist)

    @property
    def weights(self):
        return self._w

    @property
    def prior(self):
        return self._prior


class ConditionalMaxentModel(object):

    def __init__(self, dim, basis, moments, data):
        """
        dim is an integer or an array-like reperesenting the prior
        basis is a function of label, data, feature index
        moments is a sequence of moments corresponding to basis 
        data is array-like with shape (number of examples, number of features)
        """
        if isinstance(dim, int):
            self._prior = np.ones(dim)
        else:
            self._prior = np.asarray(prior)
        self._prior /= np.sum(self._prior)
        self._basis = basis
        self._fxy = np.array([[[basis(x, y, i) for i in range(len(moments))] for x in range(len(self._prior))] for y in data])
        self._moments = moments
        self._w = np.zeros(len(moments))
        self._udist = CachedFunction(self.__udist)
        self._udist_fxy = CachedFunction(self.__udist_fxy)
        self._z = CachedFunction(self.__z)
        self._gradient_z = CachedFunction(self.__gradient_z)

    def __udist(self, w):
        udist, norma = safe_exp(np.dot(self._fxy, w))
        return self._prior * udist, norma
       
    def __udist_fxy(self, w):
        udist, _ = self._udist(w)
        return udist[..., None] * self._fxy

    def __z(self, w):
        udist, norma = self._udist(w)
        return np.maximum(np.sum(udist, 1), TINY), norma

    def __gradient_z(self, w):
        return np.sum(self._udist_fxy(w), 1)

    def _hessian_z(self, w):
        return sdot(np.swapaxes(self._udist_fxy(w), 1, 2), self._fxy)   

    def dual(self, w):
        z, norma = self._z(w)
        return np.dot(w, self._moments) - np.mean(np.log(z)) - norma
        
    def gradient_dual(self, w):
        z, _ = self._z(w)
        g = self._gradient_z(w)
        return self._moments - np.mean(g / z[:, None], 0)

    def hessian_dual(self, w):
        z, _ = self._z(w)
        gn = self._gradient_z(w) / z[:, None]
        Gn2 = sdot(gn[:, :, None], gn[:, None, :])       
        Hn = self._hessian_z(w) / z[:, None, None]
        return np.mean(-Hn + Gn2, 0)
        
    def fit(self, w=None, method='newton', maxiter=None, tol=1e-5):

        def cost(w):
            return -self.dual(w)
            
        def gradient_cost(w):
            return -self.gradient_dual(w)
        
        def hessian_cost(w):
            return -self.hessian_dual(w)

        if not w is None:
            self._w = np.asarray(w)
        m = min_methods[method](self._w, cost, gradient_cost, hessian_cost,
                                maxiter=maxiter, tol=tol,
                                verbose=VERBOSE)
        self._w = m.argmin()
        
    def dist(self, y=None, w=None):
        if w is None:
            w = self._w
        if y is None:
            udist, _ = self._udist(self._w)
            return udist / np.sum(udist, 1)[:, None]            
        fx = np.array([[self._basis(x, y, i) for i in range(len(self._moments))] for x in range(len(self._prior))])
        udist, _ = safe_exp(np.dot(fx, w))
        p = self._prior * udist
        return p / np.maximum(np.sum(p), TINY)

    @property
    def weights(self):
        return self._w

    @property
    def prior(self):
        return self._prior

