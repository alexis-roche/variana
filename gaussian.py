"""
A class to represent unnormalized Gaussian distributions.
"""

import numpy as np
from scipy.stats import norm
 
from .utils import (force_tiny, force_finite, hdot, decomp_sym_matrix)



def Z_to_K(Z, dim, detV):
    return Z * force_tiny(detV) ** (-.5)\
        * (2 * np.pi) ** (-.5 * dim)


def K_to_Z(K, dim, detV):
    if detV < 0:
        return np.inf
    return K * (2 * np.pi) ** (.5 * dim) * detV ** .5


def invV_to_theta(invV):
    A = -invV + .5 * np.diag(np.diagonal(invV))
    return A[np.triu_indices(A.shape[0])]


def moments_to_theta(K, m, invV):
    theta2 = invV_to_theta(invV)
    theta1 = np.dot(invV, m)
    theta0 = np.log(K) - .5 * np.dot(m, theta1)
    return np.concatenate((np.array((theta0,)), theta1, theta2))


def theta_to_invV(theta):
    dim = int(-1 + np.sqrt(1 + 8 * theta.size)) / 2
    A = np.zeros([dim, dim])
    A[np.triu_indices(dim)] = theta
    return -(A + A.T)


def _sample_dim(dim):
    return int(-1.5 + np.sqrt(.25 + 2 * dim))


def silver_section(dim):
    aux = .5 * ( 1 + (1 / (2 * dim + 1)) ** (1 / dim))
    return np.exp(- .5 * norm.ppf(aux) ** 2) / (np.sqrt(2 * np.pi) * (1 - aux))


def _quad3(m, sqrtV, rule):
    """
    Return point and weight arrays with respective shapes (dim, 2*dim+1) and (2*dim+1,)
    """
    dim = len(m)
    npts = 2 * dim + 1
    # create output arrays
    xs = np.zeros((dim, npts))
    # compute weights
    if rule == 'balanced':
        ws = np.full(npts, 1 / npts)
        shift = silver_section(dim)
    else:
        if rule == 'optimal_d4':
            shift = np.sqrt(3)
        elif rule == 'exact_d3_uniform':
            shift = np.sqrt(dim + .5)
        elif rule == 'exact_d3_positive':
            shift = np.sqrt(dim)
        else:
            shift = rule        
        ws = np.zeros(npts)
        tmp = 1 / (shift ** 2)
        ws[0] = 1 - dim * tmp
        ws[1:] = .5 * tmp
    # compute points
    tmp = shift * sqrtV
    xs.T[...] = m.T
    xs[:, 1:(dim + 1)] += tmp
    xs[:, (dim + 1):] -= tmp
    return xs, ws


class Gaussian(object):
    """
    A class to describe unnormalized Gaussian distributions under the
    form:

    g(x) = K exp[(x-m)'*A*(x-m)] with A = -.5*inv(V)

    theta_dim = (dim * (dim + 1)) / 2 + dim + 1
    """
    def __init__(self, m=None, V=None, K=None, Z=None, theta=None):
        # If theta is provided, ignore other parameters
        if not theta is None:
            self._set_theta(theta)
            self._dim = _sample_dim(len(self._theta))
        else:
            m = np.asarray(m)
            self._dim = m.size
            m = np.reshape(m, (self._dim,))
            V = np.reshape(np.asarray(V), (self._dim, self._dim))
            self._fill_cache_from_moments(m, V, K, Z)
            self._theta = moments_to_theta(self._K, self._m, self._invV)

    def _init_cache(self):
        self._K = None
        self._Z = None
        self._m = None
        self._V = None
        self._invV = None
        self._detV = None
        self._sqrtV = None

    def _fill_cache_from_moments(self, m, V, K=None, Z=None):
        # Mean and variance
        self._m = m
        self._V = V
        # Compute the inverse and the square root of the variance
        # matrix
        abs_s, sign_s, P = decomp_sym_matrix(V)
        self._invV = np.dot(np.dot(P, np.diag(sign_s / abs_s)), P.T)
        self._detV = np.prod(abs_s * sign_s)
        self._sqrtV = np.dot(np.dot(P, np.diag(abs_s ** .5)), P.T)
        # Normalization constant
        if not K is None:
            self._K = float(K)
            self._Z = K_to_Z(self._K, self._dim, self._detV)
        else:
            if Z is None:
                Z = 1.0
            self._K = Z_to_K(Z, self._dim, self._detV)
            self._Z = Z

    def _fill_cache(self):
        """
        Convert theta to K, m, V
        """
        theta = self._theta
        dim = self._dim
        invV = theta_to_invV(theta[(dim + 1):])
        abs_s, sign_s, P = decomp_sym_matrix(invV)
        self._invV = invV
        inv_s = sign_s / abs_s
        self._V = np.dot(np.dot(P, np.diag(inv_s)), P.T)
        self._detV = np.prod(inv_s)
        self._sqrtV = np.dot(np.dot(P, np.diag(abs_s ** - .5)), P.T)
        self._m = np.dot(self._V, theta[1:(dim + 1)])
        self._K = force_finite(force_tiny(np.exp(theta[0] + .5 * hdot(self._m, invV))))
        self._Z = K_to_Z(self._K, self._dim, self._detV)

    @property
    def dim(self):
        return self._dim

    @property
    def K(self):
        # exponential factor
        if self._K is None:
            self._fill_cache()
        return self._K

    @property
    def Z(self):
        # normalizing constant
        if self._Z is None:
            self._fill_cache()
        return self._Z

    @property
    def m(self):
        if self._m is None:
            self._fill_cache()
        return self._m

    @property
    def V(self):
        if self._V is None:
            self._fill_cache()
        return self._V

    @property
    def invV(self):
        if self._invV is None:
            self._fill_cache()
        return self._invV

    @property
    def sqrtV(self):
        if self._sqrtV is None:
            self._fill_cache()
        return self._sqrtV

    def _get_theta(self):
        return self._theta
    
    def _set_theta(self, theta):
        self._theta = np.asarray(theta).squeeze()
        self._init_cache()
      
    theta = property(_get_theta, _set_theta)

    def mahalanobis(self, xs):
        if xs.ndim == 1:
            m = self.m
        else:
            m = np.expand_dims(self.m, -1)
        ys = xs - m
        return np.sum(ys * np.dot(self.invV, ys), 0)

    def log(self, xs):
        return np.log(self.K) - .5 * self.mahalanobis(xs)

    def __call__(self, xs):
        """
        Evaluate the Gaussian at specified points.
        xs must have shape (dim, npts)
        """
        return self.K * np.exp(-.5 * self.mahalanobis(xs))

    def copy(self):
        return self.__class__(theta=self._theta)

    def __rmul__(self, c):
        theta = self._theta.copy()
        theta[0] += np.log(c)
        return self.__class__(theta=theta)

    def normalize(self):
        return (1 / self.Z) * self

    def __mul__(self, other):
        return self.__class__(theta=self._theta + other._theta)
        
    def __truediv__(self, other):
        return self.__class__(theta=self._theta - other._theta)
        
    def __pow__(self, power):
        return self.__class__(theta=power * self._theta)

    def random(self, ndraws=1):
        """
        Return an array with shape (dim, ndraws)
        """
        xs = np.dot(self.sqrtV, np.random.normal(size=(self._dim, ndraws)))
        return (self.m + xs.T).T  # preserves shape

    def quad3(self, rule):
        return _quad3(self.m, self.sqrtV, rule)

    def kl_div(self, other):
        """
        Return the kl divergence D(self, other) where other is another
        Gaussian instance.
        """
        other_Z = other.Z
        if np.isinf(other_Z):
            return np.inf
        Z = self.Z
        dm = self.m - other.m
        dV = np.dot(other.invV, self.V)
        err = -np.log(force_tiny(np.linalg.det(dV)))
        err += np.sum(np.diag(dV)) - dm.size
        err += np.dot(dm.T, np.dot(other.invV, dm))
        err = np.maximum(.5 * err, 0.0)
        z_err = np.maximum(Z * np.log(Z / force_tiny(other_Z)) + other_Z - Z, 0.0)
        return Z * err + z_err

    def integral(self):
        Z = self.Z
        m = self.m
        I1 = Z * m
        I2 = Z * (self.V
                  + np.dot(m.reshape((self._dim, 1)),
                           m.reshape((1, self._dim))))[\
            np.triu_indices(self._dim)]
        return np.concatenate((np.array((Z,)), I1, I2))

    def __str__(self):
        s = 'Gaussian distribution with parameters:\n'
        s += str(self.K) + '\n'
        s += str(self.m) + '\n'
        s += str(self.V) + '\n'
        return s


class GaussianFamily(object):

    def __init__(self, dim):
        self.dim = dim
        self.theta_dim = (dim * (dim + 1)) / 2 + dim + 1

    def design_matrix(self, pts):
        """
        pts: array with shape (dim, npts)
        Returns an array with shape (theta_dim, npts)
        """
        I, J = np.triu_indices(pts.shape[0])
        F = np.array([pts[i, :] * pts[j, :] for i, j in zip(I, J)])
        return np.concatenate((np.ones((1, pts.shape[1])), pts, F))

    def from_integral(self, integral):
        Z = integral[0]
        m = integral[1: (self.dim + 1)] / Z
        V = np.zeros((self.dim, self.dim))
        idx = np.triu_indices(self.dim)
        V[idx] = integral[(self.dim + 1):] / Z
        V.T[np.triu_indices(self.dim)] = V[idx]
        V -= np.dot(m.reshape(m.size, 1), m.reshape(1, m.size))
        return Gaussian(m, V, Z=Z)

    def from_theta(self, theta):
        return Gaussian(theta=theta)

    def check(self, obj):
        return isinstance(obj, Gaussian)



def moments_to_factor_theta(K, m, invv):
    theta2 = -.5 * invv
    theta1 = invv * m
    theta0 = np.log(K) - .5 * np.dot(m, theta1)
    return np.concatenate((np.array((theta0,)), theta1, theta2))

    
class FactorGaussian(Gaussian):

    def __init__(self, m=None, v=None, K=None, Z=None, theta=None):
        if not theta is None:
            self._set_theta(theta)
            self._dim = (len(self._theta) - 1) // 2
        else:
            m = np.asarray(m)
            self._dim = m.size
            m = np.reshape(m, (self._dim,))
            v = np.reshape(np.asarray(v), (self._dim,))
            self._fill_cache_from_moments(m, v, K, Z)
            self._theta = moments_to_factor_theta(self._K, self._m, self._invv)

    def _init_cache(self):
        self._K = None
        self._Z = None
        self._m = None
        self._v = None
        self._invv = None
        self._detV = None

    def _fill_cache_from_moments(self, m, v, K=None, Z=None):
        m = np.asarray(m)
        dim = m.size
        # Mean and variance
        m = np.reshape(m, (dim,))
        v = np.reshape(v, (dim,))
        self._dim = dim
        self._m = m
        self._v = force_tiny(v)
        self._invv = 1 / self._v
        self._detV = np.prod(self._v)
        # Normalization constant
        if not K is None:
            self._K = float(K)
            self._Z = K_to_Z(self._K, self._dim, self._detV)
        else:
            if Z is None:
                Z = 1.0
            self._K = Z_to_K(Z, self._dim, self._detV)
            self._Z = Z

    def _fill_cache(self):
        theta = self._theta
        dim = self._dim
        invv = -2 * theta[(dim + 1):]
        self._invv = force_tiny(invv)
        self._v = 1 / self._invv
        self._m = self._v * theta[1:(dim + 1)]
        self._K = force_finite(force_tiny(np.exp(theta[0] + .5 * np.dot(self._m, self._invv * self._m))))
        self._detV = np.prod(self._v)
        self._Z = K_to_Z(self._K, self._dim, self._detV)

    @property
    def V(self):
        if self._v is None:
            self._fill_cache()
        return np.diag(self._v)

    @property
    def v(self):
        if self._v is None:
            self._fill_cache()
        return self._v

    @property
    def invV(self):
        if self._invv is None:
            self._fill_cache()
        return np.diag(self._invv)

    @property
    def sqrtV(self):
        if self._v is None:
            self._fill_cache()
        return np.diag(np.sqrt(np.abs(self._v)))

    def mahalanobis(self, xs):
        if xs.ndim == 1:
            m = self.m
            invv = self.invv
        else:
            m = np.expand_dims(self.m, -1)
            invv = np.expand_dims(self.invv, -1)
        return np.sum(invv * ((xs - m) ** 2), 0)

    def __str__(self):
        s = 'Factored Gaussian distribution with parameters:\n'
        s += str(self.K) + '\n'
        s += str(self.m) + '\n'
        s += 'diag(' + str(self.v) + ')\n'
        return s

    def embed(self):
        """
        Return equivalent instance of the parent class
        """
        return Gaussian(self.m, self.V, K=self.K)

    def random(self, ndraws=1):
        xs = (np.sqrt(np.abs(self.v)) * \
              np.random.normal(size=(self._dim, ndraws)).T).T
        return (self.m + xs.T).T  # preserves shape

    def quad3(self, rule):
        return _quad3(self.m, self.sqrtV, rule)

    def kl_div(self, other):
        other_Z = other.Z
        if np.isinf(other_Z):
            return np.inf
        Z = self.Z
        dm = self.m - other.m
        dv = other.invv * self.v 
        err = -np.log(force_tiny(np.prod(dv)))
        err += np.sum(dv) - dm.size
        err += np.dot(dm * other._invv, dm)
        err = np.maximum(.5 * err, 0.0)
        z_err = np.maximum(Z * np.log(Z / force_tiny(other_Z)) + other_Z - Z, 0.0)
        return Z * err + z_err
    
    def integral(self):
        Z = self.Z
        m = self.m
        I1 = Z * m
        I2 = Z * (self._v + m ** 2)
        return np.concatenate((np.array((Z,)), I1, I2))

    

class FactorGaussianFamily(object):

    def __init__(self, dim):
        self.dim = dim
        self.theta_dim = 2 * dim + 1

    def design_matrix(self, pts):
        """
        pts: array with shape (dim, npts)
        Returns an array with shape (theta_dim, npts)
        """
        return np.concatenate((np.ones((1, pts.shape[1])), pts,  pts ** 2))

    def from_integral(self, integral):
        Z = integral[0]
        m = integral[1: (self.dim + 1)] / Z
        v = integral[(self.dim + 1):] / Z - m ** 2
        return FactorGaussian(m, v, Z=Z)

    def from_theta(self, theta):
        return FactorGaussian(theta=theta)

    def check(self, obj):
        return isinstance(obj, FactorGaussian)



def as_gaussian(g):
    if isinstance(g, Gaussian) or isinstance(g, FactorGaussian):
        return g
    if len(g) == 2:
        m, V = np.asarray(g[0]), np.asarray(g[1])
    else:
        raise ValueError('input should be a length-2 sequence')
    if V.ndim < 2:
        G = FactorGaussian(m, V)
    elif V.ndim == 2:
        G = Gaussian(m, V)
    else:
        raise ValueError('input variance not understood')
    return G
    

def instantiate_family(key, dim): 
    """
    Instantiate Gaussian family
    """
    if key == 'gaussian':
        return GaussianFamily(dim)
    elif key == 'factor_gaussian':
        return FactorGaussianFamily(dim)
    else:  # if key not in families.keys():
        raise ValueError('unknown family')


def laplace_approximation(m, u, g, h):
    """
    m: approximation point
    u: log function value
    g: gradient of log function
    h: Hessian or Hessian diagonal of log function
    """
    dim = g.shape[-1]
    theta = np.zeros(2 * dim + 1)
    theta[1+dim:] = .5 * h
    aux = h * m
    theta[1:1+dim] = g - aux
    theta[0] = u - np.dot(g, m) + .5 * np.dot(m.T, aux)
    if h.shape == g.shape:
        return FactorGaussian(theta=theta)
    elif h.shape[-2:] == (dim, dim):
        raise ValueError('Full Gaussian Laplace not implemented yet')
    else:
        raise ValueError('unknown family')

