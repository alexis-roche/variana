"""
A class to represent unnormalized Gaussian distributions.
"""

import numpy as np
from scipy.stats import norm
 
from .utils import TINY


GAUSS_CONSTANT = .5 * np.log(2 * np.pi)


def force_tiny(x):
    return np.maximum(x, TINY)


def safe_inv(x):
    sign_x = 1 - 2 * (x < 0)
    aux = x.copy().astype(float)
    msk = np.abs(x) < TINY
    aux[msk] = sign_x[msk] * TINY
    return 1 / aux


def hdot(x, A):
    return np.dot(x, np.dot(A, x))


def logZ_to_logK(logZ, dim, logdetV):
    return logZ - .5 * logdetV - dim * GAUSS_CONSTANT


def logK_to_logZ(logK, dim, logdetV):
    return logK + .5 * logdetV + dim * GAUSS_CONSTANT


def invV_to_theta(invV):
    A = -.5 * invV
    return A[np.triu_indices(A.shape[0])]


def theta_to_invV(theta):
    dim = int(-1 + np.sqrt(1 + 8 * len(theta))) // 2
    A = np.zeros([dim, dim])
    A[np.triu_indices(dim)] = -2 * theta
    I, J = np.triu_indices(dim, 1)
    A[J, I] = A[I, J]
    return A


def sample_dim(tdim):
    return int(-1.5 + np.sqrt(.25 + 2 * tdim))


def theta_dim(dim):
    return (dim * (dim + 1)) // 2 + dim + 1


def silver_section(dim):
    aux = .5 * ( 1 + (1 / (2 * dim + 1)) ** (1 / dim))
    return np.exp(- .5 * norm.ppf(aux) ** 2) / (np.sqrt(2 * np.pi) * (1 - aux))


def quad3(m, sqrtV, rule):
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


def safe_log(x):
    return np.log(force_tiny(x))



class Gaussian(object):
    """
    A class to describe unnormalized Gaussian distributions under the
    form:

    g(x) = K exp[(x-m)'*A*(x-m)] with A = -.5*inv(V)

    theta_dim = (dim * (dim + 1)) / 2 + dim + 1

    If theta is provided, ignore other parameters
    """
    def __init__(self, m=None, V=None, logK=None, logZ=None, theta=None):
        self._init_cache()
        if theta is None:
            m = np.asarray(m)
            self._dim = m.size
            m = np.reshape(m, (self._dim,))
            V = np.reshape(np.asarray(V), (self._dim, self._dim))
            self._fill_cache(m, V, logK, logZ)
            theta2 = invV_to_theta(self._invV)
            theta1 = np.dot(self._invV, self._m)
            theta0 = self._logK - .5 * np.dot(self._m, theta1)
            self._theta = np.concatenate((np.array((theta0,)), theta1, theta2))
        else:
            self._theta = np.asarray(theta).squeeze()
            self._dim = sample_dim(len(self._theta))
            
    def _init_cache(self):
        self._logK = None
        self._logZ = None
        self._m = None
        self._V = None
        self._invV = None
        self._logdetV = None
        self._sqrtV = None

    def _fill_cache(self, m, V, logK, logZ):
        """
        Compute auxiliary quantities: inverse, square root and determinant
        of variance, normalizing constants.
        """
        self._m = m
        self._V = V
        # Compute the inverse and the square root of the variance
        # matrix
        v, P = np.linalg.eigh(V)
        invv = safe_inv(v)
        v = 1 / force_tiny(invv)
        self._invV = np.dot(np.dot(P, np.diag(invv)), P.T)
        self._logdetV = np.sum(np.log(v))
        self._sqrtV = np.dot(np.dot(P, np.diag(np.abs(v) ** .5)), P.T)
        ###
        if not logK is None:
            self._logK = logK
            self._logZ = logK_to_logZ(self._logK, self._dim, self._logdetV)
        else:
            if logZ is None:
                logZ = 0.0
            self._logZ = logZ
            self._logK = logZ_to_logK(self._logZ, self._dim, self._logdetV)

    def _update_cache(self):
        """
        Convert theta to logK, m, V
        """
        self._invV = theta_to_invV(self._theta[(self._dim + 1):])
        invv, P = np.linalg.eigh(self._invV)
        v = 1 / force_tiny(invv)
        self._V = np.dot(np.dot(P, np.diag(v)), P.T)
        self._logdetV = np.sum(np.log(v))
        self._sqrtV = np.dot(np.dot(P, np.diag(np.abs(v) ** .5)), P.T)
        self._m = np.dot(self._V, self._theta[1:(self._dim + 1)])
        self._logK = self._theta[0] + .5 * hdot(self._m, self._invV)
        self._logZ = logK_to_logZ(self._logK, self._dim, self._logdetV)

    @property
    def dim(self):
        return self._dim

    @property
    def logK(self):
        if self._logK is None:
            self._update_cache()
        return self._logK

    @property
    def K(self):
        return np.exp(self.logK)

    @property
    def logZ(self):
        if self._logZ is None:
            self._update_cache()
        return self._logZ

    @property
    def Z(self):
        return np.exp(self.logZ)

    @property
    def m(self):
        if self._m is None:
            self._update_cache()
        return self._m

    @property
    def V(self):
        if self._V is None:
            self._update_cache()
        return self._V

    @property
    def invV(self):
        if self._invV is None:
            self._update_cache()
        return self._invV

    @property
    def sqrtV(self):
        if self._sqrtV is None:
            self._update_cache()
        return self._sqrtV

    @property
    def theta(self):
        return self._theta
    
    def set_theta(self, theta, indices=None):
        if indices is None:
            indices = slice(0, len(self._theta))
        self._theta[indices] = np.asarray(theta).squeeze()
        self._init_cache()
      
    def mahalanobis(self, xs):
        if xs.ndim == 1:
            m = self.m
        else:
            m = np.expand_dims(self.m, -1)
        ys = xs - m
        return np.sum(ys * np.dot(self.invV, ys), 0)

    def log(self, xs):
        return self.logK - .5 * self.mahalanobis(xs)

    def __call__(self, xs):
        """
        Evaluate the Gaussian at specified points.
        xs must have shape (dim, npts)
        """
        return np.exp(self.log(xs))

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
        return quad3(self.m, self.sqrtV, rule)

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
        err = -safe_log(np.linalg.det(dV))
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
                    m.reshape((1, self._dim))))[np.triu_indices(self._dim)]
        return np.concatenate((np.array((Z,)), I1, I2))

    def __str__(self):
        s = 'Gaussian distribution with parameters:\n'
        s += 'K = %f\n' % self.K
        s += 'm = %s\n' % self.m
        s += 'V = %s\n' % self.V
        return s

    def cleanup(self):
        self._init_cache()


class GaussianFamily(object):

    def __init__(self, dim):
        self._dim = dim
        self._theta_dim = (dim * (dim + 1)) / 2 + dim + 1

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
        m = integral[1: (self._dim + 1)] / Z
        V = np.zeros((self._dim, self._dim))
        idx = np.triu_indices(self._dim)
        V[idx] = integral[(self._dim + 1):] / Z
        V.T[np.triu_indices(self._dim)] = V[idx]
        V -= np.dot(m.reshape(m.size, 1), m.reshape(1, m.size))
        return Gaussian(m, V, Z=Z)

    def from_theta(self, theta):
        return Gaussian(theta=theta)

    def check(self, obj):
        return isinstance(obj, Gaussian)

    @property
    def dim(self):
        return self._dim

    @property
    def theta_dim(self):
        return self._theta_dim
    

   
class FactorGaussian(Gaussian):

    def __init__(self, m=None, v=None, logK=None, logZ=None, theta=None):
        self._init_cache()
        if not theta is None:
            self._theta = np.asarray(theta).squeeze()
            self._dim = (len(self._theta) - 1) // 2
        else:
            m = np.asarray(m)
            self._dim = m.size
            m = np.reshape(m, (self._dim,))
            v = np.reshape(np.asarray(v), (self._dim,))
            self._fill_cache(m, v, logK, logZ)
            theta2 = -.5 * self._invv
            theta1 = self._invv * self._m
            theta0 = self._logK - .5 * np.dot(self._m, theta1)
            self._theta = np.concatenate((np.array((theta0,)), theta1, theta2))


            
    def _init_cache(self):
        self._logK = None
        self._logZ = None
        self._m = None
        self._v = None
        self._invv = None
        self._logdetV = None

    def _fill_cache(self, m, v, logK, logZ):
        """
        Compute auxiliary quantities: inverse, square root and determinant
        of variance, normalizing constants.
        """
        m = np.asarray(m)
        dim = m.size
        m = np.reshape(m, (dim,))
        v = np.reshape(v, (dim,))
        self._dim = dim
        self._m = m
        self._invv = safe_inv(v)
        self._v = 1 / force_tiny(self._invv)
        self._logdetV = np.sum(np.log(self._v))
        if not logK is None:
            self._logK = logK
            self._logZ = logK_to_logZ(self._logK, self._dim, self._logdetV)
        else:
            if logZ is None:
                logZ = 0.0
            self._logZ = logZ
            self._logK = logZ_to_logK(self._logZ, self._dim, self._logdetV)

    def _update_cache(self):
        self._invv = force_tiny(-2 * self._theta[(self._dim + 1):])
        self._v = 1 / self._invv
        self._m = self._v * self._theta[1:(self._dim + 1)]
        self._logK = self._theta[0] + .5 * np.dot(self._m, self._invv * self._m)
        self._logdetV = np.sum(np.log(self._v))
        self._logZ = logK_to_logZ(self._logK, self._dim, self._logdetV)

    @property
    def V(self):
        if self._v is None:
            self._update_cache()
        return np.diag(self._v)

    @property
    def v(self):
        if self._v is None:
            self._update_cache()
        return self._v

    @property
    def invv(self):
        if self._invv is None:
            self._update_cache()
        return self._invv

    @property
    def invV(self):
        if self._invv is None:
            self._update_cache()
        return np.diag(self._invv)

    @property
    def sqrtV(self):
        if self._v is None:
            self._update_cache()
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
        s = 'Factor Gaussian distribution with parameters:\n'
        s += 'K = %f\n' % self.K
        s += 'm = %s\n' % self.m
        s += 'diag(V) = %s\n' % self.v
        return s
    
    def embed(self):
        """
        Return equivalent instance of the parent class
        """
        return Gaussian(self.m, self.V, logK=self.logK)

    def random(self, ndraws=1):
        xs = (np.sqrt(np.abs(self.v)) * \
              np.random.normal(size=(self._dim, ndraws)).T).T
        return (self.m + xs.T).T  # preserves shape

    def quad3(self, rule):
        return quad3(self.m, self.sqrtV, rule)

    def kl_div(self, other):
        other_Z = other.Z
        if np.isinf(other_Z):
            return np.inf
        Z = self.Z
        dm = self.m - other.m
        dv = other.invv * self.v 
        err = -np.sum(safe_log(dv))
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

    

class FactorGaussianFamily(GaussianFamily):

    def __init__(self, dim):
        self._dim = dim
        self._theta_dim = 2 * dim + 1

    def design_matrix(self, pts):
        """
        pts: array with shape (dim, npts)
        Returns an array with shape (theta_dim, npts)
        """
        return np.concatenate((np.ones((1, pts.shape[1])), pts,  pts ** 2))

    def from_integral(self, integral):
        Z = integral[0]
        m = integral[1: (self._dim + 1)] / Z
        v = integral[(self._dim + 1):] / Z - m ** 2
        return FactorGaussian(m, v, logZ=safe_log(Z))

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
    dim = len(m)
    if h.shape == g.shape:
        theta = np.zeros(2 * dim + 1)
        theta[(1 + dim):] = .5 * h  
        aux = h * m
        theta[1:(1 + dim)] = g - aux
        theta[0] = u - np.dot(g, m) + .5 * np.dot(m.T, aux)
        return FactorGaussian(theta=theta)
    else:
        theta = np.zeros(theta_dim(dim))
        theta[(1 + dim):] = .5 * h[np.triu_indices(dim)]
        aux = np.dot(h, m)
        theta[1:(1 + dim)] = g - aux
        theta[0] = u - np.dot(g, m) + .5 * np.dot(m.T, aux)
        return Gaussian(theta=theta)
