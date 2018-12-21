import numpy as np
import scipy.optimize as spo

from .utils import sdot, minimizer, CachedFunction, force_tiny



class Maxent(object):

    def __init__(self, basis, moment, prior=None):
        """
        basis is an array (targets, moments)
        moment is a sequence of mean values corresponding to basis
        prior is None or an array-like reperesenting the prior
        """
        self._init_basis(basis)
        self._init_optimizer()
        self._init_prior(prior)
        self._init_moment(moment)

    def _init_basis(self, basis):
        self._basis = np.asarray(basis)
        if self._basis.ndim == 1:
            self._basis = self._basis[:, None]

    def _init_optimizer(self):
        moments = self._basis.shape[-1]
        self._lda = np.zeros(moments)
        self._udist = CachedFunction(self.__udist)
        self._udist_basis = CachedFunction(self.__udist_basis)
        self._z = CachedFunction(self.__z)
        self._gradient_z = CachedFunction(self.__gradient_z)

    def _init_prior(self, prior):
        if prior is None:
            self._prior = np.ones(self._basis.shape[-2])
        else:
            self._prior = np.asarray(prior)
        self._prior /= np.sum(self._prior)        

    def _init_moment(self, moment):
        self._moment = np.asarray(moment)
        if self._moment.ndim == 0:
            self._moment = self._moment[None]

    def __udist(self, lda):
        aux = np.dot(self._basis, lda)
        norma = aux.max()
        return self._prior * np.exp(aux - norma), norma
        
    def __udist_basis(self, lda):
        return self._udist(lda)[0][:, None] * self._basis
            
    def __z(self, lda):
        udist, norma = self._udist(lda)
        return np.sum(udist), norma

    def __gradient_z(self, lda):
        return np.sum(self._udist_basis(lda), 0)
    
    def _hessian_z(self, lda):
        return np.dot(self._udist_basis(lda).T, self._basis)
        
    def dual(self, lda):
        z, norma = self._z(lda)
        return np.dot(lda, self._moment) - np.log(force_tiny(z)) - norma
        
    def gradient_dual(self, lda):
        z, _ = self._z(lda)
        return self._moment - self._gradient_z(lda) / z
        
    def hessian_dual(self, lda):
        z, _ = self._z(lda)
        g = self._gradient_z(lda)[:, None]
        return -self._hessian_z(lda) / z + np.dot(g, g.T) / (z  ** 2)

    def fit(self, method='newton', positive_weights=False, weight=None, **kwargs):
        if not weight is None:
            self._lda = np.asarray(weight)
        f = lambda lda: -self.dual(lda)
        grad_f = lambda lda: -self.gradient_dual(lda)
        hess_f = lambda lda: -self.hessian_dual(lda)
        if positive_weights:
            proj = lambda lda: (lda >= 0) * lda
            self._lda = proj(self._lda)
        else:
            proj = None
        m = minimizer(method, self._lda, f, grad_f, hess_f, proj=proj, **kwargs)
        self._lda = m.argmin()
        self._optimizer = m

    def dist(self):
        udist, _ = self._udist(self._lda)
        return udist / np.sum(udist)

    def score(self):
        return self.dual(self._lda)

    @property
    def weight(self):
        return self._lda

    @property
    def prior(self):
        return self._prior



def reshape_data(data):
    """
    Try to convert the input into a 2d array with shape (n_points, n_features)
    """
    out = np.asarray(data)
    if out.ndim == 0:
        out = np.reshape(out, (1, 1))
    elif out.ndim == 1:
        out = out[:, None]
    elif out.ndim > 2:
        raise ValueError('Cannot process input data')
    return out


def safe_exp_dist(fxy, lda, prior):
    aux = np.dot(fxy, lda)
    norma = aux.max(1)
    return prior * np.exp(aux - norma[:, None]), norma


def normalize_dist(p, tiny=1e-25):
    out = np.full(p.shape, 1 / p.shape[1])
    aux = np.sum(p, 1)
    nonzero = aux > tiny
    out[nonzero] = p / aux[nonzero][:, None]
    return out


class ConditionalMaxent(Maxent):

    def __init__(self, basis_generator, moment, data, prior=None, data_weight=None):
        """
        basis_generator is a function that takes an array of data with
        shape (examples, features) and returns an array with shape
        (examples, targets, moments)
        moment is a sequence of mean values corresponding to basis 
        data is a sequence of length equal to the number of examples
        prior is None or an array-like reperesenting the prior

        """
        self._init_data(data, data_weight)
        self._init_basis(basis_generator)
        self._init_optimizer()
        self._init_prior(prior)
        self._init_moment(moment)

    def _init_data(self, data, data_weight):
        self._data = reshape_data(data)
        if data_weight is None:
            self._w = None
        else:
            aux = np.asarray(data_weight)
            self._w = aux / aux.sum()
        if self._w is None:
            self._sample_mean = lambda x: np.mean(x, 0)
        else:
            self._sample_mean = lambda x: np.sum(self._w.reshape([x.shape[0]] + [1] * (len(x.shape) - 1)) * x, 0)

    def _init_basis(self, basis_generator):
        self._basis_generator = basis_generator
        self._basis = basis_generator(self._data)
        
    def _init_optimizer(self):
        moments = self._basis.shape[-1]
        self._lda = np.zeros(moments)
        self._udist = CachedFunction(self.__udist)
        self._udist_basis = CachedFunction(self.__udist_basis)
        self._z = CachedFunction(self.__z)
        self._gradient_z = CachedFunction(self.__gradient_z)

    def __udist(self, lda):
        return safe_exp_dist(self._basis, lda, self._prior)

    def __udist_basis(self, lda):
        return self._udist(lda)[0][..., None] * self._basis

    def __z(self, lda):
        udist, norma = self._udist(lda)
        return np.sum(udist, 1), norma

    def __gradient_z(self, lda):
        return np.sum(self._udist_basis(lda), 1)

    def _hessian_z(self, lda):
        return sdot(np.swapaxes(self._udist_basis(lda), 1, 2), self._basis)   
    
    def dual(self, lda):
        z, norma = self._z(lda)
        return np.dot(lda, self._moment) - self._sample_mean(np.log(force_tiny(z)) + norma)
        
    def gradient_dual(self, lda):
        z, _ = self._z(lda)
        g = self._gradient_z(lda)
        return self._moment - self._sample_mean(g / z[:, None])

    def hessian_dual(self, lda):
        z, _ = self._z(lda)
        gn = self._gradient_z(lda) / z[:, None]
        H1 = sdot(gn[:, :, None], gn[:, None, :])       
        H2 = self._hessian_z(lda) / z[:, None, None]
        return self._sample_mean(-H2 + H1)

    def dist(self, data=None, weight=None):
        if weight is None:
            weight = self._lda
        if data is None:
            return normalize_dist(self._udist(weight)[0])
        fxy = self._basis_generator(reshape_data(data))
        p, _ = safe_exp_dist(fxy, weight, self._prior)
        return normalize_dist(p)
    
    @property
    def data(self):
        return self._data

    @property
    def data_weight(self):
        if self._w is None:
            return np.full(self._data.shape[0], 1 / self._data.shape[0])
        return self._w

    

    
#########################################################################
# Maxent classifier
#########################################################################

class MaxentClassifier(ConditionalMaxent):

    def __init__(self, data, target, basis_generator, prior=None):
        """
        data (n, n_features)
        target (n, )
        Use empirical moments
        """
        self._init_dataset(data, target, prior)
        self._init_basis(basis_generator)
        self._init_optimizer()
        self._init_moment()

    def _init_dataset(self, data, target, prior):
        # Set prior and weight data accordingly
        self._target = np.asarray(target)
        targets = self._target.max() + 1
        prop = np.array([np.mean(self._target == x) for x in range(targets)])
        data_weighting = True
        if prior is None:
            prior = np.ones(targets)
        elif prior == 'empirical':
            prior = prop
            data_weighting = False
        self._init_prior(prior)
        if data_weighting:
            data_weight = (self._prior / prop)[target]
        else:
            data_weight = None
        self._init_data(data, data_weight)

    def _init_moment(self):
        # Use empirical moments
        self._moment = np.sum(self._w[:, None] * self._basis[range(self._basis.shape[0]), self._target], 0)


#########################################################################
# Bayesian composite inference
#########################################################################

GAUSS_CONSTANT = .5 * np.log(2 * np.pi)


def log_lik1d(z, m, s):
    s = force_tiny(s)
    return -(GAUSS_CONSTANT + np.log(s) + .5 * ((z - m) / s) ** 2)


def mean_log_lik1d(s):
    s = force_tiny(s)
    return -(GAUSS_CONSTANT + np.log(s) + .5)


class GaussianCompositeInference(MaxentClassifier):

    def __init__(self, data, target, prior=None, supercomposite=False, homoscedastic=False):
        """
        data (n, n_features)
        target (n, )
        """
        self._homoscedastic = bool(homoscedastic)
        self._supercomposite = bool(supercomposite)
        self._init_dataset(data, target, prior)        
        self._init_training()
        self._init_basis(self._make_basis_generator())
        self._init_optimizer()
        self._init_moment()
        
    def _init_training(self):
        # Pre-training: feature-based ML parameter estimates
        targets = len(self._prior)
        means = np.array([np.mean(self._data[self._target == x], 0) for x in range(targets)])
        res2 = (self._data - means[self._target]) ** 2
        if self._homoscedastic:
            devs = np.repeat(np.sqrt(self._sample_mean(res2))[None, :], targets, axis=0)
        else:
            devs = np.array([np.sqrt(np.mean(res2[self._target == x], 0)) for x in range(targets)])
        self._means = means
        self._devs = devs

    def _make_basis_generator(self):
        targets = len(self._prior)
        def basis_generator(data):
            examples, features = data.shape
            out = np.zeros((examples, targets, features))
            for x in range(targets):
                out[:, x, :] = log_lik1d(data, self._means[x], self._devs[x])
            return out
        def basis_generator_super(data):
            examples, features = data.shape
            out = np.zeros((examples, targets, targets * features))
            for x in range(targets):
                n_x = features * x
                out[:, x, n_x:(n_x + features)] = log_lik1d(data, self._means[x], self._devs[x])
            return out
        if self._supercomposite:
            return basis_generator_super
        return basis_generator
    
    def _check_moment(self):
        # Faster than empirical mean log-likelihood values but does
        # not work if super composite and homoscedastic
        moment = self._prior[:, None] * np.array([mean_log_lik1d(self._devs[x]) for x in range(len(self._prior))])
        if self._supercomposite:
            moment = moment.ravel()
        else:
            moment = moment.sum(0)
        return moment


#########################################################################
# Logistic regression
#########################################################################

class LogisticRegression(MaxentClassifier):

    def __init__(self, data, target, prior=None):
        self._init_dataset(data, target, prior)
        self._init_basis(self._make_basis_generator())
        self._init_optimizer()
        self._init_moment()

    def _make_basis_generator(self):
        targets = len(self._prior)
        def basis_generator(data):
            examples, features = data.shape
            n = features + 1
            out = np.zeros((examples, targets, n * targets))
            for x in range(targets):
                nx = n * x
                out[:, x, nx] = 1
                out[:, x, (nx + 1):(nx + n)] = data
            return out
        return basis_generator



##################################
# Obsolete, kept for testing
##################################

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

class MaxentGKL(object):

    def __init__(self, targets, basis, moment):
        """
        targets is an integer or an array-like reperesenting the prior
        basis is a function of label, data, feature index
        moment is a sequence of moment corresponding to basis
        """
        if isinstance(targets, int):
            self._prior = np.ones(targets)
        else:
            self._prior = np.asarray(prior)
        self._prior /= np.sum(self._prior)
        self._fx = np.array([[1] + [basis(x, i) for i in range(len(moment))] for x in range(len(self._prior))])
        self._moment = np.concatenate(([1.], moment))
        self._lda = np.zeros(self._fx.shape[-1])
        self._dist = CachedFunction(self.__dist)
        self._dist_fx = CachedFunction(self.__dist_fx)

    def __dist(self, lda):
        return self._prior * np.exp(np.dot(self._fx, lda))
       
    def __dist_fx(self, lda):
        return self._dist(lda)[:, None] * self._fx

    def dist(self):
        return self._dist(self._lda)
    
    def dual(self, lda):
        return np.dot(lda, self._moment) - np.sum(self._dist(lda)) + 1

    def gradient_dual(self, lda):
        return self._moment - np.sum(self._dist_fx(lda), 0)

    def hessian_dual(self, lda):
        return -np.dot(self._dist_fx(lda).T, self._fx)

    def fit(self, method='newton', weight=None, **kwargs):

        def cost(lda):
            return -self.dual(lda)
            
        def gradient_cost(lda):
            return -self.gradient_dual(lda)
        
        def hessian_cost(lda):
            return -self.hessian_dual(lda)
                    
        if not weight is None:
            self._lda = np.asarray(lda)
        m = minimizer(method, self._lda, cost, gradient_cost, hessian_cost, **kwargs)
        self._lda = m.argmin()
        
    @property
    def weight(self):
        return self._lda

    @property
    def prior(self):
        return self._prior
