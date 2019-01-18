import numpy as np
import scipy.optimize as spo

from .utils import sdot, minimizer, aid



class MaxentCache(object):

    def __init__(self):
        self.reinit()

    def reinit(self):        
        self._weight = None
        self._udist = None
        self._norma = None
        self._z = None
        self._udist_basis = None
        self._grad_z = None

    def same(self, weight):
        return np.array_equal(weight, self._weight)
        
    def update_z(self, weight, norma, udist, z):
        # If weight has been modified in place, we need to copy it
        if self._weight is None:
            self._weight = weight
        elif aid(weight) == aid(self._weight):
            self._weight = weight.copy()
        else:
            self._weight = weight
        self._weight = weight.copy()
        self._udist = udist
        self._norma = norma
        self._z = z
        self._udist_basis = None
        self._grad_z = None

    def update_grad_z(self, weight, udist_basis, grad_z):
        if not self.same(weight):
            raise ValueError('Cannot run update_grad_z before update_z')
        self._udist_basis = udist_basis
        self._grad_z = grad_z



class Maxent(object):

    def __init__(self, basis, moment, prior=None, tiny=1e-100):
        """
        basis is an array (targets, moments)
        moment is a sequence of mean values corresponding to basis
        prior is None or an array-like reperesenting the prior
        """
        self._init_basis(basis)
        self._init_optimizer(tiny)
        self._init_prior(prior)
        self._init_moment(moment)

    def _init_basis(self, basis):
        self._basis = np.asarray(basis)
        if self._basis.ndim == 1:
            self._basis = self._basis[:, None]

    def _init_optimizer(self, tiny):
        self._tiny = float(tiny)
        self.set_weight(0)
        self._cache = MaxentCache()
        
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

    def _update_z(self, weight):
        if self._cache.same(weight):
            return
        aux = np.dot(self._basis, weight)
        norma = aux.max()
        udist = self._prior * np.exp(aux - norma)
        z = np.sum(udist)
        self._cache.update_z(weight, norma, udist, z)

    def _update_z_and_grad(self, weight):
        self._update_z(weight)
        if not self._cache._grad_z is None:
            return
        udist_basis = self._cache._udist[:, None] * self._basis
        grad_z = np.sum(udist_basis, 0)
        self._cache.update_grad_z(weight, udist_basis, grad_z)

    def dual(self, weight):
        self._update_z(weight)
        return np.dot(weight, self._moment) - np.log(np.maximum(self._cache._z, self._tiny)) - self._cache._norma
        
    def gradient_dual(self, weight):
        self._update_z_and_grad(weight)
        return self._moment - self._cache._grad_z / self._cache._z
        
    def hessian_dual(self, weight):
        self._update_z_and_grad(weight)
        g1 = self._cache._grad_z[:, None] / self._cache._z
        H1 = np.dot(g1, g1.T)
        H2 = np.dot(self._cache._udist_basis.T, self._basis) / self._cache._z
        return H1 - H2
        
    def fit(self, method='lbfgs', positive_weights=False, weight=None, tol=1e-5, maxiter=10000):
        if not weight is None:
            self.set_weight(weight)
        f = lambda weight: -self.dual(weight)
        grad_f = lambda weight: -self.gradient_dual(weight)
        hess_f = lambda weight: -self.hessian_dual(weight)
        bounds = None
        if positive_weights:
            bounds = [(0, None) for i in range(len(self._weight))]
        m = minimizer(method, self._weight, f, grad_f, hess_f, bounds=bounds, tol=tol, maxiter=maxiter)
        self._weight = m.argmin()
        return m.info()
        
    def dist(self):
        self._update_z(self._weight)
        udist = self._cache._udist
        return udist / np.sum(udist)

    def score(self):
        return self.dual(self._weight)

    @property
    def weight(self):
        return self._weight

    def set_weight(self, weight):
        moments = self._basis.shape[-1]
        weight = np.asarray(weight)
        if weight.ndim == 0:
            weight = np.full(moments, float(weight))
        if len(weight) != moments:
            raise ValueError('Inconsistent weight length')
        self._weight = weight

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


def safe_exp_dist(basis, weight, prior):
    aux = np.dot(basis, weight)
    norma = aux.max(1)
    return prior * np.exp(aux - norma[:, None]), norma


def normalize_dist(p, tiny):
    out = np.full(p.shape, 1 / p.shape[1])
    aux = np.sum(p, 1)
    nonzero = aux > tiny
    out[nonzero] = p[nonzero] / aux[nonzero][:, None]
    return out


class ConditionalMaxent(Maxent):

    def __init__(self, basis_generator, moment, data, prior=None, data_weight=None, tiny=1e-100):
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
        self._init_optimizer(tiny)
        self._init_prior(prior)
        self._init_moment(moment)

    def _init_data(self, data, data_weight):
        self._data = reshape_data(data)
        if data_weight is None:
            self._data_weight = None
        else:
            aux = np.asarray(data_weight)
            self._data_weight = aux / aux.sum()
        if self._data_weight is None:
            self._sample_mean = lambda x: np.mean(x, 0)
        else:
            ###self._sample_mean = lambda x: np.sum(self._data_weight.reshape([x.shape[0]] + [1] * (len(x.shape) - 1)) * x, 0)
            self._sample_mean = lambda x: np.sum(self._data_weight.reshape([x.shape[0]] + [1] * (x.ndim - 1)) * x, 0)

    def _init_basis(self, basis_generator):
        self._basis_generator = basis_generator
        self._basis = basis_generator(self._data)

    def _update_z(self, weight):
        if self._cache.same(weight):
            return
        udist, norma = safe_exp_dist(self._basis, weight, self._prior)
        z = np.sum(udist, 1)
        self._cache.update_z(weight, norma, udist, z)

    def _update_z_and_grad(self, weight):
        self._update_z(weight)
        if not self._cache._grad_z is None:
            return
        udist_basis = self._cache._udist[..., None] * self._basis
        grad_z = np.sum(udist_basis, 1)
        self._cache.update_grad_z(weight, udist_basis, grad_z)

    def dual(self, weight):
        self._update_z(weight)
        return np.dot(weight, self._moment) - self._sample_mean(np.log(np.maximum(self._cache._z, self._tiny)) + self._cache._norma)

    def gradient_dual(self, weight):
        self._update_z_and_grad(weight)
        return self._moment - self._sample_mean(self._cache._grad_z / self._cache._z[:, None])
        
    def hessian_dual(self, weight):
        self._update_z_and_grad(weight)      
        g1 = self._cache._grad_z / self._cache._z[:, None]
        H1 = sdot(g1[:, :, None], g1[:, None, :])
        H2 = sdot(np.swapaxes(self._cache._udist_basis, 1, 2), self._basis) / self._cache._z[:, None, None]
        return self._sample_mean(H1 - H2)
    
    def dist(self, data=None, weight=None):
        if weight is None:
            weight = self._weight
        if data is None:
            self._update_z(weight)
            return normalize_dist(self._cache._udist, self._tiny)
        p, _ = safe_exp_dist(self._basis_generator(reshape_data(data)), weight, self._prior)
        return normalize_dist(p, self._tiny)
    
    @property
    def data(self):
        return self._data

    @property
    def data_weight(self):
        if self._data_weight is None:
            return np.full(self._data.shape[0], 1 / self._data.shape[0])
        return self._data_weight

    

    
#########################################################################
# Maxent classifier
#########################################################################

class MaxentClassifier(ConditionalMaxent):

    def __init__(self, data, target, basis_generator, prior=None, tiny=1e-100):
        """
        data (n, n_features)
        target (n, )
        Use empirical moments
        """
        self._init_dataset(data, target, prior)
        self._init_basis(basis_generator)
        self._init_optimizer(tiny)
        self._init_moment()

    def _init_dataset(self, data, target, prior):
        # Set prior and weight data accordingly
        self._target = np.asarray(target)
        targets = self._target.max() + 1
        count = np.array([np.sum(self._target == x) for x in range(targets)])
        data_weighting = True
        if prior is None:
            prior = np.ones(targets)
        elif prior == 'empirical':
            prior = count / len(self._target)
            data_weighting = False
        self._init_prior(prior)
        if data_weighting:
            data_weight = (self._prior / count)[target]
        else:
            data_weight = None
        self._init_data(data, data_weight)

    def _init_moment(self):
        # Use empirical moments
        self._moment = self._sample_mean(self._basis[range(self._basis.shape[0]), self._target])


#########################################################################
# Bayesian composite inference
#########################################################################

GAUSS_CONSTANT = .5 * np.log(2 * np.pi)


def log_lik1d(z, m, s, tiny=1e-100, big=1000):
    s = np.maximum(s, tiny)
    return np.clip(-(GAUSS_CONSTANT + np.log(s) + .5 * ((z - m) / s) ** 2), -big, big)


def mean_log_lik1d(s, tiny):
    s = np.maximum(s, tiny)
    return -(GAUSS_CONSTANT + np.log(s) + .5)


class GaussianCompositeInference(MaxentClassifier):

    def __init__(self, data, target, prior=None, homo_sced=0, ref_class=None, tiny=1e-100, max_log=1000):
        """
        data (n, n_features)
        target (n, )
        """
        self._homo_sced = max(0, min(1, float(homo_sced)))
        self._ref_class = None
        if not ref_class is None:
            self._ref_class = int(ref_class)
        self._max_log = float(max_log)
        self._init_dataset(data, target, prior)        
        self._init_training()
        self._init_basis(self._make_basis_generator(float(tiny), float(max_log)))
        self._init_optimizer(tiny)
        self._init_moment()
        
    def _init_training(self):
        # Pre-training: feature-based ML parameter estimates
        targets = len(self._prior)
        self._means = np.array([np.mean(self._data[self._target == x], 0) for x in range(targets)])
        res2 = (self._data - self._means[self._target]) ** 2
        var = np.array([np.mean(res2[self._target == x], 0) for x in range(targets)])
        if self._homo_sced > 0:
            var = self._homo_sced * np.sum(self._prior[:, None] * var, 0) + (1 - self._homo_sced) * var
        self._devs = np.sqrt(var)

    def _make_basis_generator(self, tiny, max_log):
        targets = len(self._prior)
        zob = lambda z, m, s: log_lik1d(z, m, s, tiny, max_log)
        def basis_generator(data):
            examples, features = data.shape
            out = np.zeros((examples, targets, features))
            for x in range(targets):
                out[:, x, :] = zob(data, self._means[x], self._devs[x])
            return out
        def basis_generator_super(data):
            examples, features = data.shape
            out = np.zeros((examples, targets, targets * features))
            ll_ref = zob(data, self._means[self._ref_class], self._devs[self._ref_class])
            for x in range(targets):
                n_x = features * x
                out[:, x, n_x:(n_x + features)] = zob(data, self._means[x], self._devs[x]) - ll_ref
            return out
        if self._ref_class is None:
            return basis_generator
        else:
            return basis_generator_super
    
    def _check_moment(self):
        # Faster than empirical mean log-likelihood values but does
        # not work if super composite and homoscedastic
        moment = self._prior[:, None] * np.array([mean_log_lik1d(self._devs[x]) for x in range(len(self._prior))])
        if self._ref_class is None:
            moment = moment.sum(0)
        else:
            moment = moment.ravel()
        return moment

    @property
    def class_weight(self):
        if self._ref_class is None:
            return self._weight
        else:
            return self._weight.reshape((len(self._prior), self._data.shape[1]))


#########################################################################
# Logistic regression
#########################################################################

class LogisticRegression(MaxentClassifier):

    def __init__(self, data, target, prior=None, tiny=1e-100):
        self._init_dataset(data, target, prior)
        self._init_basis(self._make_basis_generator())
        self._init_optimizer(tiny)
        self._init_moment()

    def _make_basis_generator(self):
        targets = len(self._prior)
        def basis_generator(data):
            examples, features = data.shape
            n = features + 1
            out = np.zeros((examples, targets, n * targets))
            for x in range(1, targets):
                nx = n * x
                out[:, x, nx] = 1
                out[:, x, (nx + 1):(nx + n)] = data
            return out
        return basis_generator

    @property
    def class_weight(self):
        return self._weight.reshape((len(self._prior), 1 + self._data.shape[1]))


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
        self._weight = np.zeros(self._fx.shape[-1])

    def _dist(self, weight):
        return self._prior * np.exp(np.dot(self._fx, weight))
       
    def _dist_fx(self, weight):
        return self._dist(weight)[:, None] * self._fx

    def dist(self):
        return self._dist(self._weight)
    
    def dual(self, weight):
        return np.dot(weight, self._moment) - np.sum(self._dist(weight)) + 1

    def gradient_dual(self, weight):
        return self._moment - np.sum(self._dist_fx(weight), 0)

    def hessian_dual(self, weight):
        return -np.dot(self._dist_fx(weight).T, self._fx)

    def fit(self, method='lbfgs', weight=None):

        def cost(weight):
            return -self.dual(weight)
            
        def gradient_cost(weight):
            return -self.gradient_dual(weight)
        
        def hessian_cost(weight):
            return -self.hessian_dual(weight)
                    
        if not weight is None:
            self._weight = np.asarray(weight)
        m = minimizer(method, self._weight, cost, gradient_cost, hessian_cost)
        self._weight = m.argmin()
        
    @property
    def weight(self):
        return self._weight

    @property
    def prior(self):
        return self._prior
