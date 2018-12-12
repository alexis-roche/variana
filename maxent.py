import numpy as np
import scipy.optimize as spo

from .utils import sdot, minimizer, CachedFunction, force_tiny, safe_exp



def aval_cochonne(basis, targets, moments):
    """
    Output has shape targets, moments
    """
    out = np.array([[basis(x, i)\
                     for i in range(moments)]\
                    for x in range(targets)])
    return out.reshape(out.shape[0:2])



class Maxent(object):

    def __init__(self, targets, basis, moment):
        """
        targets is an integer or an array-like reperesenting the prior
        basis is a function of label, data, feature index
        moment is a sequence of mean values corresponding to basis 
        """
        self._init_prior(targets)
        self._init_basis(basis)
        self._init_moment(moment)
        self._init_optimizer()

    def _init_prior(self, targets):
        try:
            self._prior = np.ones(int(targets))
        except:
            self._prior = np.asarray(targets)
        self._prior /= np.sum(self._prior)

    def _init_basis(self, basis):
        self._basis = basis

    def _init_moment(self, moment):
        self._moment = np.asarray(moment)

    def _init_optimizer(self):
        self._lda = np.zeros(len(self._moment))
        self._fx = aval_cochonne(self._basis, len(self._prior), len(self._moment))
        self._udist = CachedFunction(self.__udist)
        self._udist_fx = CachedFunction(self.__udist_fx)
        self._z = CachedFunction(self.__z)
        self._gradient_z = CachedFunction(self.__gradient_z)
        
    def __udist(self, lda):
        udist, norma = safe_exp(np.dot(self._fx, lda))
        return self._prior * udist, norma
       
    def __udist_fx(self, lda):
        udist, _ = self._udist(lda)
        return udist[:, None] * self._fx

    def __z(self, lda):
        udist, norma = self._udist(lda)
        return force_tiny(np.sum(udist)), norma

    def __gradient_z(self, lda):
        return np.sum(self._udist_fx(lda), 0)

    def _hessian_z(self, lda):
        return np.dot(self._udist_fx(lda).T, self._fx)

    def dual(self, lda):
        z, norma = self._z(lda)
        return np.dot(lda, self._moment) - np.log(z) - norma

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

    @property
    def weight(self):
        return self._lda

    @property
    def prior(self):
        return self._prior


def eval_basis(basis, data, targets, moments):
    """
    Output has shape points, targets, moments
    """
    out = np.array([[[basis(x, y, i)\
                      for i in range(moments)]\
                     for x in range(targets)]\
                    for y in data])
    return out.reshape(out.shape[0:3])


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


class ConditionalMaxent(Maxent):

    def __init__(self, targets, basis, moment, data, data_weight=None):
        """
        targets is an integer or an array-like reperesenting the prior
        basis is a function of label, data, feature index
        moment is a sequence of mean values corresponding to basis 
        data is a sequence of length equal to the number of examples
        """
        self._init_prior(targets)
        self._init_basis(basis)
        self._init_moment(moment)
        self._init_data(data, data_weight)
        self._init_optimizer()

    def _init_data(self, data, data_weight):
        self._data = reshape_data(data)
        if data_weight is None:
            self._w = None
        else:
            aux = np.asarray(data_weight)
            self._w = aux / aux.sum()

    def _init_optimizer(self):
        self._lda = np.zeros(len(self._moment))
        self._fxy = eval_basis(self._basis, self._data, len(self._prior), len(self._moment))
        self._udist = CachedFunction(self.__udist)
        self._udist_fxy = CachedFunction(self.__udist_fxy)
        self._z = CachedFunction(self.__z)
        self._gradient_z = CachedFunction(self.__gradient_z)
        if self._w is None:
            self._sample_mean = lambda x: np.mean(x, 0)
        else:
            self._sample_mean = lambda x: np.sum(self._w.reshape([x.shape[0]] + [1] * (len(x.shape) - 1)) * x, 0)

    def __udist(self, lda):
        udist, norma = safe_exp(np.dot(self._fxy, lda))
        return self._prior * udist, norma
       
    def __udist_fxy(self, lda):
        udist, _ = self._udist(lda)
        return udist[..., None] * self._fxy

    def __z(self, lda):
        udist, norma = self._udist(lda)
        return force_tiny(np.sum(udist, 1)), norma

    def __gradient_z(self, lda):
        return np.sum(self._udist_fxy(lda), 1)

    def _hessian_z(self, lda):
        return sdot(np.swapaxes(self._udist_fxy(lda), 1, 2), self._fxy)   
    
    def dual(self, lda):
        z, norma = self._z(lda)
        return np.dot(lda, self._moment) - self._sample_mean(np.log(z)) - norma
        
    def gradient_dual(self, lda):
        z, _ = self._z(lda)
        g = self._gradient_z(lda)
        return self._moment - self._sample_mean(g / z[:, None])

    def hessian_dual(self, lda):
        z, _ = self._z(lda)
        gn = self._gradient_z(lda) / z[:, None]
        Gn2 = sdot(gn[:, :, None], gn[:, None, :])       
        Hn = self._hessian_z(lda) / z[:, None, None]
        return self._sample_mean(-Hn + Gn2)

    def dist(self, data=None, weight=None):
        if weight is None:
            weight = self._lda
        if data is None:
            udist, _ = self._udist(self._lda)
            return udist / np.sum(udist, 1)[:, None]
        data = reshape_data(data)
        fxy = eval_basis(self._basis, data, len(self._prior), len(self._moment))
        p = self._prior * safe_exp(np.dot(fxy, weight))[0]
        return p / force_tiny(np.sum(np.sum(p, 1)[:, None]))

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

    def __init__(self, data, target, basis, moments, prior=None):
        """
        data (n, n_features)
        target (n, )
        moments is an int
        Use empirical moments
        """
        self._init_training(data, target, prior)
        self._init_basis(basis)
        self._init_moment(moments)
        self._init_optimizer()

    def _init_training(self, data, target, prior):
        # Set prior and weight data accordinhgly
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

    def _init_moment(self, moments):
        # Assume moments is an int, compute the empirical moments
        self._moment = np.array([np.mean([self._basis(x, y, i)\
                                          for x, y in zip(self._target, self._data)])\
                                 for i in range(moments)])
        

#########################################################################
# Bayesian composite inference
#########################################################################

_GAUSS = .5 * np.log(2 * np.pi)
log_lik1d = lambda z, m, s: -(_GAUSS + np.log(s) + .5 * ((z - m) / s) ** 2)
mean_log_lik1d = lambda s: -(_GAUSS + np.log(s) + .5)

class GaussianCompositeInference(MaxentClassifier):

    def __init__(self, data, target, prior=None, supercomposite=False, homoscedastic=False):
        """
        data (n, n_features)
        target (n, )
        """
        self._homoscedastic = bool(homoscedastic)
        self._supercomposite = bool(supercomposite)
        self._init_training(data, target, prior)
        self._pre_train()
        self._init_basis()
        self._init_moment()
        self._init_optimizer()

    def _pre_train(self):
        # Pre-training: feature-based ML parameter estimates
        targets = len(self._prior)
        means = np.array([np.mean(self._data[self._target == x], 0) for x in range(targets)])
        res2 = (self._data - means[self._target]) ** 2
        if self._homoscedastic:
            devs = np.repeat(np.sqrt(np.mean(res2, 0))[None, :], targets, axis=0)
        else:
            devs = np.array([np.sqrt(np.mean(res2[self._target == x], 0)) for x in range(targets)])
        self._means = means
        self._devs = devs
        
    def _init_basis(self):
        features = self._data.shape[-1]
        if self._supercomposite:
            def basis(x, y, j):
                a = j // features
                i = j % features
                return (x == a) * log_lik1d(y[i], self._means[x, i], self._devs[x, i])
            self._basis = basis
        else:
            self._basis = lambda x, y, i: log_lik1d(y[i], self._means[x, i], self._devs[x, i])

    def _init_moment(self):
        # Optional computation, faster than empirical mean log-likelihood values
        moment = self._prior[:, None] * np.array([mean_log_lik1d(self._devs[x]) for x in range(len(self._prior))])
        if self._supercomposite:
            self._moment = moment.ravel()
        else:
            self._moment = moment.sum(0)


#########################################################################
# Logistic regression
#########################################################################

class LogisticRegression(MaxentClassifier):

    def __init__(self, data, target, prior=None):
        """
        data (n, n_features)
        target (n, )
        Use empirical moments
        """
        self._init_training(data, target, prior)
        self._init_basis()
        self._init_moment((self._data.shape[-1] + 1) * len(self._prior))
        self._init_optimizer()

    def _init_basis(self):
        n = self._data.shape[-1] + 1
        def basis(x, y, j):
            a = j // n
            i = j % n
            if i == 0:
                return (x == a)
            else:
                return (x == a) * y[i - 1]
        self._basis = basis


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
