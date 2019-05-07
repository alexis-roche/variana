import numpy as np
import scipy.optimize as spo

from .utils import sdot, minimizer, probe_time

TINY = 1e-100


class MaxentCache(object):

    def __init__(self):
        self.reinit()

    def reinit(self, inplace_update=False):        
        self._param = None
        self._udist = None
        self._norma = None
        self._z = None
        self._udist_basis = None
        self._grad_log_z = None
        self._inplace_update = inplace_update

    def same(self, param):
        return np.array_equal(param, self._param)

    def _store_param(self, param):
        # If param is modified in place, we need to copy it
        if self._inplace_update:
            self._param = param.copy()
        else:
            self._param = param
   
    def update1(self, param, norma, udist, z):
        self._store_param(param)
        self._udist = udist
        self._norma = norma
        self._z = z
        self._udist_basis = None
        self._grad_log_z = None

    def update2(self, param, udist_basis, grad_log_z):
        if not self.same(param):
            raise RuntimeError('Cannot run update2 before update1')
        self._udist_basis = udist_basis
        self._grad_log_z = grad_log_z

        

class Maxent(object):

    def __init__(self, basis, moment, damping=0, bounds=None, prior=None):
        """
        basis is an array (targets, moments)
        moment is a sequence of mean values corresponding to basis
        prior is None or an array-like reperesenting the prior
        """
        self._init_basis(basis)
        self._init_optimizer(damping, bounds)
        self._init_prior(prior)
        self._init_moment(moment)

    def _init_basis(self, basis):
        self._basis = np.asarray(basis)
        if self._basis.ndim == 1:
            self._basis = self._basis[:, None]
        self._targets, self._params = self._basis.shape
        self._examples = 0
        
    def _init_optimizer(self, damping, bounds):
        self._param = np.zeros(self._params)
        self._damping = np.full(self._params, damping / max(1, self._examples), dtype=float)
        self._bounds = bounds
        self._cache = MaxentCache()
        
    def _init_prior(self, prior):
        if prior is None:
            self._prior = np.ones(self._targets)
        else:
            self._prior = np.asarray(prior)
        self._prior /= np.sum(self._prior)        

    def _init_moment(self, moment):
        self._moment = np.asarray(moment)
        if self._moment.ndim == 0:
            self._moment = self._moment[None]

    def _update1(self, param):
        if self._cache.same(param):
            return
        aux = np.dot(self._basis, param)
        norma = aux.max()
        udist = self._prior * np.exp(aux - norma)
        z = np.sum(udist)
        self._cache.update1(param, norma, udist, z)

    def _update2(self, param):
        self._update1(param)
        if not self._cache._grad_log_z is None:
            return
        udist_basis = self._cache._udist[:, None] * self._basis
        grad_log_z = np.sum(udist_basis, 0) / self._cache._z
        self._cache.update2(param, udist_basis, grad_log_z)

    def dual(self, param):
        self._update1(param)
        relevance = np.dot(param, self._moment)
        log_partition = np.log(np.maximum(self._cache._z, TINY)) + self._cache._norma
        return relevance - log_partition

    def gradient_dual(self, param):
        self._update2(param)
        return self._moment - self._cache._grad_log_z
        
    def hessian_dual(self, param):
        self._update2(param)
        H1 = np.dot(self._cache._grad_log_z[:, None], self._cache._grad_log_z[None, :])
        H2 = np.dot(self._cache._udist_basis.T, self._basis) / self._cache._z
        return H1 - H2
        
    def _opt_param(self, optimizer, tol, maxiter):
        self._cache.reinit(inplace_update=optimizer in ('lbfgs',))
        if self._damping.max() == 0:
            f = lambda param: -self.dual(param)
            grad_f = lambda param: -self.gradient_dual(param)
            hess_f = lambda param: -self.hessian_dual(param)
        else:
            f = lambda param: -self.dual(param) + .5 * np.sum(self._damping * param ** 2)
            grad_f = lambda param: -self.gradient_dual(param) + self._damping * param
            hess_f = lambda param: -self.hessian_dual(param) + np.diag(self._damping)
        m = minimizer(optimizer, self._param, f, grad_f, hess_f, bounds=self._bounds, tol=tol, maxiter=maxiter)
        self._param = m.argmin()
        return m.info()

    def fit(self, optimizer='lbfgs', tol=1e-5, maxiter=10000):
        return self._opt_param(optimizer, tol, maxiter)

    def dist(self):
        self._update1(self._param)
        udist = self._cache._udist
        return udist / np.sum(udist)

    @property
    def score(self):
        return self.dual(self._param)
    
    @property
    def param(self):
        return self._param

    @property
    def prior(self):
        return self._prior

    @property
    def moment(self):
        return self._moment
    
    @property
    def achieved_moment(self):
        return np.sum(self.dist()[:, None] * self._basis, 0)


def reshape_data(data):
    """
    Try to convert the input into a 2d array with shape (examples, features)
    """
    out = np.asarray(data)
    if out.ndim == 0:
        out = np.reshape(out, (1, 1))
    elif out.ndim == 1:
        out = out[:, None]
    elif out.ndim > 2:
        raise ValueError('Cannot process input data')
    return out



def safe_exp_dot(basis, param, axis=None):
    """
    Compute exp(dot(basis, param)) "in two pieces"
    Returns two arrays a and b such that:
    exp(dot(basis, param)) = exp(b) a

    basis should be of shape (examples, targets, params)
    param should be of shape (params,)
    """
    aux = np.dot(basis, param)
    norma = aux.max(axis)
    if axis is None:
        return np.exp(aux - norma), norma
    elif axis == 1:  
        return np.exp(aux - norma[:, None]), norma
    elif axis == 0:
        return np.exp(aux - norma[None, :]), norma


def normalize_dist(p):
    squeeze = False
    if p.ndim < 2:
        squeeze = True
        p = p[None, :]
    out = np.full(p.shape, 1 / p.shape[1])
    aux = np.sum(p, 1)
    nonzero = aux > TINY
    out[nonzero] = p[nonzero] / aux[nonzero][:, None]
    if squeeze:
        return out.squeeze()
    return out



class ConditionalMaxent(Maxent):

    def __init__(self, basis_generator, moment, data, prior=None, bounds=None, data_weight=None, damping=0):
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
        self._init_optimizer(damping, bounds)
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
            self._sample_mean = lambda x: np.sum(self._data_weight.reshape([x.shape[0]] + [1] * (x.ndim - 1)) * x, 0)

    def _init_basis(self, basis_generator):
        self._basis_generator = basis_generator
        self._basis = basis_generator(self._data)
        self._examples, self._targets, self._params = self._basis.shape
        self._features = self._data.shape[1]
        
    def _update1(self, param):
        if self._cache.same(param):
            return
        aux, norma = safe_exp_dot(self._basis, param, axis=1)
        udist = self._prior * aux
        z = np.sum(udist, 1)
        self._cache.update1(param, norma, udist, z)

    def _update2(self, param):
        self._update1(param)
        if not self._cache._grad_log_z is None:
            return
        udist_basis = self._cache._udist[..., None] * self._basis
        grad_log_z = np.sum(udist_basis, 1) / self._cache._z[:, None]
        self._cache.update2(param, udist_basis, grad_log_z)

    def dual(self, param):
        self._update1(param)
        relevance = np.dot(param, self._moment)
        log_partition = self._sample_mean(np.log(np.maximum(self._cache._z, TINY)) + self._cache._norma)
        return relevance - log_partition
    
    def gradient_dual(self, param):
        self._update2(param)
        return self._moment - self._sample_mean(self._cache._grad_log_z)
        
    def hessian_dual(self, param):
        self._update2(param)      
        g1 = self._cache._grad_log_z
        H1 = sdot(g1[:, :, None], g1[:, None, :])
        H2 = sdot(np.swapaxes(self._cache._udist_basis, 1, 2), self._basis) / self._cache._z[:, None, None]
        return self._sample_mean(H1 - H2)
    
    def dist(self, data=None, param=None):
        if param is None:
            param = self._param
        if data is None:
            self._update1(param)
            return normalize_dist(self._cache._udist)
        aux, _ = safe_exp_dot(self._basis_generator(reshape_data(data)), param, axis=1)
        return normalize_dist(self._prior * aux)

    @property
    def data(self):
        return self._data

    @property
    def data_weight(self):
        if self._data_weight is None:
            return np.full(self._examples, 1 / self._examples)
        return self._data_weight
    
    @property
    def achieved_moment(self):
        return self._sample_mean(np.sum(self.dist()[:, :, None] * self._basis, 1))

    
# ***********************************************************************
# Maxent classifier
# ***********************************************************************

class MaxentClassifier(ConditionalMaxent):

    def __init__(self, data, target, basis_generator, prior=None, damping=0, bounds=None):
        """
        data (examples, features)
        target (examples, )
        Use empirical moments
        """
        self._init_dataset(data, target, prior)
        self._init_basis(basis_generator)
        self._init_optimizer(damping, bounds)
        self._init_moment()

    def _init_dataset(self, data, target, prior):
        # Set prior and param data accordingly
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
        self._moment = self._sample_mean(self._basis[range(self._examples), self._target])

 

# ***********************************************************************
# Bayesian composite inference
# ***********************************************************************

GAUSS_CONSTANT = .5 * np.log(2 * np.pi)


def log_lik1d(z, m, s, max_log=1000):
    s = np.maximum(s, TINY)
    return np.clip(-(GAUSS_CONSTANT + np.log(s) + .5 * ((z - m) / s) ** 2), -max_log, max_log)


def mean_log_lik1d(s):
    s = np.maximum(s, TINY)
    return -(GAUSS_CONSTANT + np.log(s) + .5)


def step_vector(size, start, val):
    out = np.zeros(size)
    out[start:] = val   
    return out


def make_bounds(positive_weight, params, targets, offsets):
    if not positive_weight:
        return None
    if offsets > 0:
        bounds = [(None, None) for i in range(targets - 1)]
    else:
        bounds = []
    bounds += [(0, None) for i in range(params - offsets)]
    return bounds


class GaussianCompositeInference(MaxentClassifier):

    def __init__(self, data, target, prior=None, positive_weight=True, damping=0, homo_sced=0, ref_class=None, offset=False, max_log=1000):
        """
        data (examples, features)
        target (examples, )
        """
        self._homo_sced = max(0, min(1, float(homo_sced)))
        self._ref_class = None
        if not ref_class is None:
            self._ref_class = int(ref_class)
        self._use_offset = bool(offset)
        self._max_log = float(max_log)
        self._init_dataset(data, target, prior)        
        self._init_training()
        self._init_basis(self._make_basis_generator(self._max_log))
        self._offsets = self._use_offset * (self._targets - 1)
        self._init_optimizer(step_vector(self._params, self._offsets, damping),
                             make_bounds(positive_weight, self._params, self._targets, self._offsets))
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

    def _make_basis_generator(self, max_log):
        targets = len(self._prior)
        features = self._data.shape[1]
        offsets = self._use_offset * (targets - 1)
        basis_fun = lambda z, m, s: log_lik1d(z, m, s, max_log)
        def basis_generator(data):
            out = np.zeros((data.shape[0], targets, offsets + features))
            if self._use_offset:
                for x in range(1, targets):
                    out[:, x, x - 1] = 1
            for x in range(targets):
                out[:, x, offsets:] = basis_fun(data, self._means[x], self._devs[x])
            return out
        def basis_generator_super(data):
            out = np.zeros((data.shape[0], targets, offsets + features * (targets - 1)))
            ll_ref = basis_fun(data, self._means[self._ref_class], self._devs[self._ref_class])
            indexes = list(range(targets))
            indexes.pop(self._ref_class)
            if self._use_offset:
                for i, x in enumerate(indexes):
                    out[:, x, i] = 1
            for i, x in enumerate(indexes):
                start = offsets + features * i
                out[:, x, start:(start + features)] = basis_fun(data, self._means[x], self._devs[x]) - ll_ref
            return out
        if self._ref_class is None:
            return basis_generator
        else:
            return basis_generator_super
        
    def _check_moment(self):
        # Faster than empirical mean log-likelihood values but does
        # not work if super composite and homoscedastic
        moment = self._prior[:, None] * np.array([mean_log_lik1d(self._devs[x]) for x in range(self._targets)])
        if self._ref_class is None:
            moment = moment.sum(0)
        else:
            moment = moment.ravel()
        return moment

    def fit(self, objective='maxent', optimizer='lbfgs', tol=1e-5, maxiter=10000):
        if objective in ('naive', 'agnostic'):
            self._param[0:self._offsets] = 0
            if objective == 'naive':
                aux = 1
            else:
                aux = 1 / self._features
            self._param[self._offsets:] = aux
            return {}
        return self._opt_param(optimizer, tol, maxiter)
    
    def _offset(self, param=None):
        if param is None:
            param = self._param
        if not self._use_offset:
            return np.zeros(self._targets)
        if self._ref_class is None:
            return np.concatenate(((0,), param[0:(self._targets - 1)]))
        out = np.zeros(self._targets)
        indexes = list(range(self._targets))
        indexes.pop(self._ref_class)
        out[indexes] = param[np.arange(self._targets - 1)]
        return out

    def _weight(self, param=None):
        if param is None:
            param = self._param
        if self._ref_class is None:
            return param[self._offsets:]
        out = np.zeros((self._targets, self._features))
        indexes = list(range(self._targets))
        indexes.pop(self._ref_class)
        out[indexes, :] = param[self._offsets:].reshape((self._targets - 1, self._features))
        return out

    @property
    def offset(self):
        return self._offset()

    @property
    def weight(self):
        return self._weight()

    @property
    def reference(self):
        return normalize_dist(self._prior * np.exp(self._offset()))
    

    
# ***********************************************************************
# Logistic regression
# ***********************************************************************

class LogisticRegression(MaxentClassifier):

    def __init__(self, data, target, prior=None, damping=0, offset=True):
        self._use_offset = bool(offset)
        self._init_dataset(data, target, prior)
        self._init_basis(self._make_basis_generator())
        self._offsets = self._use_offset * (self._targets - 1)
        self._init_optimizer(step_vector(self._params, self._offsets, damping), None)
        self._init_moment()

    def _make_basis_generator(self):
        targets = len(self._prior)
        offsets = self._use_offset * (targets - 1)
        features = self._data.shape[1]
        def basis_generator(data):
            out = np.zeros((data.shape[0], targets, offsets + features * (targets - 1)))
            if self._use_offset:
                for x in range(1, targets):
                    out[:, x, x - 1] = 1
            for x in range(1, targets):
                start = offsets + features * (x - 1)
                out[:, x, start:(start + features)] = data
            return out
        return basis_generator

    def _offset(self, param=None):
        if param is None:
            param = self._param
        if not self._use_offset:
            return np.zeros(self._targets)
        return np.concatenate(((0,), param[0:(self._targets - 1)]))

    def _weight(self, param=None):
        if param is None:
            param = self._param
        out = np.zeros((self._targets, self._features))
        out[1:, :] = param[self._offsets:].reshape((self._targets - 1, self._features))
        return out

    @property
    def offset(self):
        return self._offset()

    @property
    def weight(self):
        return self._weight()

    @property
    def reference(self):
        return normalize_dist(self._prior * np.exp(self._offset()))


# ***********************************************************************
# Minimum information likelihood
# ***********************************************************************

class MininfLikelihood(object):

    def __init__(self, obj):
        self._obj = obj
        self._basis = obj._basis
        self._basis_generator = obj._basis_generator
        self._data = obj._data
        self._moment = obj._moment
        self._prior = obj._prior
        self._damping = obj._damping
        self._bounds = obj._bounds
        self._examples = obj._examples
        self._targets = obj._targets
        self._params = obj._params
        self._features = obj._features
        ###self._data_weight = obj.data_weight.copy()
        self._data_weight = np.ones(self._examples) / self._examples
        self._param = np.zeros(self._params)
        self._cache = MaxentCache()
        
    def _update1(self, param):
        if self._cache.same(param):
            return
        aux, norma = safe_exp_dot(self._basis, param)
        udist = self._data_weight[:, None] * self._prior * aux        
        z = np.sum(udist)
        self._cache.update1(param, norma, udist, z)

    def _update2(self, param):
        self._update1(param)
        if not self._cache._grad_log_z is None:
            return
        udist_basis = self._cache._udist[..., None] * self._basis
        grad_log_z = np.sum(udist_basis, (0, 1)) / self._cache._z
        self._cache.update2(param, udist_basis, grad_log_z)

    def dual(self, param):
        self._update1(param)
        return np.dot(param, self._moment) - np.log(np.maximum(self._cache._z, TINY)) - self._cache._norma

    def gradient_dual(self, param):
        self._update2(param)
        return self._moment - self._cache._grad_log_z

    def hessian_dual(self, param):
        self._update2(param)      
        t = self._cache._grad_log_z
        H1 = np.dot(t[:, None], t[None, :])
        aux = self._examples * self._targets
        H2 = np.sum(sdot(self._cache._udist_basis.reshape((aux, self._params, 1)), self._basis.reshape((aux, 1, self._params))), 0) / self._cache._z
        return H1 - H2
                        
    def _a_step(self, optimizer='lbfgs', tol=1e-5, maxiter=10000):
        self._cache.reinit()
        if self._damping.max() == 0:
            f = lambda param: -self.dual(param)
            grad_f = lambda param: -self.gradient_dual(param)
            hess_f = lambda param: -self.hessian_dual(param)
        else:
            f = lambda param: -self.dual(param) + .5 * np.sum(self._damping * param ** 2)
            grad_f = lambda param: -self.gradient_dual(param) + self._damping * param
            hess_f = lambda param: -self.hessian_dual(param) + np.diag(self._damping)
        m = minimizer(optimizer, self._param, f, grad_f, hess_f, bounds=self._bounds, tol=tol, maxiter=maxiter)
        self._param = m.argmin()
        return m.info()

    def _b_step(self):
        self._data_weight = np.sum(self.joint_dist(), 1)

    @probe_time
    def _fit(self, optimizer, tol, maxiter):
        for it in range(maxiter):
            prev_param = self._param.copy()
            info = self._a_step(optimizer, tol, maxiter)
            self._b_step()
            delta = np.max(np.abs(self._param - prev_param))
            if delta < tol:
                it = it + 1
                break
        return info, it + 1
            
    def fit(self, optimizer='lbfgs', tol=1e-5, maxiter=10000):
        time, info, it = self._fit(optimizer, tol, maxiter)
        info['time'] = time
        info['BA iterations'] = it
        return info
        
    def joint_dist(self, param=None):
        if param is None:
            param = self._param
        self._update1(param)
        udist = self._cache._udist
        z = self._cache._z
        return udist / z
            
    def dist(self, data=None, param=None):
        if param is None:
            param = self._param
        return self._obj.dist(data, param)
        
    @property
    def score(self):
        return self.dual(self._param)
    
    @property
    def param(self):
        return self._param

    @property
    def prior(self):
        return self._prior

    @property
    def moment(self):
        return self._moment
    
    @property
    def achieved_moment(self):
        return np.sum(self.joint_dist()[..., None] * self._basis, (0,1))
    
    @property
    def data(self):
        return self._data

    @property
    def data_weight(self):
        if self._data_weight is None:
            return np.full(self._examples, 1 / self._examples)
        return self._data_weight

    @property
    def offset(self):
        if hasattr(self._obj, 'offset'):
            return self._obj._offset(self._param)
        raise ValueError('underlying method has no offset attribute')

    @property
    def weight(self):
        if hasattr(self._obj, 'weight'):
            return self._obj._weight(self._param)
        raise ValueError('underlying method has no weight attribute')



