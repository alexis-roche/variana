import sys

from variana.maxent import GaussianCompositeInference, MaxentCache, reshape_data, normalize_dist
from variana.maxent import LogisticRegression
from variana.utils import sdot, minimizer

from sklearn import datasets
import numpy as np
import pylab as pl


TEST_SIZE = 0.2
TOL = 1e-5
POSITIVE_WEIGHTS = True
HOMO_SCED = 1
REF_CLASS = None


def nice_lady(basis, weight):
    aux = np.dot(basis, weight)
    norma = aux.max(0)
    return np.exp(aux - norma), norma


def safe_exp_dist2(basis, weight, data_weight):
    aux, norma = nice_lady(basis, weight)
    return data_weight[:, None] * aux, norma



class MininfLikelihood(object):

    def __init__(self, obj):
        self._basis = obj._basis
        self._basis_generator = obj._basis_generator
        self._data = obj._data
        self._moment = obj._moment
        self._prior = obj._prior
        self._tiny = obj._tiny
        self._data_weight = obj.data_weight.copy()
        self._weight = obj._weight.copy()
        ###self._weight = np.zeros(len(self._moment))
        self._cache = MaxentCache()

    def _update_z(self, weight):
        if self._cache.same(weight):
            return
        udist, norma = safe_exp_dist2(self._basis, weight, self._data_weight)
        z = np.sum(udist, 0)
        self._cache.update_z(weight, norma, udist, z)

    def _update_z_and_grad(self, weight):
        self._update_z(weight)
        if not self._cache._grad_z is None:
            return
        udist_basis = self._cache._udist[..., None] * self._basis
        grad_z = np.sum(udist_basis, 0)
        self._cache.update_grad_z(weight, udist_basis, grad_z)

    def dual(self, weight):
        self._update_z(weight)
        z = self._cache._z
        norma = self._cache._norma
        return np.dot(weight, self._moment) - np.sum(self._prior * (np.log(np.maximum(z, self._tiny)) + norma))

    def gradient_dual(self, weight):
        self._update_z_and_grad(weight)
        z = self._cache._z
        grad_z = self._cache._grad_z
        return self._moment - np.sum(self._prior[:, None] * (grad_z / z[:, None]), 0)

    def hessian_dual(self, weight):
        self._update_z_and_grad(weight)      
        z = self._cache._z
        grad_z = self._cache._grad_z
        udist_basis = self._cache._udist_basis
        g1 = grad_z / z[:, None]
        H1 = sdot(g1[:, :, None], g1[:, None, :])
        H2 = sdot(np.swapaxes(np.swapaxes(udist_basis, 0, 1), 1, 2), np.swapaxes(self._basis, 0, 1)) / z[:, None, None]
        return np.sum(self._prior[:, None, None] * (H1 - H2), 0)
                        
    def _a_step(self, method='lbfgs', positive_weights=False, tol=1e-5, maxiter=10000):
        self._cache.reinit()
        f = lambda weight: -self.dual(weight)
        grad_f = lambda weight: -self.gradient_dual(weight)
        hess_f = lambda weight: -self.hessian_dual(weight)
        bounds = None
        if positive_weights:
            bounds = [(0, None) for i in range(len(self._weight))]
        m = minimizer(method, self._weight, f, grad_f, hess_f, bounds=bounds, tol=tol, maxiter=maxiter)
        self._weight = m.argmin()
        return m.info()

    def _b_step(self):
        self._data_weight = np.sum(self._prior * self.gen_dist(), 1)

    def fit(self,method='lbfgs', positive_weights=False, tol=1e-5, maxiter=10000):
        for i in range(maxiter):
            prev_weight = self._weight.copy()
            self._a_step(method, positive_weights, tol, maxiter)
            self._b_step()
            delta = np.max(np.abs(self._weight - prev_weight))
            if delta < tol:
                print('Convergence achieved after %d iterations' % (i + 1))
                break
       
    def gen_dist(self, weight=None):
        if weight is None:
            weight = self._weight
        self._update_z(weight)
        udist = self._cache._udist
        z = self._cache._z
        return udist / z

    def z(self, weight=None):
        if weight is None:
            weight = self._weight
        self._update_z(weight)
        return np.exp(self._cache._norma) * self._cache._z

    def test_gen_dist(self, weight=None):
        if weight is None:
            weight = self._weight
        self._update_z(weight)
        udist = self._cache._udist
        print(np.sum(udist, 0))
        print(self.z(weight))
    
    def dist(self, data=None, weight=None):
        if weight is None:
            weight = self._weight
        if data is None:
            data = self._data
        self._update_z(weight)
        basis = self._basis_generator(reshape_data(data))
        udist, norma = nice_lady(basis, weight)
        p = self._prior * (udist / self._cache._z) * np.exp(norma - self._cache._norma)
        return normalize_dist(p, self._tiny)
        

        

        
def one_hot_encoding(target):
    out = np.zeros((len(target), target.max() + 1))
    out[range(len(target)), target] = 1
    return out


def load(dataset, test_size=0.25, random_state=None):
    loaders = {'iris': datasets.load_iris,
               'digits': datasets.load_digits,
               'wine': datasets.load_wine,
               'breast_cancer': datasets.load_breast_cancer}
    data = loaders[dataset]()
    data, target = data.data, data.target
    n = data.shape[0]
    n_train = int((1 - test_size) * data.shape[0])
    p = np.random.permutation(n)
    train = p[0:n_train]
    test = p[n_train:]
    return data[train], target[train], data[test], target[test]


def accuracy(target, dist):
    return np.sum(target == np.argmax(dist, 1)) / len(target)


def cross_entropy(target, dist, tiny=1e-50):
    return -np.sum(one_hot_encoding(target) * np.log(np.maximum(tiny, dist))) / len(target)


def comparos(d1, d2):
    aux = np.max(np.abs(d1 - d2), 1)
    return np.mean(aux), np.argmax(aux)

dataset = 'iris'
method = 'lbfgs'
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    if len(sys.argv) > 2:
        method = sys.argv[2]

data, target, t_data, t_target = load(dataset, test_size=TEST_SIZE)


m = GaussianCompositeInference(data, target, homo_sced=HOMO_SCED, ref_class=REF_CLASS)
info = m.fit(method=method, tol=TOL, positive_weights=POSITIVE_WEIGHTS)
print('Learning time: %f sec' % info['time'])
print('Weight sum = %f' % np.sum(m.weight))
print('Weight ratio = %f' % (np.sum(m.weight) / data.shape[1]))
print('Weight sparsity = %f' % (np.sum(m.weight==0) / data.shape[1]))

mil = MininfLikelihood(m)
mil.fit(method=method, tol=TOL, positive_weights=POSITIVE_WEIGHTS)

d = m.dist()
dmil = mil.dist()

i = comparos(d, dmil)[1]

def zob(idx=None):
    if idx is None:
        idx = np.random.randint(data.shape[0])
    pl.figure()
    pl.plot(d[idx, :], 'b')
    pl.plot(dmil[idx, :], 'orange')
    pl.show()


### Evaluation
t_d = m.dist(t_data)
t_dmil = mil.dist(t_data)

acc_m = accuracy(t_target, t_d)
cen_m = cross_entropy(t_target, t_d)
acc_mil = accuracy(t_target, t_dmil)
cen_mil = cross_entropy(t_target, t_dmil)

print('MAXENT: acc=%f, cen=%f' % (acc_m, cen_m))
print('MIL: acc=%f, cen=%f' % (acc_mil, cen_mil))

###zob(i)

pl.figure()
###pl.plot(m._data_weight, 'b')
###pl.plot(mil._data_weight, 'orange')
###pl.plot(m._weight, 'b')
pl.plot(mil._weight, 'orange')
pl.show()

