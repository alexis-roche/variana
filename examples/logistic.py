import sys

from variana.maxent import GaussianCompositeInference
from variana.maxent import LogisticRegression as LogisticRegression2
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn import datasets
import numpy as np
import pylab as pl


TEST_SIZE = 0.2
TOL = 1e-20
POSITIVE_WEIGHTS = True
HOMO_SCED = 0
REF_CLASS = None


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


class Evaluator(object):

    def __init__(self, name, lr, data, target, t_data, t_target):
        self._name = name
        if isinstance(lr, LogisticRegression):
            self._dist = lr.predict_proba(data)
            self._t_dist = lr.predict_proba(t_data)
        else:
            self._dist = lr.dist()
            self._t_dist = lr.dist(t_data)
        self._cross_entropy = cross_entropy(target, self._dist)
        self._t_cross_entropy = cross_entropy(t_target, self._t_dist)
        self._accuracy = accuracy(target, self._dist)
        self._t_accuracy = accuracy(t_target, self._t_dist)
        self._lr = lr
        
    @property
    def train_cross_entropy(self):
        return self._cross_entropy

    @property
    def test_cross_entropy(self):
        return self._t_cross_entropy

    @property
    def train_accuracy(self):
        return self._accuracy

    @property
    def test_accuracy(self):
        return self._t_accuracy

    def disp(self, compare=(), grad_test=False):
        print('************ %s' % self._name)
        print('Train cross-entropy = %f, accuracy = %f' % (self.train_cross_entropy, self.train_accuracy))
        print('Test cross-entropy = %f, accuracy = %f' % (self.test_cross_entropy, self.test_accuracy))
        for other in compare:
            self.disp_compare(other)
        if grad_test:
            self.disp_grad_test()
        
    def disp_compare(self, other):
        print('Comparison (%s/%s): %f, %d' % (self._name, other._name, *comparos(self._dist, other._dist)))
    
    def disp_grad_test(self):
        if isinstance(self._lr, LogisticRegression):
            print('No grad test available for sklearn implementation')
            return
        g = self._lr.gradient_dual(self._lr.weight)
        print('Grad test = %f' % np.max(np.abs(g)))
        
    
dataset = 'iris'
method = 'lbfgs'
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    if len(sys.argv) > 2:
        method = sys.argv[2]

data, target, t_data, t_target = load(dataset, test_size=TEST_SIZE)


###mod = LogisticRegression(C=np.inf, class_weight='balanced', solver='lbfgs', multi_class='ovr', tol=TOL, max_iter=10000)
mod = LogisticRegression(C=np.inf, class_weight='balanced', solver='lbfgs', multi_class='multinomial', tol=TOL, max_iter=10000)
mod.fit(data, target)
emod = Evaluator('sklearn', mod, data, target, t_data, t_target)
emod.disp()

mod2 = LogisticRegression2(data, target)
info2 = mod2.fit(method='lbfgs', tol=TOL)
emod2 = Evaluator('variana', mod2, data, target, t_data, t_target)
emod2.disp(compare=(emod,), grad_test=True)
print('Learning time: %f sec' % info2['time'])

mod3 = GaussianCompositeInference(data, target, homo_sced=HOMO_SCED, ref_class=REF_CLASS)

mod3.set_weight(1)
jc = Evaluator('naive', mod3, data, target, t_data, t_target)
jc.disp()
mod3.set_weight(1 / len(mod3.weight))
jc2 = Evaluator('e-bayes', mod3, data, target, t_data, t_target)
jc2.disp()
mod3.set_weight(0)

info3 = mod3.fit(method=method, tol=TOL, positive_weights=POSITIVE_WEIGHTS)
emod3 = Evaluator('composite', mod3, data, target, t_data, t_target)
emod3.disp(compare=(emod, emod2), grad_test=True)
print('Learning time: %f sec' % info3['time'])
print('Weight sum = %f' % np.sum(mod3.weight))
print('Weight ratio = %f' % (np.sum(mod3.weight) / data.shape[1]))
print('Weight sparsity = %f' % (np.sum(mod3.weight==0) / data.shape[1]))


mod4 = GaussianCompositeInference(data, target, homo_sced=HOMO_SCED, ref_class=REF_CLASS)
info4 = mod4.fit(method=method, tol=TOL, positive_weights=False)
emod4 = Evaluator('composite', mod4, data, target, t_data, t_target)
emod4.disp(compare=(emod, emod2), grad_test=True)
print('Learning time: %f sec' % info4['time'])
print('Weight sum = %f' % np.sum(mod4.weight))
print('Weight ratio = %f' % (np.sum(mod4.weight) / data.shape[1]))
print('Weight sparsity = %f' % (np.sum(mod4.weight==0) / data.shape[1]))

print()
classes = 1 + target.max()
print('Number of classes: %d' % classes)
print('Number of features: %d' % data.shape[1])
print('Number of examples: %d' % data.shape[0])
print('Logistic regression parameters: %d' % len(mod2.weight))
print('Composite inference parameters: %d' % len(mod3.weight))
print('Chance cross-entropy: %f' % np.log(classes))

def zob(idx):
    pl.figure()
    pl.plot(emod._dist[idx, :], 'b:')
    try:
        pl.plot(emod2._dist[idx, :], 'b')
    except:
        print('emod2 does not exist')
    try:
        pl.plot(emod3._dist[idx, :], 'orange')
    except:
        print('emod3 does not exist')
    pl.show()


pl.plot(mod3.weight)
pl.plot(mod4.weight, 'r')
pl.show()

