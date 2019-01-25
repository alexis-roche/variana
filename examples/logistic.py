import sys

from variana.maxent import GaussianCompositeInference, LogisticRegression, MininfLikelihood
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn import datasets
import numpy as np
import pylab as pl


TEST_SIZE = 0.2
TOL = 1e-5
HOMO_SCED = 1
REF_CLASS = None
DAMPING = 0


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
        if isinstance(lr, SklearnLR):
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
        if isinstance(self._lr, SklearnLR):
            print('No grad test available for sklearn implementation')
            return
        g = self._lr.gradient_dual(self._lr.param)
        print('Grad test = %f' % np.max(np.abs(g)))
        

dataset = 'iris'
optimizer = 'lbfgs'
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    if len(sys.argv) > 2:
        optimizer = sys.argv[2]

data, target, t_data, t_target = load(dataset, test_size=TEST_SIZE)

C = 1 / max(data.shape[0] * DAMPING, 1e-100)
###C = 1 / DAMPING
m = SklearnLR(C=C, class_weight='balanced', multi_class='multinomial', solver='lbfgs', tol=TOL, max_iter=10000)
m.fit(data, target)
em = Evaluator('sklearn', m, data, target, t_data, t_target)
em.disp()

m1 = LogisticRegression(data, target, damping=DAMPING)
info1 = m1.fit(optimizer=optimizer, tol=TOL)
em1 = Evaluator('variana', m1, data, target, t_data, t_target)
em1.disp(compare=(em,), grad_test=True)
print('Learning time: %f sec' % info1['time'])

m2 = GaussianCompositeInference(data, target, damping=DAMPING, homo_sced=HOMO_SCED, ref_class=REF_CLASS)
m2.fit(objective='naive')
Evaluator('naive', m2, data, target, t_data, t_target).disp()
m2.fit(objective='agnostic')
Evaluator('agnostic', m2, data, target, t_data, t_target).disp()

info2 = m2.fit(optimizer=optimizer, tol=TOL)
em2 = Evaluator('composite', m2, data, target, t_data, t_target)
em2.disp(compare=(em, em1), grad_test=True)
print('Learning time: %f sec' % info2['time'])
print('Weight sum = %f' % np.sum(m2.weight))
print('Weight ratio = %f' % (np.sum(m2.weight) / m2.weight.size))
print('Weight sparsity = %f' % (np.sum(m2.weight==0) / m2.weight.size))

m3 = GaussianCompositeInference(data, target, positive_weight=False, damping=DAMPING, homo_sced=HOMO_SCED, ref_class=REF_CLASS)
info3 = m3.fit(optimizer=optimizer, tol=TOL)
em3 = Evaluator('composite with equality constraints', m3, data, target, t_data, t_target)
em3.disp(compare=(em, em1, em2), grad_test=True)
print('Learning time: %f sec' % info3['time'])

m4 = MininfLikelihood(m2)
info4 = m4.fit(optimizer=optimizer, tol=TOL)
em4 = Evaluator('composite MIL', m4, data, target, t_data, t_target)
em4.disp(compare=(em, em1, em2), grad_test=True)

print()
classes = 1 + target.max()
print('Number of classes: %d' % classes)
print('Number of features: %d' % data.shape[1])
print('Number of examples: %d' % data.shape[0])
print('Logistic regression parameters: %d' % len(m1.param))
print('Composite inference parameters: %d' % len(m2.param))
print('Chance cross-entropy: %f' % np.log(classes))

def zob(idx):
    pl.figure()
    pl.plot(em._dist[idx, :], 'g:')
    pl.plot(em2._dist[idx, :], 'g')
    pl.plot(em2._dist[idx, :], 'b')
    pl.plot(em3._dist[idx, :], 'r')
    pl.plot(em4._dist[idx, :], 'orange')
    pl.legend(('LR sklearn', 'LR', 'CBI', 'CBI-', 'CBI-MIL'))
    pl.show()

pl.figure()
pl.plot(m2.weight.ravel())
pl.plot(m3.weight.ravel(), 'r')
pl.plot(m4.weight.ravel(), 'orange')
pl.show()

