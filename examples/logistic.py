import sys

from variana.maxent import GaussianCompositeInference
from variana.maxent import LogisticRegression as LogisticRegression2
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn import datasets
import numpy as np
import pylab as pl


TEST_SIZE = 0.2
POSITIVE_WEIGHTS = True
HOMOSCEDASTIC = False
SUPERCOMPOSITE = False


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
    

def cross_entropy(target, dist, tiny=1e-50):
    return -np.sum(one_hot_encoding(target) * np.log(np.maximum(tiny, dist))) / len(target)


def comparos(d1, d2):
    aux = np.max(np.abs(d1 - d2), 1)
    return np.mean(aux), np.argmax(aux)


class Evaluator(object):

    def __init__(self, name, lr, data, target, t_data, t_target, **kwargs):
        self._name = name
        if isinstance(lr, LogisticRegression):
            lr.fit(data, target, **kwargs)
            self._dist = lr.predict_proba(data)
            self._t_dist = lr.predict_proba(t_data)
        else:
            lr.fit(**kwargs)
            self._dist = lr.dist()
            self._t_dist = lr.dist(t_data)
        self._cross_entropy = cross_entropy(target, self._dist)
        self._t_cross_entropy = cross_entropy(t_target, self._t_dist)
        self._lr = lr
        
    @property
    def train_cross_entropy(self):
        return self._cross_entropy

    @property
    def test_cross_entropy(self):
        return self._t_cross_entropy

    def disp(self):
        print('Train cross-entropy %s = %f' % (self._name, self.train_cross_entropy))
        print('Test cross-entropy %s = %f' % (self._name, self.test_cross_entropy))

    def compare(self, other):
        print('Comparison (%s/%s): %f, %d' % (self._name, other._name, *comparos(self._dist, other._dist)))
    
    def grad_test(self):
        if isinstance(self._lr, LogisticRegression):
            print('No grad test available for sklearn implementation')
            return
        g = self._lr.gradient_dual(self._lr.weight)
        print('Grad test = %f' % np.max(np.abs(g)))
        
    
dataset = 'iris'
method = 'newton'
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    if len(sys.argv) > 2:
        method = sys.argv[2]

data, target, t_data, t_target = load(dataset, test_size=TEST_SIZE)


print('*************************************')
lr = LogisticRegression(C=np.inf, class_weight='balanced', solver='lbfgs', multi_class='multinomial')
elr = Evaluator('sklearn', lr, data, target, t_data, t_target)
elr.disp()

print('*************************************')
lr2 = LogisticRegression2(data, target)
elr2 = Evaluator('variana', lr2, data, target, t_data, t_target, method=method)
elr2.grad_test()
elr2.compare(elr)
elr2.disp()


print('*************************************')
lr3 = GaussianCompositeInference(data, target,
                                 homoscedastic=HOMOSCEDASTIC,
                                 supercomposite=SUPERCOMPOSITE)
elr3 = Evaluator('composite', lr3, data, target, t_data, t_target, method=method,
                 positive_weights=POSITIVE_WEIGHTS)
elr3.grad_test()
elr3.compare(elr)
elr3.compare(elr2)
elr3.disp()


def zob(idx):
    pl.figure()
    pl.plot(elr._dist[idx, :], 'b:')
    try:
        pl.plot(elr2._dist[idx, :], 'b')
    except:
        print('elr2 does not exist')
    try:
        pl.plot(elr3._dist[idx, :], 'orange')
    except:
        print('elr3 does not exist')
    pl.show()

