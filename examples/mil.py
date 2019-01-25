import sys

from variana.maxent import GaussianCompositeInference, MininfLikelihood
from variana.maxent import LogisticRegression

from sklearn import datasets
import numpy as np
import pylab as pl


TEST_SIZE = 0.2
TOL = 1e-5
POSITIVE_WEIGHT = True
HOMO_SCED = 1
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

dataset = 'iris'
optimizer = 'lbfgs'
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    if len(sys.argv) > 2:
        optimizer = sys.argv[2]

data, target, t_data, t_target = load(dataset, test_size=TEST_SIZE)


m = GaussianCompositeInference(data, target, positive_weight=POSITIVE_WEIGHT, homo_sced=HOMO_SCED, ref_class=REF_CLASS)
info = m.fit(optimizer=optimizer, tol=TOL)
print('Learning time: %f sec' % info['time'])
print('Weight sum = %f' % np.sum(m.weight))
print('Weight ratio = %f' % (np.sum(m.weight) / data.shape[1]))
print('Weight sparsity = %f' % (np.sum(m.weight==0) / data.shape[1]))

mil = MininfLikelihood(m)
infomil = mil.fit(optimizer=optimizer, tol=TOL)

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

pl.figure()
pl.title('Data weights')
pl.plot(m._data_weight, 'b')
pl.plot(mil._data_weight, 'orange')

pl.figure()
pl.title('Parameters')
pl.plot(m._param, 'b')
pl.plot(mil._param, 'orange')
pl.show()



