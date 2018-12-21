import sys

from variana.maxent import GaussianCompositeInference
from variana.maxent import LogisticRegression as LogisticRegression2
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import pylab as pl


TEST_SIZE = 0.2


def one_hot_encoding(target):
    out = np.zeros((len(target), target.max() + 1))
    out[range(len(target)), target] = 1
    return out


def load(dataset):
    loaders = {'iris': datasets.load_iris,
               'digits': datasets.load_digits,
               'wine': datasets.load_wine,
               'breast_cancer': datasets.load_breast_cancer}
    data = loaders[dataset]()
    return data.data, data.target


def cross_entropy(target, dist, tiny=1e-50):
    return -np.sum(one_hot_encoding(target) * np.log(np.maximum(tiny, dist))) / len(target)


def comparos(d1, d2):
    aux = np.max(np.abs(d1 - d2), 1)
    return np.mean(aux), np.argmax(aux)


dataset = 'iris'
if len(sys.argv) > 1:
    dataset = sys.argv[1]

data, target = load(dataset)

# DEBUG
###data = data[:, 0:2]

# sklearn implementation
print('sklearn')
lr = LogisticRegression(C=np.inf, class_weight='balanced', solver='lbfgs', multi_class='multinomial')
lr.fit(data, target)
dist = lr.predict_proba(data)
print('Cross-entropy sklearn = %f' % cross_entropy(target, dist))

# variana implementation
print('variana')
lr2 = LogisticRegression2(data, target)
print('Opt starts')
lr2.fit(verbose=True)
dist2 = lr2.dist()
print('Cross-entropy variana = %f' % cross_entropy(target, dist2))

g = lr2.gradient_dual(lr2.weight)
print('Grad test = %f' % np.max(np.abs(g)))
print(comparos(dist, dist2))


"""
# Composite likelihood
print('composite stuff')
lr3 = GaussianCompositeInference(data, target, homoscedastic=True)
lr3.fit()
print('Opt starts')
dist3 = lr3.dist()
print('Cross-entropy composite = %f' % cross_entropy(target, dist3))
print(comparos(dist, dist3))
"""

def zob(idx):
    pl.figure()
    pl.plot(dist[idx, :], 'b')
    try:
        pl.plot(dist2[idx, :], 'g')
    except:
        print('dist2 does not exist')
    try:
        pl.plot(dist3[idx, :], 'orange')
    except:
        print('dist3 does not exist')
    pl.show()


#y, yt, x, xt = train_test_split(data, target, test_size=TEST_SIZE)


