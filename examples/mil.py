import sys

from variana import GaussianCompositeInference, MininfLikelihood, LogisticRegression

from sklearn import datasets
import numpy as np
import pylab as pl


TEST_SIZE = 0.2
POSITIVE_WEIGHT = True
HOMO_SCED = 1
REF_CLASS = None
TRIALS = 10

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


dataset = 'iris'
if len(sys.argv) > 1:
    dataset = sys.argv[1]

weights, weights_mil = [], []

for i in range(TRIALS):
    print('Trial %d/%d' % (i + 1, TRIALS))
    
    data, target, t_data, t_target = load(dataset, test_size=TEST_SIZE)

    m = GaussianCompositeInference(data, target, positive_weight=POSITIVE_WEIGHT, homo_sced=HOMO_SCED, ref_class=REF_CLASS)
    m.fit()
    weights.append(m.weight)

    mil = MininfLikelihood(m)
    mil.fit()
    weights_mil.append(mil.weight)

weights = np.array(weights)
weights_mil = np.array(weights_mil)

s = np.std(weights, 0)
s_mil = np.std(weights_mil, 0)


