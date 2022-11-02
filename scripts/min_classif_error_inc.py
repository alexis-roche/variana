from sys import argv
from time import time
from os.path import join
import numpy as np

from variana import FactorGaussian
from variana.variana import IncrementalEP

from load_data import get_data
from logistic import LogisticRegression
from loss import (logistic_loss, logistic_d1, logistic_d2,
                  hinge_loss, hinge_d1, hinge_d2,
                  quasi01_loss, quasi01_d1, quasi01_d2)




DATASET = 'bank'
NITERS = 452
#DATASET = 'adult'
#NITERS = 325
LOSS = 'logistic'  # logistic, hinge, quasi_0-1
if len(argv) > 1:
    DATASET = argv[1] 
    if len(argv) > 2:
        NITERS = argv[2]
        if leng(argv) > 3:
 	    LOSS = argv[3]

if LOSS == 'quasi':
    LOSS = 'quasi 0-1'

print DATASET, LOSS, NITERS


EVAL_TOTAL_COST = True
SHUFFLE_DATA = True

METHODS = 'laplace', 'quick_laplace', 'quadrature', 'variational'
SHORT_METHODS = 'LA', 'QLA', 'GQ', 'VQ'
COLORS = 'blue', 'green', 'red', 'orange'

BETA = 1
PRIOR_VAR = 25
BATCH_SIZE = 100
MINIMIZER = 'newton'


def rename(dataset):
    if dataset == 'bank':
        return 'bank marketing'
    elif dataset == 'adult':
        return 'census income'
    else:
        return dataset

def simul_data(npts, dim, noise=0):
    X = np.random.normal(size=(npts, dim))
    X[:, -1] = 1
    theta = np.ones(dim)
    y = np.sign(np.dot(X, theta) + noise * np.random.normal(size=(npts,)))
    return y, X

    
def slices(npts, size):
    I = range(npts)
    j, out = 0, []
    for i in range(npts / size + 1):
        if j < npts:
            out.append(I[j:j+size])
        j += size
    return out


def load_data():
    y, x, _ = get_data(DATASET)
    # subtract mean to each feature
    xn = x - np.mean(x, 0)
    # append a constant baseline
    nfeat = x.shape[1]
    X = np.ones((x.shape[0], nfeat + 1))
    X[:, 0:nfeat] = xn
    # normalize to unit norm
    X /= np.sqrt(np.sum(X ** 2, 0))
    if SHUFFLE_DATA:
        y, X = shuffle(y, X)
    return y, X


def shuffle(y, X):
    I = np.random.permutation(X.shape[0])
    X = X[I, :]
    y = y[I]
    return y, X


class ClassifIncrementalEP(object):

    def __init__(self, y, X, loss='hinge', prior_var=None):
        self.y = y
        self.X = X
        self.npts, self.dim = X.shape
        if prior_var is None:
            prior_var = 1e20
        self.prior_var = float(prior_var)
        self.loss_d1 = None
        self.loss_d2 = None
        if loss == 'zero_one':
            self.loss = zero_one_loss
        elif loss == 'logistic':
            self.loss = logistic_loss
            self.loss_d1 = logistic_d1
            self.loss_d2 = logistic_d2
        elif loss == 'hinge':
            self.loss = hinge_loss
            self.loss_d1 = hinge_d1
            self.loss_d2 = hinge_d2
        else:
            self.loss = quasi01_loss
            self.loss_d1 = quasi01_d1
            self.loss_d2 = quasi01_d2
        
    def cost(self, theta, k):
        # theta has shape (dim, n) or (dim,)
        if theta.ndim == 1:
            theta = np.reshape(theta, (len(theta), 1))
        fk = np.sum(self.X[k, :] * theta.T, 1)
        return np.squeeze(self.loss(self.y[k], fk))

    def gradient_cost(self, theta, k):
        # theta has shape (dim, n) or (dim,)
        if theta.ndim == 1:
            theta = np.reshape(theta, (len(theta), 1))
        fk = np.sum(self.X[k, :] * theta.T, 1)
        tmp = -self.y[k] * self.X[k]
        F1u = self.loss_d1(-self.y[k] * fk)
        return tmp.reshape((len(theta), 1)) * F1u

    def hessian_cost(self, theta, k):
        # theta has shape (dim, n) or (dim,)
        if theta.ndim == 1:
            theta = np.reshape(theta, (len(theta), 1))
        fk = np.sum(self.X[k, :] * theta.T, 1)
        tmp = self.X[k] ** 2
        F2u = self.loss_d1(-self.y[k] * fk)
        return tmp.reshape((len(theta), 1)) * F2u
    
    def total_cost(self, theta):
        return np.sum([self.cost(theta, k) for k in range(self.npts)])

    def total_regularized_cost(self, theta):
        return self.total_cost(theta) + (.5 / self.prior_var) * np.sum((theta ** 2), 0)

    def run(self, batch_size, beta, method, niters=5):

        beta = float(beta)
        prior = FactorGaussian(np.zeros(self.dim), self.prior_var * np.ones(self.dim))
        A = IncrementalEP(prior, method=method, minimizer=MINIMIZER)

        def target(theta):
            tmp = [self.cost(theta, k) for k in factor]
            if theta.ndim > 1:
                return -beta * np.sum(tmp, 0)
            else:
                return -beta * np.sum(tmp)

        def gradient_target(theta):
            tmp = [self.gradient_cost(theta, k) for k in factor]
            return -beta * np.sum(tmp, 0).squeeze()

        def hessian_target(theta):
            tmp = [self.hessian_cost(theta, k) for k in factor]
            return -beta * np.sum(tmp, 0).squeeze()

        k0 = 0
        k1 = batch_size

        C, dx = [], []
        t0 = time()
        for i in range(niters):

            factor = range(k0, k1)
            print('Iter %d/%d, Slice: %d, %d' % (i+1, niters, k0, k1))

            x0 = A.gaussian.m
            if EVAL_TOTAL_COST:
                C.append(self.total_regularized_cost(x0))
            else:
                C.append(0.)

            A.update_factor(target, gradient=gradient_target, hessian=hessian_target)
            x = A.gaussian.m
            dx.append(np.sqrt(np.sum((x-x0)**2)))

            # update slice
            k0 = k1
            k1 += batch_size
            
        dt = 1000 * (time() - t0) / float(niters)
        return A, C, dx, dt


# Load data
y, X = load_data()
EP = ClassifIncrementalEP(y, X, loss=LOSS, prior_var=PRIOR_VAR)

A, C, dx, dt = {}, {}, {}, {}
for met in METHODS:
    A[met], C[met], dx[met], dt[met] = EP.run(BATCH_SIZE, BETA, met, niters=NITERS)

# Timing
for met in METHODS:
    mt = np.mean(dt[met])
    m1t = np.median(dt[met])
    print('Time per mini-batch iteration: %s, mean=%f (ms) median=%f (ms)' % (met, mt, m1t))


