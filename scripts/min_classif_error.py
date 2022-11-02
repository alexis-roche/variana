from sys import argv
from time import time
from os.path import join
import numpy as np

from variana import Variana, Gaussian, FactorGaussian, NumEP

from load_data import get_data
from logistic import LogisticRegression
from loss import (logistic_loss, logistic_d1, logistic_d2,
                  hinge_loss, hinge_d1, hinge_d2,
                  quasi01_loss, quasi01_d1, quasi01_d2)

#DATASET = 'haberman'  # (306, 4)
#DATASET = 'ionosphere' # (351, 33)
#DATASET = 'wdbc' # (569, 31)
#DATASET = 'parkinsons'  # (195, 23)
#DATASET = 'SPECTF' # (80, 45)
#DATASET = 'wpbc' # (194, 34)


DATASET = 'haberman'
LOSS = 'logistic'
if len(argv) > 1:
    DATASET = argv[1] 
    if len(argv) > 2:
        LOSS = argv[2]

if LOSS == 'quasi':
    LOSS = 'quasi 0-1'

print DATASET, LOSS

METHODS = 'laplace', 'quick_laplace', 'quadrature', 'variational'
SHORT_METHODS = 'LA', 'QLA', 'GQ', 'VQ'
COLORS = 'blue', 'green', 'red', 'orange'
NITERS = 5
BETA = 1
PRIOR_VAR = 25
BATCH_SIZE = 10
ANNEALING_STEP = 0.0
MINIMIZER = 'newton'

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
    return y, X


class ClassifEP(object):

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

    def run(self, batch_size, beta, method, init_mean=None, niters=5, annealing_step=0):
        factors = slices(self.npts, batch_size)
        nfactors = len(factors)
        beta = float(beta)

        def target(theta, a):
            tmp = [self.cost(theta, k) for k in factors[a]]
            if theta.ndim > 1:
                return -beta * np.sum(tmp, 0)
            else:
                return -beta * np.sum(tmp)

        def gradient_target(theta, a):
            tmp = [self.gradient_cost(theta, k) for k in factors[a]]
            return -beta * np.sum(tmp, 0).squeeze()

        def hessian_target(theta, a):
            tmp = [self.hessian_cost(theta, k) for k in factors[a]]
            return -beta * np.sum(tmp, 0).squeeze()

        prior = FactorGaussian(np.zeros(self.dim), self.prior_var * np.ones(self.dim))
        A = NumEP(target, range(nfactors), prior, method=method, minimizer=MINIMIZER, gradient=gradient_target, hessian=hessian_target)
        #A = NumEP(target, range(nfactors), prior, method=method, minimizer=MINIMIZER)
        C = []
        c = self.total_cost(A.gaussian.m)
        cr = self.total_regularized_cost(A.gaussian.m)
        C.append((c, cr))
        dt = []

        for i in range(niters):
            t0 = time()
            # simulated annealing
            beta += annealing_step
            #print('Temperature = %f' % (1/float(beta)))
            #print('Iter %d/%d' % (i+1, niters))
            for a in range(nfactors):
                A.update_factor(a)
                c = self.total_cost(A.gaussian.m)
                cr = self.total_regularized_cost(A.gaussian.m)
                C.append((c, cr))
                dt.append(time() - t0)

        return A, np.array(C), 1000 * np.array(dt) / float(nfactors)


# Load data
y, X = load_data()

LR = LogisticRegression(y, X, prior_var=PRIOR_VAR)
mgt = LR.map()

EP = ClassifEP(y, X, loss=LOSS, prior_var=PRIOR_VAR)
"""
npts, dim = X.shape
theta = np.random.rand(dim)
c = EP.cost(theta, 0)
"""

if LOSS != 'logistic':
    from scipy.optimize import fmin_powell
    cost = lambda theta: EP.total_regularized_cost(theta)
    mgt = fmin_powell(cost, mgt)


# Perform simulations
A, C, dt = {}, {}, {}
for met in METHODS:
    A[met], C[met], dt[met] = EP.run(BATCH_SIZE, BETA, met, niters=NITERS, annealing_step=ANNEALING_STEP)

for met in METHODS:
    mt = np.mean(dt[met])
    m1t = np.median(dt[met])
    print('Time per mini-batch iteration [%s]: mean=%f (ms) median=%f (ms)' % (met, mt, m1t))


