import os
import sys
import numpy as np
from scipy.optimize import fmin_ncg, fmin_l_bfgs_b

from variana.dist_fit import BridgeApproximation, LaplaceApproximation


def logistic(x):
    return 1 / (1 + np.exp(np.minimum(-x, 100)))


class LogisticRegression(object):

    def __init__(self, y, X, prior_var):
        """
        Logistic regression object

        y: 1d array of +/- 1, length nb subjects
        X: nb subjects x nb features 2d array

        Will compute the distribution of
        w: 1d array, length nb features
        """
        self._y = y
        self._X = X
        self._prior_var = prior_var
        self._w = None
        self._f = None
        self._yf = None
        self._p = None
        self._wmax = None

    def update_cache(self, w):
        if not np.array_equal(w, self._w):
            f = np.dot(self._X, w)
            yf = self._y * f
            self._f = f
            self._yf = yf
            self._p = logistic(yf)
            self._w = w.copy()

    def posterior(self, w):
        self.update_cache(w)
        return np.prod(self._p)

    def log_likelihood(self, w):
        self.update_cache(w)
        return np.sum(np.log(np.maximum(self._p, 1e-100)))

    def log_posterior(self, w):
        self.update_cache(w)
        return self.log_likelihood(w) + self.log_prior(w)

    def log_posterior_grad(self, w):
        self.update_cache(w)
        a = self._y * (1 - self._p)
        return np.dot(self._X.T, a) + self.log_prior_grad(w)

    def log_posterior_hess(self, w):
        self.update_cache(w)
        a = self._p * (1 - self._p)
        return -np.dot(a * self._X.T, self._X) + self.log_prior_hess(w)

    def log_prior(self, w):
        return -.5 * np.sum(w ** 2) / self._prior_var

    def log_prior_grad(self, w):
        return -w / self._prior_var

    def log_prior_hess(self, w):
        log_prior_hess = np.zeros((w.size, w.size))
        np.fill_diagonal(log_prior_hess, -1 / self._prior_var)
        return log_prior_hess

    def fit(self, tol=1e-6):
        """
        Compute the maximum a posteriori regression coefficients.
        """
        cost = lambda w: -self.log_posterior(w)
        grad = lambda w: -self.log_posterior_grad(w)
        hess = lambda w: -self.log_posterior_hess(w)
        w0 = np.zeros(self._X.shape[1])
        w = fmin_ncg(cost, w0, grad, fhess=hess, avextol=tol, disp=True)
        #w = fmin_l_bfgs_b(cost, w0, grad, pgtol=tol, disp=True)
        self._wmax = w
        return w

    def accuracy(self, w=None, klass=None):
        """
        Compute the correct classification rate for a given set of
        regression coefficients.
        """
        if w is None:
            w = self._wmax
        self.update_cache(w)
        if klass in (-1, 1):
            msk = np.where(self._y == klass)
        else:
            msk = slice(0, self._y.size)
        yf = self._yf[msk]
        y = self._y[msk]
        errors = np.where(yf < 0)
        return float(y.size - len(errors[0])) / y.size

    @property
    def weight(self):
        return self._wmax

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y


def load_data(dataset):
    try:
        data = np.loadtxt(os.path.join('data', '%s.data' % dataset), delimiter=',')
    except:
        data = np.loadtxt(os.path.join('data', '%s.data' % dataset), delimiter=',', dtype=str)

    label = data[:, -1]
    try:
        positive = label.max()
    except:
        positive = label[0]
    target = 2 * (label == positive) - 1
    features = data[:, 0:-1].astype(float)

    return target, features


def make_design(features, baseline=True):
    # normalize to unit norm
    features = features - np.mean(features, 0)
    norm = np.sqrt(np.sum(features ** 2, 0))
    keep = (norm > 1e-5)
    X = features[:, keep] / norm[keep]
    # add baseline regressor
    if baseline:
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    return X


def disp_likelihood(idx):
    import pylab as pl
    w = lr.weight.copy()
    x = w[idx] + np.linspace(-np.sqrt(prior_var), np.sqrt(prior_var))
    log_t = []
    for xx in x:
        w[idx] = xx
        log_t.append(lr.log_likelihood(w))
    log_t = np.array(log_t)
    pl.figure()
    pl.plot(x, log_t)
    pl.ylabel('Log-likelihood')
    pl.xlabel('Parameter (%d)' % idx)
    pl.show()


######################################################################
# Main
######################################################################

dataset = 'haberman'
if len(sys.argv) > 1:
    dataset = sys.argv[1]

target, features = load_data(dataset)

# Classical logistic regression
prior_var = 1e4
X = make_design(features)
lr = LogisticRegression(target, X, prior_var)
lr.fit()

# Laplace approximation
l = LaplaceApproximation(lr.log_posterior, lr.weight, grad=lr.log_posterior_grad, hess=lr.log_posterior_hess)
ql = l.fit()

# Bridge approximation
alpha = .5
learning_rate = .1
niter = 100
log_target = lambda w: lr.log_posterior(w)
stride = None
start = (np.zeros(lr.X.shape[1]), np.full(lr.X.shape[1], prior_var))
v = BridgeApproximation(log_target, start, alpha, prior_var, learning_rate=learning_rate, stride=stride, method='laplace')
q = v.fit(niter=niter)

# Print out some stuff
print('Laplace')
print('Log-likelihood = %f, Peak = %f' % (lr.log_likelihood(ql.m), ql.logK))
print('Bridge')
print('Log-likelihood = %f, Peak = %f' % (lr.log_likelihood(q.m), q.logK))
