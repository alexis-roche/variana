import numpy as np
from scipy.optimize import fmin_ncg

force_positive = lambda x: np.maximum(x, 1e-25)


def logistic(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(object):

    def __init__(self, y, X, prior_var=None):
        """
        Logistic regression object

        y: 1d array of +/- 1, length nb subjects
        X: nb subjects x nb features 2d array

        Will compute the distribution of
        w: 1d array, length nb features
        """
        self.y = y
        self.X = X
        self.prior_var = prior_var
        self.cache = {'w': None, 'f': None, 'p': None, 'map': None}

    def update_cache(self, w):
        if not w is self.cache['w']:
            f = np.dot(self.X, w)
            yf = self.y * f
            self.cache['f'] = f
            self.cache['yf'] = yf
            self.cache['p'] = logistic(yf)
            self.cache['w'] = w

    def posterior(self, w):
        self.update_cache(w)
        return np.prod(self.cache['p'])

    def log_posterior(self, w):
        self.update_cache(w)
        return np.sum(np.log(self.cache['p'])) + self.log_prior(w)

    def log_posterior_grad(self, w):
        self.update_cache(w)
        a = self.y * (1 - self.cache['p'])
        return np.dot(self.X.T, a) + self.log_prior_grad(w)

    def log_posterior_hess(self, w):
        self.update_cache(w)
        a = self.cache['p'] * (1 - self.cache['p'])
        return -np.dot(a * self.X.T, self.X) + self.log_prior_hess(w)

    def log_prior(self, w):
        if self.prior_var is None:
            return 0
        tmp = -.5 * w.size * np.log(2 * np.pi * self.prior_var)
        return tmp - .5 * np.sum(w ** 2) / self.prior_var

    def log_prior_grad(self, w):
        if self.prior_var is None:
            return 0
        return -w / self.prior_var

    def log_prior_hess(self, w):
        if self.prior_var is None:
            return 0
        log_prior_hess = np.zeros((w.size, w.size))
        np.fill_diagonal(log_prior_hess, -1 / self.prior_var)
        return log_prior_hess

    def map(self, tol=1e-8):
        """
        Compute the maximum a posteriori regression coefficients.
        """
        cost = lambda w: -self.log_posterior(w)
        grad = lambda w: -self.log_posterior_grad(w)
        hess = lambda w: -self.log_posterior_hess(w)
        w0 = np.zeros(self.X.shape[1])
        w = fmin_ncg(cost, w0, grad, fhess=hess, avextol=tol, disp=False)
        self.cache['map'] = w
        return w

    def accuracy(self, w, klass=None):
        """
        Compute the correct classification rate for a given set of
        regression coefficients.
        """
        self.update_cache(w)
        if klass in (-1, 1):
            msk = np.where(self.y == klass)
        else:
            msk = slice(0, self.y.size)
        yf = self.cache['yf'][msk]
        y = self.y[msk]
        errors = np.where(yf < 0)
        return float(y.size - len(errors[0])) / y.size
