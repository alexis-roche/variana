import numpy as np
from scipy.special import gamma


def safe_abs(x, tiny=1e-100):
    return np.maximum(np.abs(x), tiny)


class ExponentialPowerLaw():

    def __init__(self, m, s2, logK=0, beta=2.0, tiny=1e-100):
        """
        Distribution form:

        K * exp(- (|x-m|/s) ** beta)
        """
        self._m = np.asarray(m)
        self._dim = len(self._m)
        self._beta = float(beta)
        g3 = gamma(3 / self._beta)
        g1 = gamma(1 / self._beta)
        self._v = (g3 / g1) * np.asarray(s2)
        self._s = np.sqrt(np.asarray(s2))
        self._logK = logK
        self._logZ = logK + self._dim * np.log(2 * g1 / self._beta) + np.sum(np.log(self._s))

    def log(self, x):
        xc = (x - self._m) / self._s
        return self._logK - np.sum(safe_abs(xc) ** self._beta, 0)

    def grad_log(self, x):
        xc = (x - self._m) / self._s
        return -self._beta * xc * safe_abs(xc) ** (self._beta - 2) / self._s

    def hess_diag_log(self, x):
        xc = (x - self._m) / self._s
        return -self._beta * (self._beta - 1) * safe_abs(xc) ** (self._beta - 2) / (self._s ** 2)

    @property
    def m(self):
        return self._m

    @property
    def v(self):
        return self._v

    @property
    def logK(self):
        return self._logK

    @property
    def K(self):
        return np.exp(self._logK)

    @property
    def logZ(self):
        return self._logZ

    @property
    def Z(self):
        return np.exp(self._logZ)

    

