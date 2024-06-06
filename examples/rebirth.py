import sys
import argparse
import numpy as np

from variana import VariationalSampling, FactorGaussian

DIM = 10
BETA = 1
PROPER = True


class ToyDist(object):

    def __init__(self, K, m, v, beta=BETA, proper=PROPER):
        self._K = float(K)
        self._m = np.asarray(m)
        self._v = np.asarray(v)
        self._s = np.sqrt(self._v)
        self._beta = float(beta)
        self._proper = bool(PROPER)
        self._dim = len(self._m)

    def log(self, x):
        return np.log(self._K) \
            - np.sum(((2 * self._proper - 1) ** np.arange(self._dim))\
                     * np.abs((x - self._m) / self._s) ** self._beta, 0) / self._beta

    def __call__(self, x):
        return np.exp(self.log(x))
    
    @property
    def K(self):
        return self._K

    @property
    def m(self):
        return self._m

    @property
    def v(self):
        return self._v



############################################################
# MAIN
############################################################

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', default='lbfgs')
parser.add_argument('--family', default='factor_gaussian')
parser.add_argument('--ndraws')
args = parser.parse_args()

optimizer = args.optimizer
family = args.family
ndraws = None if args.ndraws is None else int(args.ndraws)

vmax = None if family == 'gaussian'\
    else 1e5
proxy = 'likelihood' if optimizer == 'likelihood'\
    else 'discrete_kl'
    

"""
K = 1
m = np.zeros(DIM)
v = np.ones(DIM)
"""
K = np.random.rand()
m = np.random.rand(DIM) - .5
v = 1 + np.random.rand(DIM)

f = ToyDist(K, m, v)
pi = FactorGaussian(np.zeros(DIM), np.ones(DIM))
vs = VariationalSampling(f.log, pi, ndraws=ndraws)
g, info = vs.fit(family=family, proxy=proxy, vmax=vmax, optimizer=optimizer, output_info=True)

print(info)
print(g)

rel_err = lambda x, y: np.max(np.abs(y - x)) / np.maximum(1, np.max(np.abs(x)))
print('Order-0 error = %f' % rel_err(f.K, g.K))
print('Order-1 error = %f' % rel_err(f.m, g.m))
print('Order-2 error = %f' % rel_err(f.v, g.v))

####################################################
# CUSTOM OPTIMIZER
####################################################
SQRT_TWO = np.sqrt(2)
vs2 = VariationalSampling(f.log, pi, ndraws=ndraws)

# Iteration
vs2._sample()

q = FactorGaussian(np.zeros(DIM), np.ones(DIM))

x = vs2._x[

phi1 = (x - q._m[:, None]) / np.sqrt(q._v[:, None])
phi2 = (phi1 ** 2 - 1) / SQRT_TWO

zob = np.vstack((np.ones(x.shape[1]), phi1, phi2)) / np.sqrt(q.Z)
