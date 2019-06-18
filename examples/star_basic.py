import numpy as np
from variana.dist_fit import StarApproximation, LaplaceApproximation
from variana.toy_dist import ExponentialPowerLaw

              
dim = 3
vmax = 1e4
alpha = 0.1
learning_rate = 1
beta = 2
if dim < 3:
    block_size = None
else:
    block_size = (dim // 3, dim // 3, dim - 2 * (dim // 3))
niter = 100

# Target distribution
K = np.random.rand()
c = 5 * (np.random.rand(dim) - .5)
target = ExponentialPowerLaw(c, np.ones(dim), logK=np.log(K))

# Laplace approximation
l = LaplaceApproximation(target.log, np.zeros(dim))
ql = l.fit()

# Star approximation
v = StarApproximation(target.log, (np.zeros(dim), np.ones(dim)),
                      alpha, vmax, learning_rate=learning_rate, block_size=block_size)
q = v.fit(niter=niter)

print('Error on factor = %f' % np.abs(q.logZ - target.logZ))
print('Error on mean = %f' % np.max(np.abs(q.m - target.m)))
print('Error on variance = %f' % np.max(np.abs(q.v - target.v)))
