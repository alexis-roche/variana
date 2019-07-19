import numpy as np
from variana.dist_fit import BridgeApproximation, LaplaceApproximation
from variana.gaussian import FactorGaussian


def toy_score(x, center=None, K=1, power=2, proper=True):
    if not center is None:
        x = x - center
    return np.log(K) - np.sum(((2 * proper - 1) ** np.arange(len(x))) * np.abs(x) ** power, 0) / power


def pseudo_hessian(q):
    m2 = q.m ** 2
    mom1 = m2 + q.v
    mom2 = 3 * q.v ** 2 + 6 * q.v * m2 + m2 ** 2
    return q.Z * np.append(1, np.concatenate((mom1, mom2)))


dim = 5
power = 2

#K = 1
K = np.random.rand()
#c = np.zeros(dim)
c = 5 * (np.random.rand(dim) - .5)

# Laplace approximation
log_target = lambda x: toy_score(x, c, K, power)
l = LaplaceApproximation(log_target, np.zeros(dim))
ql = l.fit()

# Online VS
alpha = 0.1
beta = 1 / (100 * dim)
vmax = 100
niter = 1000
nsubiter = 10 * dim
eps0 =  1
decay = 0

## Log-target
log_factor = lambda x: alpha * toy_score(x, c, K, power)

## Init 
q = FactorGaussian(np.zeros(dim), np.full(dim, vmax), logK=0)
rec = []

u = np.zeros(2 * dim + 1)
v = np.zeros(2 * dim + 1)

for j in range(niter):
    print('Iteration: %d' % j)
    w = q ** (1 - alpha)
    eps = eps0 / (1 + decay * j)
    ## Loop to minimize D(wf||wg)
    for i in range(nsubiter):
        x = np.sqrt(w.v) * np.random.normal(size=dim) + w.m
        aux = q(x) ** alpha
        delta = aux - np.exp(log_factor(x))
        phi_x = np.append(1, np.concatenate((x, x ** 2)))
        u = (1 - beta) * u + beta * delta * phi_x
        v = (1 - beta) * v + beta * aux * (phi_x ** 2)
        rec.append(delta)

    ###vv = pseudo_hessian(q) / w.Z
    ###aux = vv / v
    ###print('ratio = %f' % (aux.min() / aux.max()))
    
    theta = q.theta - eps * (u / v)
    theta[(dim + 1):] = np.minimum(theta[(dim + 1):], -.5 / vmax)
    q = FactorGaussian(theta=theta)

    
rec = np.array(rec)

