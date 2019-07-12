"""TODO

Relier cet algorithme à la descente stochastique 2e ordre que nous
avons implémetentée pour ReidNet (SEP).

Comprendre les conditions de fonctionnement en grande
dimension. A-t-on besoin de travailler par blocs de paramètres?

A priori sur les paramètres de 2nd ordre de la forme: log pi(theta) =
lda * theta?

Scale initial variance automatically. Should we start from small or
large variance??? Clearly, the best is to start from a good guess. For
CNN, we can use standard tensor flow random initializer as a starting
point.

Optimal parameters? 

For proxy == 'likelihood', we seem to need niter *= 100 and gamma /=
100 for similar convergence as for proxy == 'discrete_kl'.
"""
import numpy as np
import pylab as pl

from variana.dist_fit import LaplaceApproximation, OnlineIProj, OnlineContextFit, OnlineMProj, OnlineStarFit, OnlineLaplaceFit
from variana.toy_dist import ExponentialPowerLaw

dim = 10
beta = 2
vmax = 1e2
K = np.random.rand()
m = 5 * (np.random.rand(dim) - .5)
s2 = 5 * (np.random.rand(dim) + 1)
###K, m, v = 1, 0, 1

# Target definition
target = ExponentialPowerLaw(m, s2, logK=np.log(K), beta=beta)

# Laplace approximation
l = LaplaceApproximation(target.log, np.zeros(dim), grad=target.grad_log, hess_diag=target.hess_diag_log)
ql = l.fit()

# Online parameters
proxy = 'discrete_kl'
###proxy = 'likelihood'
k_gamma = .01
gamma_s = k_gamma / np.sqrt(dim)
alpha = .1
###alpha = .1 / np.sqrt(dim)
gamma = gamma_s / alpha
niter = int(10 / gamma_s)

print('Dimension = %d, beta = %f' % (dim, beta))

"""
q0 = OnlineContextFit(target.log, (np.zeros(dim), np.ones(dim)), gamma_s, vmax=vmax, proxy=proxy)
q0.run(niter, record=True)
g = q0.factor_fit()
q0.disp('context')
"""

print('I-projection: gamma = %f' % gamma_s)
q = OnlineIProj(target.log, (np.zeros(dim), np.ones(dim)), gamma_s, vmax=vmax)
q.ground_truth(target.logZ, target.m, target.v)
q.run(niter, record=True)
print('Error = %3.2f %3.2f %3.2f' % q.error())
q.disp('I-projection')

print('Star fit: alpha = %f, gamma = %f' % (alpha, gamma))
qs = OnlineStarFit(target.log, (np.zeros(dim), np.ones(dim)), alpha, gamma, vmax=vmax, proxy=proxy)
qs.ground_truth(target.logZ, target.m, target.v)
qs.run(niter, record=True)
"""
qs.reset(alpha=.5)
qs.run(niter, record=True)
"""
print('Error = %3.2f %3.2f %3.2f' % qs.error())
qs.disp('star (alpha=%.2f)' % alpha)

print('Online Laplace fit: alpha = %f' % alpha)
qst = OnlineLaplaceFit(target.log, (np.zeros(dim), np.ones(dim)), alpha, vmax=vmax, grad=target.grad_log, hess_diag=target.hess_diag_log)
qst.ground_truth(target.logZ, target.m, target.v)
qst.run(niter, record=True)
print('Error = %3.2f %3.2f %3.2f' % qst.error())
qst.disp('star-taylor (alpha=%.2f)' % alpha)

print('M-projection: gamma = %f' % gamma_s)
q1 = OnlineMProj(target.log, qs, gamma_s, lda=.75)
q1.ground_truth(target.logZ, target.m, target.v)
q1.run(niter, nsubiter=1, record=True)
print('Error = %3.2f %3.2f %3.2f' % q1.error())
q1.disp('M-projection')
