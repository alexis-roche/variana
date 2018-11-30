"""
z|x ~ N(mx, sx)

Homoscedastic sx = s
=> p(x|z) 
eq exp{-.5[(z-mx)/s]^2}
eq exp( z mx/s^2 - .5 * mx^2/s^2 )

=> n + 1 generative parameters
=> n discriminative parameters
Comparable generative model: n parameters

Heteroscedasticity
=> p(x|z) 
eq exp{-.5[(z-mx)/sx]^2}
eq exp( -.5 * (z/sx)^2 + (z mx/sx^2) - .5 * mx^2 )

=> 2n generative parameters
=> n discriminative parameters
Comparative generative model: 2n parameters


N = (100, 3)

x = N(0, I3)
Ax = N(0, AAT)

AAT = exp(-d/a)

d=0 ==> AAT = 1
d=1 ==> exp(-1/a) = c

a = 1/( -log(c))

"""

import numpy as np

from variana.maxent import MaxentModel, ConditionalMaxentModel, MaxentModelGKL

SIZE = 100
FEATURES = 3
FEATURE_CORRELATION = 0.5
HOMOSCEDASTIC = False
POSITIVE_WEIGHTS = False

GAUSS = .5 * np.log(2 * np.pi)


def mahalanobis(z, m, s):
    return ((z - m) / s) ** 2


def log_lik1d(z, m, s):
    return - (GAUSS + np.log(s) + .5 * mahalanobis(z, m, s))


def mean_log_lik1d(s):
    return - (GAUSS + np.log(s) + .5)


def random_means(n):
    return 3 * (np.random.rand(FEATURES) - .5)


def random_devs(n):
    return 1 + np.random.rand(FEATURES)


def generate_noise(size, dim, autocorr=0):
    wn = np.random.normal(size=(size, dim))
    if autocorr == 0:
        return wn
    I, J = np.mgrid[0:dim, 0:dim]
    A2 = np.exp(np.abs(I-J) * np.log(np.maximum(autocorr, 1e-10)))
    v, P = np.linalg.eigh(A2)
    A = np.dot(P * np.sqrt(v), P.T)
    return np.dot(wn, A)


true_means = np.array((random_means(FEATURES), random_means(FEATURES)))
if HOMOSCEDASTIC:
    devs = random_devs(FEATURES)
    true_devs = np.array((devs, devs))
else:
    true_devs = np.array((random_devs(FEATURES), random_devs(FEATURES)))

labels = np.random.randint(2, size=SIZE)
data = true_means[labels] +  generate_noise(SIZE, FEATURES, FEATURE_CORRELATION) * true_devs[labels]

means = np.array([np.mean(data[labels==0], 0), np.mean(data[labels==1], 0)])
if HOMOSCEDASTIC:
    devs = np.std(data, 0)
    devs = np.array((devs, devs))
else:
    devs = np.array([np.std(data[labels==0], 0), np.std(data[labels==1], 0)])


basis = lambda x, y, i: log_lik1d(y[i], means[x, i], devs[x, i])
moments = np.mean(mean_log_lik1d(devs[labels]), 0)
m = ConditionalMaxentModel(2, basis, moments, data)
m.fit(positive_weights=POSITIVE_WEIGHTS, verbose=True)


###################
# Checks
###################

# Weights
print('Optimal weights = %s' % m.weights)

# Mean-value constraints
print('Dual function value = %f' % m.dual(m.weights))
print('Gradient: %s' % m.gradient_dual(m.weights))
###print('Hessian: %s' % m.hessian_dual(m.weights))

# Dual should equate with KL-divergence
bayes_factors = m.dist() / m.prior
div = np.mean(np.log([r[x] for x, r in zip(labels, bayes_factors)]))
print('Actual KL divergence = %f' % div)

# Tests
div_test = np.abs(div - m.dual(m.weights))
g =  m.gradient_dual(m.weights)

grad_test1 = True
if np.max(m.weights) > 0:
    grad_test1 = np.max(g[m.weights > 0])
grad_test2 = True
if np.min(m.weights) == 0:
    grad_test2 = np.max(g[m.weights == 0]) < 0

print('Div test: %f' % div_test)
print('Grad test 1: %f' % grad_test1)
print('Grad test 2: %s' % grad_test2)
