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
"""

import numpy as np

from variana.maxent import MaxentModel, ConditionalMaxentModel, MaxentModelGKL

N_FEATURES = 3
SIZE = 100
HOMOSCEDASTIC = False
GAUSS = .5 * np.log(2 * np.pi)


def mahalanobis(z, m, s):
    return ((z - m) / s) ** 2


def log_lik1d(z, m, s):
    return - (GAUSS + np.log(s) + .5 * mahalanobis(z, m, s))


def mean_log_lik1d(s):
    return - (GAUSS + np.log(s) + .5)


def random_means(n):
    return 3 * (np.random.rand(N_FEATURES) - .5)


def random_devs(n):
    return 1 + np.random.rand(N_FEATURES)


true_means = np.array((random_means(N_FEATURES), random_means(N_FEATURES)))
if HOMOSCEDASTIC:
    devs = random_devs(N_FEATURES)
    true_devs = np.array((devs, devs))
else:
    true_devs = np.array((random_devs(N_FEATURES), random_devs(N_FEATURES)))

labels = np.random.randint(2, size=SIZE)
noise = np.random.normal(size=(SIZE, N_FEATURES))
data = true_means[labels] +  noise * true_devs[labels]

means = np.array([np.mean(data[labels==0], 0), np.mean(data[labels==1], 0)])
if HOMOSCEDASTIC:
    devs = np.std(data, 0)
    devs = np.array((devs, devs))
else:
    devs = np.array([np.std(data[labels==0], 0), np.std(data[labels==1], 0)])

    
basis = lambda x, y, i: log_lik1d(y[i], means[x, i], devs[x, i])
moments = np.mean(mean_log_lik1d(devs[labels]), 0)
m = ConditionalMaxentModel(2, basis, moments, data)
m.fit(positive_weights=True)


###################
# Checks
###################

# Weights
print('Optimal weights = %s' % m.weights)

# Mean-value constraints
print('Dual function value = %f' % m.dual(m.weights))
print('Gradient: %s' % m.gradient_dual(m.weights))
print('Hessian: %s' % m.hessian_dual(m.weights))

# Dual should equate with KL-divergence
bayes_factors = m.dist() / m.prior
div = np.mean(np.log([r[x] for x, r in zip(labels, bayes_factors)]))
print('Actual KL divergence = %f' % div)

