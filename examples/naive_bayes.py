import numpy as np

from variana.maxent import MaxentModel, ConditionalMaxentModel, MaxentModelGKL

N_FEATURES = 3
SIZE = 100


"""
def gauss_log_lik1d(z, m, s):
    s2 = s ** 2
    return -.5 * (np.log(2 * np.pi * s2) + (z - m) ** 2 / s2)


def mean_gauss_mahalanobis(s):
    s2 = s ** 2
    return -.5 * (np.log(2 * np.pi * s2) + 1)
"""

def mahalanobis(z, m, s):
    return ((z - m) / s) ** 2


def random_means(n):
    return 3 * (np.random.rand(N_FEATURES) - .5)


def random_devs(n):
    ##return np.random.rand(N_FEATURES)
    return np.ones(N_FEATURES)


true_means = np.array((random_means(N_FEATURES), random_means(N_FEATURES)))
true_devs = np.array((random_devs(N_FEATURES), random_devs(N_FEATURES)))

labels = np.random.randint(2, size=SIZE)
noise = np.random.normal(size=(SIZE, N_FEATURES))
data = true_means[labels] +  noise * true_devs[labels]

means = np.array([np.mean(data[labels==0], 0), np.mean(data[labels==1], 0)])
devs = np.array([np.std(data[labels==0], 0), np.std(data[labels==1], 0)])

basis = lambda x, y, i: -.5 * mahalanobis(y[i], means[x, i], devs[x, i])
moments = -.5 * np.ones(N_FEATURES)

m = ConditionalMaxentModel(2, basis, moments, data)
m.fit()


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

