import numpy as np

from variana.maxent import GaussianCompositeInference

SIZE = 100
CLASSES = 3
FEATURES = 2
FEATURE_CORRELATION = 0.2
POSITIVE_WEIGHT = False
PRIOR = None  # 'empirical'
HOMO_SCED = False
REF_CLASS = 0
OFFSET = True


def random_means(classes, features):
    return 3 * (np.random.random(size=(classes, features)) - .5)


def random_devs(classes, features):
    return 1 + np.random.random(size=(classes, features))


def generate_noise(size, dim, autocorr=0):
    wn = np.random.normal(size=(size, dim))
    if autocorr == 0:
        return wn
    I, J = np.mgrid[0:dim, 0:dim]
    A2 = np.exp(np.abs(I-J) * np.log(np.maximum(autocorr, 1e-10)))
    v, P = np.linalg.eigh(A2)
    A = np.dot(P * np.sqrt(v), P.T)
    return np.dot(wn, A)


true_means = random_means(CLASSES, FEATURES)
if HOMO_SCED:
    true_devs = np.repeat(random_devs(1, FEATURES), CLASSES, axis=0)   
else:
    true_devs = random_devs(CLASSES, FEATURES)

target = np.random.randint(CLASSES, size=SIZE)
data = true_means[target] +  generate_noise(SIZE, FEATURES, FEATURE_CORRELATION) * true_devs[target]

m = GaussianCompositeInference(data, target, prior=PRIOR, positive_weight=POSITIVE_WEIGHT, ref_class=REF_CLASS, homo_sced=HOMO_SCED, offset=OFFSET)
m.fit()


###################
# Checks
###################

# Parameters
print('Optimal parameters = %s' % m.param)

# Mean-value constraints
print('Dual function value = %f' % m.dual(m.param))
print('Gradient: %s' % m.gradient_dual(m.param))
###print('Hessian: %s' % m.hessian_dual(m.param))

# Dual should equate with KL-divergence
bayes_factors = m.dist() / m.prior
div = np.sum(m.data_weight * np.log(bayes_factors[range(SIZE), target]))
print('Actual KL divergence = %f' % div)

# Tests
div_test = np.abs(div - m.dual(m.param))
g =  m.gradient_dual(m.param)

grad_test1 = True
if np.max(m.param) > 0:
    grad_test1 = np.max(g[m.param > 0])
grad_test2 = True
if np.min(m.param) == 0:
    grad_test2 = np.max(g[m.param == 0]) < 0

print('Div test: %f' % div_test)
print('Grad test 1: %f' % grad_test1)
print('Grad test 2: %s' % grad_test2)

