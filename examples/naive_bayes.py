import numpy as np

from variana.maxent import GaussianCompositeInference

SIZE = 100
CLASSES = 3
FEATURES = 2
FEATURE_CORRELATION = 0.9
POSITIVE_WEIGHTS = False
PRIOR = None  # 'empirical'
HOMOSCEDASTIC = False
SUPERCOMPOSITE = False


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
if HOMOSCEDASTIC:
    true_devs = np.repeat(random_devs(1, FEATURES), CLASSES, axis=0)   
else:
    true_devs = random_devs(CLASSES, FEATURES)

labels = np.random.randint(CLASSES, size=SIZE)
data = true_means[labels] +  generate_noise(SIZE, FEATURES, FEATURE_CORRELATION) * true_devs[labels]

m = GaussianCompositeInference(data, labels, prior=PRIOR, supercomposite=SUPERCOMPOSITE, homoscedastic=HOMOSCEDASTIC)
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
div = np.sum(m.data_weights * np.log(bayes_factors[range(SIZE), labels]))
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

