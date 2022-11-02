import numpy as np

EPS = 0.1

# Logistic loss and derivatives
logistic_loss = lambda y, f: np.log(1 + np.exp(-y * f))
logistic_d1 = lambda u: 1. / (1. + np.exp(-u))
def logistic_d2(u):
    tmp = np.exp(-u)
    tmp2 = 1. / (1 + tmp)
    return tmp * tmp2 ** 2

# Hinge loss and derivatives
hinge_loss = lambda y, f: np.maximum(0,  1 - y * f)
hinge_d1 = lambda u: .5 * (1 + np.sign(u + 1))
hinge_d2 = lambda u: 0.

# Quasi 0-1 loss and derivatives
def quasi01_loss(y, f):
    u = -y * f
    I3 = u > 0
    I2 = (True - I3) * (u > -EPS)
    return I3 * (1 + EPS * u) + I2 * (1 + u / EPS)

def quasi01_d1(u):
    I3 = u > 0 + .5 * (u == 0)
    I2 = (True - I3) * (u > -EPS) + .5 * (u == -EPS)
    return I3 * EPS + I2 * (1 / EPS)

quasi01_d2 = lambda u: 0.

