import numpy as np
from variana import Maxent, ConditionalMaxent

optimizer = 'lbfgs'

def basis_fn(data=None):
    x = np.arange(6)
    basis = np.array((x < 2, (x==0) + (x==2))).T
    if data is None:
        return basis
    return np.array([np.abs(1 - 2 * y) * basis for y in data])


m = Maxent(basis_fn(), (.3, .5))
m.fit(optimizer=optimizer)
p = m.dist()
print(m.param)

data =  (0, 1, 0, 1, 0, 1, 0, 1)
mc = ConditionalMaxent(basis_fn, (.3, .5), data)
mc.fit(optimizer=optimizer)
pc = mc.dist(0)
print(mc.param)


