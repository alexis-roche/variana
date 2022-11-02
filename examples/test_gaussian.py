import numpy as np

from variana import Gaussian, FactorGaussian


def test1(dim):
    K = np.random.rand()
    m = np.random.rand(dim) - .5
    A = np.random.rand(dim, dim)
    V = np.dot(A, A.T)
    g = Gaussian(m, V, logK=np.log(K))
    g2 = Gaussian(theta=g.theta)
    print('Same K: %s' % np.allclose(g.K, g2.K))
    print('Same m: %s' % np.allclose(g.m, g2.m))
    print('Same V: %s' % np.allclose(g.V, g2.V))
    return g, g2


def test2(dim):
    K = np.random.rand()
    m = np.random.rand(dim) - .5
    v = np.random.rand(dim)
    g = FactorGaussian(m, v, logK=np.log(K))
    g2 = FactorGaussian(theta=g.theta)
    print('Same K: %s' % np.allclose(g.K, g2.K))
    print('Same m: %s' % np.allclose(g.m, g2.m))
    print('Same v: %s' % np.allclose(g.v, g2.v))
    return g, g2


    
    
