from __future__ import absolute_import

import numpy as np

from ..gaussian import *

from numpy.testing import assert_array_equal


SQRT_TWO_PI = np.sqrt(2 * np.pi)



def _test_init_gaussian(g, dim, Z=1, m=None, v=None, V=None, factor_gaussian=False):
    if m is None:
        m = np.zeros(dim)
    if v is None:
        v = np.ones(dim)
    if V is None:
        V = np.eye(dim)
    assert g.dim == dim
    assert g.m.shape == (dim,)
    if factor_gaussian:
        assert g.v.shape == (dim,)
    assert g.V.shape == (dim, dim)
    assert (abs(g.Z - Z) / Z) < 1e-10
    assert_array_equal(g.m, m)
    if factor_gaussian:
        assert_array_equal(g.v, v)
    assert_array_equal(g.V, V)


def test_init_normalized_factor_gaussian_dim1():
    g = FactorGaussian(np.zeros(1), np.ones(1))
    _test_init_gaussian(g, 1, factor_gaussian=True)


def test_init_normalized_factor_gaussian_dim10():
    g = FactorGaussian(np.zeros(10), np.ones(10))
    _test_init_gaussian(g, 10, factor_gaussian=True)


def test_init_normalized_factor_gaussian_dim100():
    g = FactorGaussian(np.zeros(100), np.ones(100))
    _test_init_gaussian(g, 100, factor_gaussian=True)


def test_init_normalized_gaussian_dim1():
    g = Gaussian(np.zeros(1), np.eye(1))
    _test_init_gaussian(g, 1)


def test_init_normalized_gaussian_dim10():
    g = Gaussian(np.zeros(10), np.eye(10))
    _test_init_gaussian(g, 10)


def test_init_normalized_gaussian_dim100():
    g = Gaussian(np.zeros(100), np.eye(100))
    _test_init_gaussian(g, 100)

    
def _test_laplace_approximation(dim):
    g = laplace_approximation(np.zeros(dim), 0, np.zeros(dim), -np.ones(dim))
    _test_init_gaussian(g, dim, Z=SQRT_TWO_PI**dim, factor_gaussian=True)


def test_laplace_approximation_dim1():
    _test_laplace_approximation(1)


def test_laplace_approximation_dim10():
    _test_laplace_approximation(10)


def test_laplace_approximation_dim100():
    _test_laplace_approximation(100)


def _test_evaluate_normalized_gaussian(dim):
    gf = FactorGaussian(np.zeros(dim), np.ones(dim))
    g = Gaussian(np.zeros(dim), np.eye(dim))
    x = np.random.random(dim)
    y = g(x)
    xs = np.random.random((dim, 10))
    ys = g(xs)
    assert isinstance(y, float)
    assert y == gf(x)
    assert ys.shape == (10,)
    assert_array_equal(ys, gf(xs))
    

def test_evaluate_normalized_gaussian_dim1():
    _test_evaluate_normalized_gaussian(1)


def test_evaluate_normalized_gaussian_dim10():
    _test_evaluate_normalized_gaussian(10)


def test_evaluate_normalized_gaussian_dim100():
    _test_evaluate_normalized_gaussian(100)



