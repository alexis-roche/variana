from __future__ import absolute_import

import numpy as np

from ..gaussian import *

from nose.tools import (assert_true, 
                        assert_false, 
                        assert_raises)
from numpy.testing import (assert_array_equal, 
                           assert_array_almost_equal,
                           assert_almost_equal)


SQRT_TWO_PI = np.sqrt(2 * np.pi)


def _test_factor_gaussian(g, dim, Z=1, m=None, v=None, V=None):
    if m is None:
        m = np.zeros(dim)
    if v is None:
        v = np.ones(dim)
    if V is None:
        V = np.eye(dim)
    assert_true(g.dim == dim)
    assert_true(len(g.m) == dim)
    assert_true(len(g.v) == dim)
    assert_true(g.V.shape == (dim, dim))
    assert_true((abs(g.Z - Z) / Z) < 1e-10)
    assert_array_equal(g.m, m)
    assert_array_equal(g.v, v)
    assert_array_equal(g.V, V)


def _test_gaussian(g, dim, Z=1, m=None, V=None):
    if m is None:
        m = np.zeros(dim)
    if V is None:
        V = np.eye(dim)
    assert_true(g.dim == dim)
    assert_true(len(g.m) == dim)
    assert_true(g.V.shape == (dim, dim))
    assert_true((abs(g.Z - Z) / Z) < 1e-10)
    assert_array_equal(g.m, m)
    assert_array_equal(g.V, V)


def test_normalized_factor_gaussian_dim1():
    g = FactorGaussian(np.zeros(1), np.ones(1))
    _test_factor_gaussian(g, 1)


def test_normalized_factor_gaussian_dim10():
    g = FactorGaussian(np.zeros(10), np.ones(10))
    _test_factor_gaussian(g, 10)


def test_normalized_factor_gaussian_dim100():
    g = FactorGaussian(np.zeros(100), np.ones(100))
    _test_factor_gaussian(g, 100)


def test_normalized_gaussian_dim1():
    g = Gaussian(np.zeros(1), np.eye(1))
    _test_gaussian(g, 1)


def test_normalized_gaussian_dim10():
    g = Gaussian(np.zeros(10), np.eye(10))
    _test_gaussian(g, 10)


def test_normalized_factor_dim100():
    g = Gaussian(np.zeros(100), np.eye(100))
    _test_gaussian(g, 100)


def _test_laplace_approximation(dim):
    g = laplace_approximation(np.zeros(dim), 0, np.zeros(dim), -np.ones(dim))
    _test_factor_gaussian(g, dim, Z=SQRT_TWO_PI**dim)


def test_laplace_approximation_dim1():
    _test_laplace_approximation(1)


def test_laplace_approximation_dim10():
    _test_laplace_approximation(10)


def test_laplace_approximation_dim100():
    _test_laplace_approximation(100)

