from .dist_fit import VariationalSampling, LaplaceApproximation
from .gaussian import Gaussian, FactorGaussian
from .toy_dist import ExponentialPowerLaw
from .dist_model import (Maxent,
                         ConditionalMaxent,
                         GaussianCompositeInference,
                         LogisticRegression,
                         MininfLikelihood)

def test():
    """
    Run the test suite.
    """
    import pytest
    import pathlib
    pkg_dir = pathlib.Path(__file__).resolve().parent
    return pytest.main([str(pkg_dir / "tests")])

