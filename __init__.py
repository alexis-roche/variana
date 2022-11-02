from .dist_fit import VariationalSampling, LaplaceApproximation
from .gaussian import Gaussian, FactorGaussian
from .toy_dist import ExponentialPowerLaw
from .dist_model import (Maxent,
                         ConditionalMaxent,
                         GaussianCompositeInference,
                         LogisticRegression,
                         MininfLikelihood)

from numpy.testing import Tester

test = Tester().test
bench = Tester().bench
