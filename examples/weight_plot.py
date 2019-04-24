import sys
import numpy as np
import pylab as pl
from variana.maxent import GaussianCompositeInference
from sklearn.datasets import (load_iris,
                              load_digits,
                              load_breast_cancer,
                              load_wine)


loader = {'iris': load_iris,
          'digits': load_digits,
          'breast_cancer': load_breast_cancer,
          'wine': load_wine}

dataset = 'breast_cancer'
savefig = False
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    if len(sys.argv) > 2:
        save_fig = bool(int(sys.argv[2]))

data = loader[dataset]()
m = GaussianCompositeInference(data.data, data.target, damping=1, homo_sced=True)
m.fit()

devs = np.maximum(m._devs, 1e-5)
rho2 = (devs[:, None, :] / devs[None, :, :]) ** 2
delta = (m._means[:, None, :] - m._means[None, :, :]) / devs
div = .5 * (delta ** 2 - np.log(rho2) + rho2 - 1)
R = div * m.weight
disc = np.max(div, (0, 1))

pl.figure()
pl.plot(disc, m.weight, 'o')
pl.xlabel('Feature discrimination (symmetric KL divergence)')
pl.ylabel('Composite weight')
pl.show()

if savefig:
    pl.savefig('disc_weight_plot.pdf', bbox_inches='tight')

I = np.argsort(disc)
print(np.asarray(data.feature_names)[I])
print(m.weight[I])
