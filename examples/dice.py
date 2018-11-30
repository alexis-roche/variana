import sys
import numpy as np
from variana.maxent import MaxentModel

avg_spots = 3
if len(sys.argv) > 1:
    avg_spots = float(sys.argv[1])

m = MaxentModel(6, lambda x, i: x + 1, [avg_spots])
m.fit()
pdist = m.dist()

print('Weights: %s' % m.weights)
print('Maxent distribution: %s' % pdist)
print('Average number of dots: expected = %f, achieved = %f' % (avg_spots, np.sum(pdist*np.arange(1, 7))))
