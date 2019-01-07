import sys
import numpy as np
from variana.maxent import Maxent

avg_spots = 3.5
if len(sys.argv) > 1:
    avg_spots = float(sys.argv[1])

m = Maxent(np.arange(1, 7), avg_spots, tiny=0)
m.fit()
pdist = m.dist()

print('Weights: %s' % m.weight)
print('Maxent distribution: %s' % pdist)
print('Average number of dots: expected = %f, achieved = %f' % (avg_spots, np.sum(pdist * np.arange(1, 7))))


