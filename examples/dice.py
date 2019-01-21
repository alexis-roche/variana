import sys
import numpy as np
from variana.maxent import Maxent

avg_spots = 3.5
optimizer = 'lbfgs'

if len(sys.argv) > 1:
    avg_spots = float(sys.argv[1])
    if len(sys.argv) > 2:
        optimizer = sys.argv[2]
        
m = Maxent(np.arange(1, 7), avg_spots, tiny=0)
info = m.fit(optimizer=optimizer)
pdist = m.dist()

print('Parameters: %s' % m.param)
print('Maxent distribution: %s' % pdist)
print('Average number of dots: expected = %f, achieved = %f' % (avg_spots, np.sum(pdist * np.arange(1, 7))))
print('Learning time: %f sec' % info['time'])


