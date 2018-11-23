from variana.maxent import MaxentModel


def basis(x, i):
    if i == 0:
        return x % 2 
    elif i == 1:
        return x > 3


moments = [.6, .3]

m = MaxentModel(6, basis, moments)
m.fit()

pdist = m.dist()

print('Weights: %s' % m.weights)
print('Maxent distribution: %s' % pdist)
print('Probability of even number: expected=%1.2f, achieved=%f' % (moments[0], pdist[1::2].sum()))
print('Probability of 5 or 6: expected=%1.2f, achieved=%f' % (moments[1], pdist[4:].sum()))
