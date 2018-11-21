from variana.maxent import MaxentModel


def basis(x, i):
    if i == 0:
        return x % 2 
    elif i == 1:
        return x == 3


m = MaxentModel(6, basis, [.6, .3])
m.fit()

print(m._w)
print(m.dist())
