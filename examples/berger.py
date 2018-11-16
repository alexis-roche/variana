from variana.maxent import MaxentModel

def f1(x):
    return x < 2

def f2(x):
    return x==0 or x==2


m = MaxentModel(5, (f1, f2), (0.3, 0.5))

m.fit()

p = m.dist()
