from variana.maxent import MaxentModel


# Dice
def f1(x):
    return x % 2 

def f2(x):
    return x == 3

m = MaxentModel(6, [f1, f2], [.5, .5])
m.fit()

print(m._w)
print(m.dist())
