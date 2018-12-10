from variana.maxent import *

def basis(x, i):
    if i == 0:
        return x < 2
    elif i == 1:
        return x in (0, 2)

m = Maxent(5, basis, (.3, .5))
m.fit()
p = m.dist()
print(m.weights)

m2 = MaxentGKL(5, basis, (.3, .5))
m2.fit()
p2 = m2.dist()
print(m2.weights)

data =  (0, 1, 0, 1, 0, 1, 0, 1)
m3 = ConditionalMaxent(5, lambda x, y, i: basis(x, i), (.3, .5), data)
m3.fit()
p3 = m3.dist(0)
print(m3.weights)

basis2 = lambda x, y, i: basis(x, i) * abs(1 - 2 * y)
m4 = ConditionalMaxent(5, basis2, (.3, .5), data)
m4.fit()
p4 = m4.dist(0)
print(m4.weights)

data2 = np.random.rand(100)
m5 = ConditionalMaxent(5, basis2, (.3, .5), data2)
m5.fit()
p5 = m5.dist(0)
print(m5.weights)
