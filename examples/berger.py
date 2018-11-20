from variana.maxent import *

def f1(x):
    return x < 2

def f2(x):
    return x in (0, 2)


m = MaxentModel(5, (f1, f2), (0.3, 0.5))
m.fit()
p = m.dist()

m2 = MaxentModelGKL(5, (f1, f2), (0.3, 0.5))
m2.fit()
p2 = m2.dist()

F1 = lambda x, y: f1(x)
F2 = lambda x, y: f2(x)
Y =  (0, 1, 0, 1, 0, 1, 0, 1)
m3 = ConditionalMaxentModel(5, (F1, F2), (.3, .5), Y)
m3.fit()
p3 = m3.dist(0)

FF1 = lambda x, y: f1(x) * abs(1 - 2*y) 
FF2 = lambda x, y: f2(x) * abs(1 - 2*y)

m4 = ConditionalMaxentModel(5, (FF1, FF2), (.3, .5), Y)
m4.fit()
p4 = m4.dist(0)
