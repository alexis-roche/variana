from variana.maxent import GaussianCompositeInference
from variana.maxent import LogisticRegression as LogisticRegression2
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pylab as pl

TEST_SIZE = 0.2

# import some data to play with
iris = datasets.load_iris()
data = iris.data  #[:, :2] 
target = iris.target

# sklearn implementation
print('sklearn')
lr = LogisticRegression(C=1e15, solver='lbfgs', multi_class='multinomial')
lr.fit(data, target)
dist = lr.predict_proba(data)

# variana implementation
print('variana')
lr2 = LogisticRegression2(data, target)
lr2.fit(verbose=True)
dist2 = lr2.dist()

# Composite likelihood
print('composite stuff')
lr3 = GaussianCompositeInference(data, target, supercomposite=False, homoscedastic=False)
#lr2.fit(verbose=True, method='bfgs')
lr3.fit(verbose=True)
dist3 = lr3.dist()


def zob(idx):
    pl.plot(dist[idx, :], 'b')
    pl.plot(dist2[idx, :], 'g')
    pl.plot(dist3[idx, :], 'orange')
    pl.show()


y, yt, x, xt = train_test_split(data, target, test_size=TEST_SIZE)

