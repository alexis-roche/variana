import tensorflow as tf
import numpy as np
import pylab as pl

from variana.utils import rms


FLOAT_TYPE = 'float32'


class OnlineIProj(object):

    def __init__(self, x, loss, gamma, vmax=1e5, vmin=1e-10):
        self._pmin = 1 / float(vmax)
        self._pmax = 1 / float(vmin)
        self._x = x
        self._dim = int(x.shape[0])
        self._loss = loss
        self._log_p = -loss
        self._logK = tf.Variable(0.0, trainable=False, dtype=FLOAT_TYPE)
        self._xs = tf.Variable(np.zeros(x.shape), name='sample', trainable=False, dtype=FLOAT_TYPE)
        self._m = tf.Variable(np.zeros(x.shape), name='mean', trainable=False, dtype=FLOAT_TYPE)
        self._p = tf.Variable(np.ones(x.shape), name='precision', trainable=False, dtype=FLOAT_TYPE)
        self._it = tf.Variable(0, name='global_step', trainable=False)
        self._gamma = tf.Variable(gamma, trainable=False, dtype=FLOAT_TYPE)
        self._updates = self._get_updates()
        self._init_session()
        
    def _init_session(self):
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def close(self):
        self._sess.close()        

    def update(self):
        self._sess.run(self._updates)
        
    def loss(self, x):
        self._sess.run(tf.assign(self._x, x))
        return self._sess.run(self._loss)
        
    def _get_updates(self):

        diff = self._xs - self._m
        dev2 = self._p * diff ** 2
        log_q = self._logK - .5 * tf.reduce_sum(dev2)
        delta = self._gamma * (self._log_p - log_q)
        
        new_p = tf.minimum(tf.maximum(self._p * (1 - (dev2 - 1) * delta), self._pmin), self._pmax)
        dm = delta * (self._p / new_p) * diff
        new_m = self._m + dm
        delta2 = .5 * (tf.reduce_sum(new_p * (1 / self._p + dm ** 2)) - self._dim)
        new_logK = self._logK + delta2 + delta

        ###x = tf.truncated_normal(self._m.shape, self._m, (1 / self._p) ** .5, dtype=FLOAT_TYPE)
        x = tf.random_normal(self._m.shape, self._m, (1 / self._p) ** .5, dtype=FLOAT_TYPE)

        out = []
        out.append(tf.assign(self._xs, x))
        out.append(tf.assign(self._x, x))
        out.append(tf.assign(self._p, new_p))
        out.append(tf.assign(self._m, new_m))
        out.append(tf.assign(self._logK, new_logK))
        out.append(tf.assign_add(self._it, 1))

        return out

    @property
    def m(self):
        return self._sess.run(self._m)

    @property
    def v(self):
        return self._sess.run(1 / self._p)

    @property
    def logK(self):
        return self._sess.run(self._logK)

    @property
    def xs(self):
        return self._sess.run(self._xs)
    
    @property
    def it(self):
        return self._sess.run(self._it)

    @property
    def _zob(self):
        return self._sess.run(self._x)




dim = 1
vmax = 1e2
K = np.random.rand()
m = 5 * (np.random.rand(dim) - .5)
v = 5 * (np.random.rand(dim) + 1)
#K, m, v = 1, 0, 1

x_ = tf.Variable(np.zeros(dim), dtype=FLOAT_TYPE)
loss_ = -np.log(K) + .5 * tf.reduce_sum((x_ - m) ** 2 / v)
k_gamma = .01
gamma = k_gamma / np.sqrt(dim)
niter = int(10 / gamma)

q = OnlineIProj(x_, loss_, gamma, vmax=vmax)

err = []
for step in range(niter):
    print(step)
    q.update()
    err.append((rms(q.logK - np.log(K)), rms(q.m - m), rms(q.v - v)))

pl.figure()
pl.plot(err)
pl.show()

