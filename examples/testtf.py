import tensorflow as tf
import numpy as np

DTYPE = 'float32'
DIM = 10

x_ = tf.Variable(np.full(10, DIM), dtype=DTYPE)
y_ = tf.Variable(np.zeros(DIM), dtype=DTYPE)

n_ = tf.random_normal((DIM,), dtype=DTYPE)
z_ = x_ + n_

a_ = tf.assign(x_, z_)
b_ = tf.assign(y_, .1 * z_)
c_ = tf.assign(x_, y_)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

a, b = sess.run([a_, b_])
c = sess.run(c_)

a1, b1, c1 = sess.run([a_, b_, c_])
