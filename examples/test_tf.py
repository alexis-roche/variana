import os
import tensorflow as tf
import numpy as np

from reid_net_train import *


net = ReidNetTrainCUHK03('/home/alexis/zob', FLAGS.data_dir, optimizer='sep')
precision = net._sess.run(net._optimizer._precision)

dims = net.dims

###dims = [np.prod(p.shape.as_list()) for p in net._params]


"""
m, s = net.initializer_mean_and_stddev()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mv = sess.run(m)
    sv = sess.run(s)
"""

"""
r = tf.random_uniform([2, 3], minval=-5.6, maxval=7.5, dtype=tf.float32, seed=None, name=None)

xmin = r.op.inputs[1]
aux = r.op.inputs[0].op.inputs[1]


with tf.Session() as sess:
    rv = sess.run(r)
    xminv = sess.run(xmin)
    auxv = sess.run(aux)
    
xmaxv = auxv + xminv
"""

"""
v = tf.Variable(r)
init = v.initializer
inputs = init.inputs
tens = inputs[1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = sess.run(v)

"""
###y = tf.layers.conv2d(x, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), )

