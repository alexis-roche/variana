"""
Example:

python reid_net.py tmp/0_15.jpg register 
"""
import sys
import os
from glob import glob

import tensorflow as tf
import numpy as np
import cv2


IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160
PATH = '/home/alexis/models/PersonReID'


def randomize_image(image):
    tmp = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    tmp = tf.image.random_flip_left_right(tmp)
    tmp = tf.image.random_brightness(tmp, max_delta=32. / 255.)
    tmp = tf.image.random_saturation(tmp, lower=0.5, upper=1.5)
    tmp = tf.image.random_hue(tmp, max_delta=0.2)
    tmp = tf.image.random_contrast(tmp, lower=0.5, upper=1.5)
    return tf.image.per_image_standardization(tmp)


def preprocess(images, is_train):
    npairs = images.get_shape()[1]
    split = lambda x, n:  [tf.reshape(s, s.get_shape()[1:]) for s in tf.split(x, n)]
    pair = split(images, 2)
    
    def train():
        return [[randomize_image(img)\
                 for img in split(tf.image.resize_images(p, (IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3)), npairs)]\
                for p in pair]
    def val():
        return [[tf.image.per_image_standardization(img)\
                 for img in split(tf.image.resize_images(p, (IMAGE_HEIGHT, IMAGE_WIDTH)), npairs)]\
                for p in pair]

    images1, images2 = tf.cond(is_train, train, val)
    shape = (npairs, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    return [tf.reshape(tf.concat(images1, 0), shape), tf.reshape(tf.concat(images2, 0), shape)]




def conv_layer(filters, kernel_size, pool_size, stride, name=None, **kwargs):
    #kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    return lambda inputs: \
        tf.layers.max_pooling2d(tf.layers.conv2d(inputs, filters, (kernel_size, kernel_size),
                                                 activation=tf.nn.relu,
                                                 kernel_initializer=kernel_initializer,
                                                 name=name,
                                                 **kwargs),
                                (pool_size, pool_size), (stride, stride))


def dense_layer(units, name=None, **kwargs):
    kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    return lambda inputs: tf.layers.dense(inputs, units,
                                          kernel_initializer=kernel_initializer,
                                          name=name,
                                          **kwargs)


def multilayer(conv_layers, dense_layers):
    def wrap(new, curr):
        return lambda x: new(curr(x))
    conv = lambda inputs: inputs
    for c in conv_layers:
        conv = wrap(c, conv)
    dense = lambda inputs: inputs
    for d in dense_layers:
        dense = wrap(d, dense)
    # Assume first input dimension is batch size
    return lambda inputs: dense(tf.reshape(conv(inputs), (inputs.get_shape()[0], -1)))


def network(images, is_train, offset):
    """
    dims:  (160, 60, 3), (79, 29, 6), (38, 13, 12), (18, 5, 24), (8, 1, 48), (8, 1, 24), (8, 1, 12), 48, 1
    params: 60, 120, 240, 480, 48, 24, 4656, 49
    total param count: 5677

    The network returns a 'dissimilarity' measure between the
    images. The larger the dissimilarity, the more likely the images
    represent different persons.
    """
    ###print(images.get_shape())
    images1, images2 = preprocess(images, is_train)

    with tf.variable_scope('network', reuse=tf.AUTO_REUSE):

        # Architecture
        conv1 = conv_layer(6, 3, 2, 2, name='conv1')
        conv2 = conv_layer(12, 3, 2, 2, name='conv2')
        conv3 = conv_layer(24, 3, 2, 2, name='conv3')
        conv4 = conv_layer(48, 3, 2, 2, name='conv4')
        conv5 = conv_layer(24, 1, 1, 1, name='conv5')
        conv6 = conv_layer(12, 1, 1, 1, name='conv6')
        dense1 = dense_layer(48, name='dense1')
        cnn = multilayer((conv1, conv2, conv3, conv4, conv5, conv6), (dense1,))
        dense2 = dense_layer(1, kernel_constraint=lambda x: tf.maximum(x, 0), use_bias=offset, name='dense2')

        # Actual computation
        # Note: reshape the output to have shape (batches, ) as there
        # is only one last unit
        feat1, feat2 = (cnn(images) for images in (images1, images2))
        return dense2(tf.abs(tf.subtract(feat1, feat2)))[:, 0]
    

#########################################################################################################

class ExponModel():

    def __init__(self, power=2, th=1e-10):
        self._power = power
        self._th = th

    def _remap(self, dist):
        return tf.maximum(dist ** self._power, self._th)

    def __call__(self, dist):
        return tf.exp(-self._remap(dist))
    
    def cross_entropy(self, labels, dist):
        """
        z = 1 - labels
        x = dist
        p = [exp(-x), 1 - exp(-x)]
        z * x + (z - 1) * log(1 - exp(-x))
        """
        return (1 - labels) * self._remap(dist) - labels * tf.log(1 - self(dist))



def get_initializer_mean_and_stddev(p):
    r = p.initial_value
    if 'truncated_normal' in r.name:
        return tf.fill(p.shape, r.op.inputs[1]), tf.fill(p.shape, r.op.inputs[0].op.inputs[1])
    elif 'zeros' in r.name:
        return tf.zeros(p.shape), tf.ones(p.shape)
    else:
        raise ValueError('unknown initializer')

    
class ReidNet():

    def __init__(self, path, batch_size):
        self._init_net(batch_size)
        self._init_session(path)
        
    def _init_net(self, batch_size, use_sigmoid=True):
        self._batch_size = int(batch_size)
        self._images = tf.placeholder(tf.float32, (2, self._batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='images')
        self._is_train = tf.placeholder(tf.bool, name='is_train')
        self._score = network(self._images, self._is_train, use_sigmoid)
        self._labels = tf.placeholder(tf.float32, self._batch_size, name='labels')
        
        # Note that the cross-entropy implementations built in
        # tensorflow are "fused with the softmax/sigmoid
        # implementations because their performance and numerical
        # stability are critical to efficient training"
        #
        # Source:
        # https://github.com/tensorflow/tensorflow/issues/2462        
        if use_sigmoid:
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._labels, logits=self._score))
            p1 = tf.nn.sigmoid(self._score)
            p0 = 1 - p1
        else:
            m = ExponModel()
            self._loss = tf.reduce_mean(m.cross_entropy(self._labels, self._score))
            p0 = m(self._score)
            p1 = 1 - p0
        self._proba = tf.stack((p0, p1), axis=1)

    def _init_session(self, path):
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            print('Restored checkpoint: %s' % ckpt.model_checkpoint_path)

    def save(self, path, step):
        self._saver.save(self._sess, os.path.join(path, 'model.ckpt'), step)
        files = glob(os.path.join(path, 'model.ckpt-*'))
        files = [f for f in files if int(f.split('-')[1].split('.')[0]) < step]
        for f in files:
            print('Removing: %s' % f)
            os.remove(f)

    def initializer_variance(self):
        aux = [get_initializer_mean_and_stddev(p) for p in self._params]
        return [a[1] ** 2 for a in aux]

    def __enter__(self):
        return self
            
    def __exit__(self, type, value, traceback):
        self._sess.close()

    def close(self):
        self._sess.close()

    def run(self, images):
        """
        Input must be a numpy array of size (2, n, 60, 160, 3).
        
        Returns a numpy array of shape (n, 2) representing probability
        masses. By convention, the first probability value assesses
        the 'same person' hypothesis (this will be enforced in the
        training phase).
        """
        if images.ndim == 4:
            images = np.expand_dims(images, axis=1)
        if images.shape[0] != 2 or images.shape[2:] != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            raise ValueError('Incorrect input data shape: %s' % format(images.shape))
        feed_dict={self._images: images.astype(np.float32),
                   self._is_train: False}
        return self._sess.run(self._proba, feed_dict=feed_dict)

    @property
    def dims(self):
        return [np.prod(p.shape.as_list()) for p in self._params]

    @property
    def dim(self):
        return np.sum(self.dims)


#########################################################################################################

def format_image(img):
    """
    Format image read by opencv via `img = cv2.imread(fimg)`
    """
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


class ReidNetMatch(ReidNet):

    def __init__(self, path, template_images):
        self._init_net(len(template_images))
        self._init_session(path)
        self._template_images = [format_image(img) for img in template_images]

    def match(self, test_image):
        test_image = format_image(test_image)
        images = np.array([[test_image, img] for img in self._template_images]).swapaxes(0, 1)
        return self.run(images)[:, 0]


def list_files(names):
    out = []
    for name in names:
        if not os.path.extsep in name:
            search = os.path.join(name, '*%s*' % os.extsep)
        else:
            search = name
        out += glob(search)
    return out


if __name__ == '__main__':

    test_file = sys.argv[1]   
    template_files = list_files(sys.argv[2:])

    with ReidNetMatch(PATH, [cv2.imread(f) for f in template_files]) as net:
        prob = net.match(cv2.imread(test_file))
    
    print('Test image: %s' % test_file)
    for i in range(len(prob)):
        print(' %s: match probability = %f' % (template_files[i], prob[i])) 
    if len(prob) > 1:
        i, j = np.argmax(prob[:]), np.argmin(prob[:])
        print('Best match: %s, proba=%f' % (template_files[i], prob[i]))
        print('Worst match: %s, proba=%f' % (template_files[j], prob[j]))
