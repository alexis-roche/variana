"""
python reid_net_train.py --checkpoint_period=50 --email_period=1000 --validation_runs=100 --email_password=xxx --optimizer=sep
"""

import os
import smtplib
from email.mime.text import MIMEText
import random

import tensorflow as tf
import numpy as np
import cv2

from reid_net import ReidNet, format_image, IMAGE_WIDTH, IMAGE_HEIGHT

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 150, 'batch size for training')
tf.flags.DEFINE_integer('max_steps', 210000, 'max steps for training')
tf.flags.DEFINE_string('optimizer', 'sep', 'Optimizer (momentum or sep)')

tf.flags.DEFINE_float('momentum', 0.9, 'momentum')
tf.flags.DEFINE_float('init_learning_rate', 1e-1, 'initial learning rate (aka initial step size wrt gradient)')
tf.flags.DEFINE_float('init_decay', 1e-4, 'init_decay')
tf.flags.DEFINE_float('power_decay', .75, 'power_decay')
tf.flags.DEFINE_integer('cooling_period', 10000, 'cooling_period')

tf.flags.DEFINE_integer('bootstraps', 0, 'bootstraps')

tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')
tf.flags.DEFINE_string('logs_dir', '/home/alexis/zob', 'path to logs directory')
tf.flags.DEFINE_string('npz_train', 'rec.npz', 'numpy training loss record file')
tf.flags.DEFINE_string('npz_calibrate', 'calibrate.npz', 'numpy training loss record file')
tf.flags.DEFINE_string('data_dir', '/home/alexis/data/cuhk03_release', 'path to CUHK03 data directory')
tf.flags.DEFINE_integer('checkpoint_period', 50, 'number of steps inbetween chekpoints')
tf.flags.DEFINE_integer('validation_runs', 1, 'Number of validation runs')
tf.flags.DEFINE_integer('email_period', 1000, 'number of steps inbetween email warnings')
tf.flags.DEFINE_string('email_password', '*****', 'email password')


#########################################################################################################
# I/O functions for CUHK03 dataset
#########################################################################################################

def get_pair(path, task, num_id, positive):
    pair = []
    if positive:
        value = int(random.random() * num_id)
        id = [value, value]
    else:
        while True:
            id = [int(random.random() * num_id), int(random.random() * num_id)]
            if id[0] != id[1]:
                break
    for i in range(2):
        filepath = ''
        while True:
            index = int(random.random() * 10)
            filepath = '%s/labeled/%s/%04d_%02d.jpg' % (path, task, id[i], index)
            if not os.path.exists(filepath):
                continue
            break
        pair.append(filepath)
    return pair


def get_num_id(path, task):
    files = os.listdir('%s/labeled/%s' % (path, task))
    files.sort()
    return int(files[-1].split('_')[0]) - int(files[0].split('_')[0]) + 1


def read_data(path, task, image_width, image_height, batch_size):
    """
    Labels: 
      same --> 0
      different --> 1
    """
    num_id = get_num_id(path, task)    
    batch_images = []
    labels = []
    for i in range(batch_size // 2):
        pairs = [get_pair(path, task, num_id, True), get_pair(path, task, num_id, False)]
        for pair in pairs:
            images = []
            for p in pair:
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            batch_images.append(images)
        labels.append(0)
        labels.append(1)
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)

    
#########################################################################################################
# ReID network training object
#########################################################################################################

def learning_rate(step, init_learning_rate, init_decay, power_decay, cooling_period):
    """Returns a learning rate between 0 and 1 (initial value is 1):
    
    lr = 1 / (1 + init_decay * 10^(global_step // cooling_period) * global_step))^power_decay
    
    power_decay should be in the range ]0.5, 1] in order to meet the
    Robbins-Monro conditions:
        
    (i) sum_n^infty=infty a_n = infty
    (ii) sum_n^infty an^2 < infty
        
    To see this, apply the Riemann series theorem twice:
    (i) holds iff power_decay <= 1
    (ii) holds iff 2 * power_decay > 1

    """
    if cooling_period is None:
        decay = init_decay
    else:
        decay = init_decay * 10 ** tf.to_float(step // cooling_period)
    return init_learning_rate * (1 + decay * tf.to_float(step)) ** -power_decay


class OnlineLaplaceFit(object):

    def __init__(self, params, loss, vmax=1e3, max_step_size=0.1):
        self._params = params
        self._log_target = -loss
        self._precision = [tf.Variable(np.ones(p.shape), name='precision', trainable=False, dtype=p.dtype.base_dtype.name) for p in self._params]
        self._grad = tf.gradients(self._log_target, self._params)
        self._hess = tf.gradients(self._grad, self._params)
        self._vmax = float(vmax)
        self._max_step_size = float(max_step_size)

    def get_updates(self, rho, global_step=None):
        out = []
        for x, p, g, h in zip(self._params, self._precision, self._grad, self._hess):
            new_p = (1 - rho) * p - rho * h
            new_x = x + rho * g / new_p
            new_x = x + rho * g / tf.maximum(new_p, rho / self._max_step_size + 1 / self._vmax)
            if not x.constraint is None:
                new_x = x.constraint(new_x)
            out.append(tf.assign(p, new_p))
            out.append(tf.assign(x, new_x))
        if not global_step is None:
            out.append(tf.assign_add(global_step, 1))     
        return out

    def get_loss_updates(self):
        return [self._loss]

    

class OnlineIProj(object):

    def __init__(self, params, loss, init_var, gamma=1, vmax=1e3, vmin=1e-10):
        """
        A view on the network params is created, which is reset at
        sampling time, so not reliable for loss estimation
        """
        self._gamma = float(gamma)
        self._pmin = 1 / float(vmax)
        self._pmax = 1 / float(vmin)
        self._params = params
        self._dims = [np.prod(x.shape.as_list()) for x in self._params]
        self._loss = loss
        ###self._log_p = -loss
        self._log_p = -.5 * tf.reduce_sum([tf.reduce_sum(x ** 2) for x in self._params])
        self._logK = tf.Variable(0.0, trainable=False)
        self._mean = [tf.Variable(np.zeros(p.shape), name='mean', trainable=False, dtype=p.dtype.base_dtype.name) for p in self._params]
        ###self._precision = [tf.Variable(1 / tf.maximum(tf.minimum(v, self._vmax), self._vmin), name='precision', trainable=False) for v in init_var]
        self._precision = [tf.Variable(np.ones(p.shape), name='precision', trainable=False, dtype=p.dtype.base_dtype.name) for p in self._params]
        self._sample = [tf.Variable(np.zeros(p.shape), name='sample', trainable=False, dtype=p.dtype.base_dtype.name) for p in self._params]

    def get_updates(self, rho, global_step=None):
        out = []

        diff, dev2 = [], []
        for xs, x, m, p in zip(self._sample, self._params, self._mean, self._precision):
            new_x = tf.truncated_normal(m.shape, m, (1 / p) ** .5)
            if not x.constraint is None:
                new_x = x.constraint(new_x)
            out.append(tf.assign(x, new_x))
            out.append(tf.assign(xs, new_x))
            aux = xs - m
            diff.append(aux)
            dev2.append(p * aux ** 2)
        log_q = self._logK - .5 * tf.reduce_sum([tf.reduce_sum(d2) for d2 in dev2])
        delta = self._gamma * rho * (self._log_p - log_q)

        ### DEBUG
        self._log_q = log_q
        self._delta = self._log_p - log_q
        
        delta2 = []
        for xs, x, m, p, d1, d2, dim in zip(self._sample, self._params, self._mean, self._precision, diff, dev2, self._dims):
            new_p = tf.minimum(tf.maximum(p * (1 - (d2 - 1) * delta), self._pmin), self._pmax)
            dm = d1 * (p / new_p) * delta
            new_m = m + dm
            delta2.append(tf.reduce_sum(new_p * (1 / p + dm ** 2)) - dim)
            out.append(tf.assign(p, new_p))
            out.append(tf.assign(m, new_m))
        new_logK = self._logK + .5 * tf.reduce_sum(delta2) + delta
        out.append(tf.assign(self._logK, new_logK))

        if not global_step is None:
            out.append(tf.assign_add(global_step, 1))
        
        return out

    def get_loss_updates(self):
        return [x.assign(m) for x, m in zip(self._params, self._mean)] + [self._loss]
    
    
    
class ValidationInfo(object):

    def __init__(self, acc, spe, sen, cce, conf_level=.9, bootstraps=1000, batch_size=100):
        mystats = lambda x: stats(x, conf_level=conf_level, bootstraps=bootstraps, batch_size=batch_size)
        self._acc = mystats(acc)
        self._spe = mystats(spe)
        self._sen = mystats(sen)
        self._cce = mystats(cce)

    @property
    def accuracy(self):
        return self._acc[0]

    @property
    def specificity(self):
        return self._spe[0]

    @property
    def sensitivity(self):
        return self._sen[0]

    @property
    def cross_entropy(self):
        return self._cce[0]

    def as_dict(self):
        return {'acc': self._acc,
                'spe': self._spe,
                'sen': self._sen,
                'cce': self._cce}        
        
    def as_string(self, digits=3):
        if digits is None:
            prec = '%f'
        else:
            prec = '%%.%df' % digits
        def disp(name, val):
            return '%s: %s (%s, %s)' % (name, prec, prec, prec) % val
        out = disp('Accuracy', self._acc) + '\n'
        out += disp('Specificity', self._spe) + '\n'
        out += disp('Sensitivity', self._sen) + '\n'
        out += disp('Cross-entropy', self._cce) + '\n'
        return out

    
class ReidNetTrainCUHK03(ReidNet):

    def __init__(self, path, data_path,
                 batch_size=150,
                 optimizer='momentum',
                 init_learning_rate=1e-1,
                 init_decay=1e-4,
                 power_decay=.75,
                 cooling_period=None,
                 momentum=0.9):
        self._init_net(batch_size)
        self._init_training(data_path, optimizer, init_learning_rate, init_decay, power_decay, cooling_period, momentum)
        self._init_session(path)

    def _init_training(self, data_path, optimizer, init_learning_rate, init_decay, power_decay, cooling_period, momentum):
        self._data_path = data_path
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        self._learning_rate = learning_rate(self._global_step, init_learning_rate, init_decay, power_decay, cooling_period)
        self._params = tf.trainable_variables()
        if optimizer == 'momentum':
            self._optimizer = tf.train.MomentumOptimizer((1 - momentum) * self._learning_rate, momentum)
            self._updates = self._optimizer.minimize(self._loss, global_step=self._global_step)
        elif optimizer == 'sep':
            ###self._optimizer = OnlineLaplaceFit(self._params, self._loss)
            ###self._updates = self._optimizer.get_updates(self._learning_rate, global_step=self._global_step)
            self._optimizer = OnlineIProj(self._params, self._loss, self.initializer_variance(), gamma=.01 / np.sqrt(self.dim))
            self._updates = self._optimizer.get_updates(self._learning_rate, global_step=self._global_step)
            self._loss_updates = self._optimizer.get_loss_updates()
        else:
            raise ValueError('unknown optimizer')
        
    @property
    def params(self):
        return self._sess.run(self._params)

    @property
    def global_step(self):
        return self._sess.run(self._global_step)
        
    def make_batch(self, task='train'):
        """The batch is built so as to interleave positive and negative image
        pairs in a strict manner. Data is read from the labeled/train
        sub-directory of CUHK03. At each of batch_size / 2 times, we
        get two random image pairs: one positive (same person), one
        negative (different persons). The positive pair is found by
        simply drawing a random subject ID, and then two image
        IDs. The negative pair is found by drawing two distinct
        subject IDs, and then two image IDs.
        """
        feed_images, feed_labels = read_data(self._data_path, task, IMAGE_WIDTH, IMAGE_HEIGHT, self._batch_size)
        return {self._images: feed_images,
                self._labels: feed_labels,
                self._is_train: task == 'train'}
    
    def _update(self):
        feed_dict = self.make_batch()
        return feed_dict, self._sess.run(self._updates, feed_dict=feed_dict)

    def update(self):
        feed_dict, _ = self._update()

        ### DEBUG
        print('logK = %f' % self._sess.run(self._optimizer._logK, feed_dict=feed_dict))
        print('log_p = %f' % self._sess.run(self._optimizer._log_p, feed_dict=feed_dict))
        print('log_q = %f' % self._sess.run(self._optimizer._log_q, feed_dict=feed_dict))
        print('delta = %f' % self._sess.run(self._optimizer._delta, feed_dict=feed_dict))
        """
        print('sample = %s' % np.array([np.max(xs) for xs in self._sess.run(self._optimizer._sample, feed_dict=feed_dict)]))
        print('mean = %s' % np.array([np.max(m) for m in self._sess.run(self._optimizer._mean, feed_dict=feed_dict)]))
        print('precision = %s' % np.array([np.max(p) for p in self._sess.run(self._optimizer._precision, feed_dict=feed_dict)]))
        print('params = %s' % np.array([np.max(p) for p in self._sess.run(self._params, feed_dict=feed_dict)]))
        """

        return self.loss(feed_dict)

        
    def loss(self, feed_dict):
        out = self._sess.run(self._loss_updates, feed_dict=feed_dict)
        return out[-1]
    
    def validate(self, runs, dataset='val', tol=1e-10, bootstraps=0, verbose=True):
        safelog = lambda x: np.log(np.maximum(x, tol))
        count_correct, count_ll = 0, 0
        acc, spe, sen, cce = [], [], [], []
        for _ in range(runs):
            feed_dict = self.make_batch(dataset)
            labels = feed_dict[self._labels]
            proba = self._sess.run(self._proba, feed_dict=feed_dict)
            classif = np.argmax(proba, 1)
            acc.append(classif == labels)
            spe.append(classif[labels == 0] == 0)
            sen.append(classif[labels == 1] == 1)
            cce.append(-safelog(proba[range(len(labels)), labels]))
        out = ValidationInfo(acc, sen, spe, cce, bootstraps=bootstraps)
        if verbose:
            print(out.as_string())
        return out

    def calibrate(self, runs, npz_file, dataset='val'):
        p0, p1 = [], []
        for _ in range(runs):
            feed_dict = self.make_batch(dataset)
            labels = feed_dict[self._labels]
            proba = self._sess.run(self._proba, feed_dict=feed_dict)[..., 0]
            p0.append(proba[labels==0])
            p1.append(proba[labels==1])
        np.savez(npz_file, p0=p0, p1=p1)


#########################################################################################################
# Utilities
#########################################################################################################
def ci_student(x, conf_level):
    from scipy.stats import t
    m = np.mean(x)
    df = np.size(x) - 1
    s = np.std(x) / np.sqrt(df)
    f = t.ppf((1 + conf_level) / 2, df=df)
    return m, m - f * s, m + f * s


def ci_bootstraps(x, conf_level, bootstraps, batch_size):
    if batch_size is None:
        batch_size = max(1, 10000 // bootstraps)
    m = np.mean(x)
    done = False
    xb = np.array(())
    left = bootstraps
    while left > 0:
        print('Left bootstraps: %d' % left)
        n = min(batch_size, left)
        xb = np.concatenate((xb, np.mean(np.random.choice(x, size=(n, x.size)), 1)))
        left = left - n
    m0 = np.percentile(xb, 50 * (1 - conf_level))
    m1 = np.percentile(xb, 50 * (1 + conf_level))
    return m, m0, m1
    

def stats(x, conf_level=.9, bootstraps=1000, batch_size=None):
    x = np.asarray(x).flatten()
    if bootstraps < 1:
        return ci_student(x, conf_level)
    else:
        return ci_bootstraps(x, conf_level, bootstraps, batch_size)
    

def send_email(subject, text, passwd):
    print('Sending email...')
    addr = 'alexis.roche@gmail.com'
    msg = MIMEText(text)
    msg['Subject'] = subject
    msg['From'] = addr
    msg['To'] = addr
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.starttls()
        server.login('alexis.roche', passwd)
        problems = server.sendmail(addr, [addr], msg.as_string())
        server.quit()
    except:
        print('WARNING: Could not send email')


#########################################################################################################
# Main
#########################################################################################################

def main(argv=None):
    
    if FLAGS.mode == 'test':
        net = ReidNet(FLAGS.logs_dir, 1)
    else:
        net = ReidNetTrainCUHK03(FLAGS.logs_dir,
                                 FLAGS.data_dir,
                                 batch_size=FLAGS.batch_size,
                                 optimizer=FLAGS.optimizer,
                                 init_learning_rate=FLAGS.init_learning_rate,
                                 init_decay=FLAGS.init_decay,
                                 power_decay=FLAGS.power_decay,
                                 cooling_period=FLAGS.cooling_period,
                                 momentum=FLAGS.momentum)
   

    if FLAGS.mode == 'train':

        print('Global step: %d' % net.global_step)
        npz_train = os.path.join(FLAGS.logs_dir, FLAGS.npz_train)
        if net.global_step == 0:
            rec_train = np.zeros(0)
            rec_test = np.zeros((0, 3))
        else:
            npz = np.load(npz_train)
            rec_train = npz['rec_train'][:net.global_step]
            rec_test = npz['rec_test'][npz['rec_test'][:,0] <= net.global_step, :]
            
        for step in range(net.global_step + 1, FLAGS.max_steps + 1):
            train_loss = net.update()
            rec_train = np.append(rec_train, train_loss)
            print('Step: %d, Train loss: %f' % (step, train_loss))
            np.savez(npz_train, rec_train=rec_train, rec_test=rec_test)
            
            if step % FLAGS.checkpoint_period == 0:
                net.save(FLAGS.logs_dir, step)

            if step % FLAGS.email_period == 0:
                print('Validating model...')
                test = net.validate(FLAGS.validation_runs, bootstraps=FLAGS.bootstraps)
                rec_test = np.append(rec_test, np.expand_dims((step, test.accuracy, test.cross_entropy), axis=0), axis=0)
                np.savez(npz_train, rec_train=rec_train, rec_test=rec_test)

    elif FLAGS.mode == 'val':
        test = net.validate(FLAGS.validation_runs, bootstraps=FLAGS.bootstraps)

    elif FLAGS.mode == 'calibrate':
        net.calibrate(FLAGS.validation_runs, os.path.join(FLAGS.logs_dir, FLAGS.npz_calibrate))

    elif FLAGS.mode == 'check':
        test = net.validate(FLAGS.validation_runs, dataset='train', bootstraps=FLAGS.bootstraps)

    elif FLAGS.mode == 'test':
        image1 = net.format_image(cv2.imread(FLAGS.image1))
        image2 = net.format_image(cv2.imread(FLAGS.image2))
        proba = net.run(np.array([image1, image2])).squeeze()
        i = np.argmax(proba)
        print('%s with probability %f' % ('Same' if i==0 else 'Different', proba[i]))

    net.close()


#########################################################################################################


if __name__ == '__main__':
    tf.app.run()

