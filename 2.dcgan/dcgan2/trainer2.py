import tensorflow as tf
from six.moves import xrange
import numpy as np
import time


class DCGANTrainer2(self):
    def __init__(self, sess, dcgan, data_feeder, noise_feeder, log_path='./logs/'):
        self.sess = sess
        self.dcgan = dcgan
        
        self.data_feeder = data_feeder
        self.noise_feeder = noise_feeder

        self._init_tb_summary(log_path)

    def _init_opt(self, learning_rate, beta1):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_gen, var_list=self.dcgan.vars_gen)
            self.d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_disc, var_list=self.dcgan.vars_disc)

    def _init_tb_summary(self, log_path):
        data_real = tf.summary.image('data_real', self.dcgan.disc_real.inputs[0])
        loss_disc = tf.summary.scalar('loss_disc', self.dcgan.loss_disc)
        self.summary_disc = tf.summary.merge([data_real, loss_disc])

        noise = tf.summary.histogram('noise', self.dcgan.gen.inputs[0])
        gen = tf.summary.image('gen', self.dcgan.gen.outputs[0])
        loss_gen = tf.summary.scalar('loss_gen', self.dcgan.loss_gen)
        self.summary_gen = tf.summary.merge([noise, gen, loss_gen])

        self.summary_writer = tf.summary.FileWriter(log_path, self.sess.graph)

    def train(self, batch_size, epoch, learning_rate, beta1):
        self._init_opt(learning_rate, beta1)

        tf.global_variables_initializer().run()

        start_time = time.time()
        iter = 1
        batch_per_epoch = int(self.data_feeder.num_data // batch_size)

        for e in xrange(epoch):
            for b in xrange(0, batch_per_epoch):
                x = self.data_feeder.get(batch_size)
                z = self.noise_feeder.get(batch_size)

    def pretrain_disc(self, batch_size):
        num_fetch = int(batch_size/2)
        num_iter = int(self.data_feeder.num_data // num_fetch)
        
        label_real = np.ones((num_fetch, 1, 1, 1))
        label_fake = np.zeros((num_fetch, 1, 1, 1))
        label = np.concatenate(label_real, label_fake)

        for i in xrange(0, num_iter):
            x_real = self.data_feeder.get(num_fetch)
            x_fake = np.random.uniform(size=x_real.shape)
            x = np.concatenate(x_real, x_fake)

            self.sess.run([self.d_opt, 

            
        