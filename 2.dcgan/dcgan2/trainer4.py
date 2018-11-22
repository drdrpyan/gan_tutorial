import tile_image_maker
import tensorflow as tf
import numpy as np
from six.moves import xrange
import imageio
import time
import os

class DCGANTrainer4(object):
    def __init__(self, sess, 
                 data_feeder, noise_feeder, 
                 batch_size, epoch, opt_param, log_path,
                 sample_path):
        self.sess = sess
        self.data_feeder = data_feeder
        self.noise_feeder = noise_feeder
        self.batch_size = batch_size
        self.epoch = epoch

        self.sample_path = sample_path
        self.tile_image_maker = tile_image_maker.TileImageMaker()

        self._init_input(data_feeder.shape, noise_feeder.shape)
        #self._build_graph(self._mnist_discriminator, self._mnist_generator)
        self._build_graph(self._mnist_discriminator2, self._mnist_generator2)
        self._init_loss()
        self._init_trainable_vars()
        self._init_optimizer(opt_param)
        self._init_tb(log_path)

    def _init_input(self, data_shape, noise_shape):
        self.data_real_input = tf.placeholder(tf.float32, [self.batch_size] + data_shape)
        self.noise_input = tf.placeholder(tf.float32, [self.batch_size] + noise_shape)
        self.flag_disc_train = tf.placeholder(tf.bool)
        self.flag_gen_train = tf.placeholder(tf.bool)

    def _build_graph(self, discriminator, generator):        
        self.gen_raw_out, self.gen_sig_out = generator('gen', self.noise_input, self.flag_gen_train)

        self.disc_real_raw_out, self.disc_real_sig_out = \
            discriminator('disc', self.data_real_input, self.flag_disc_train)
        self.disc_fake_raw_out, self.disc_fake_sig_out = \
            discriminator('disc', self.gen_sig_out, self.flag_disc_train, True)

    def _mnist_discriminator(self, name, input, train_flag, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            h0_0 = tf.layers.conv2d(input, 64, (5, 5), (2, 2), 'same')
            h0_1 = tf.nn.leaky_relu(h0_0)

            h1_0 = tf.layers.conv2d(h0_1, 128, (5, 5), (2, 2), 'same')
            h1_1 = tf.nn.leaky_relu(h1_0)

            kernel_size = (h1_1.shape[1], h1_1.shape[2])
            h2_0 = tf.layers.conv2d(h1_1, 1024, kernel_size, (1, 1), 'valid')
            h2_1 = tf.nn.leaky_relu(h2_0)

            h3 = tf.layers.conv2d(h2_1, 1, (1, 1), (1, 1), 'valid')

            raw_o = h3
            sig_o = tf.nn.sigmoid(h3)

        return raw_o, sig_o

    def _mnist_discriminator2(self, name, input, train_flag, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            h0_0 = tf.layers.conv2d(input, 64, (5, 5), (2, 2), 'same')
            h0_1 = tf.nn.leaky_relu(h0_0)

            h1_0 = tf.layers.conv2d(h0_1, 128, (5, 5), (2, 2), 'same')
            h1_1 = tf.contrib.layers.batch_norm(h1_0, is_training=train_flag, updates_collections=None)
            h1_2 = tf.nn.leaky_relu(h1_1)

            kernel_size = (h1_2.shape[1], h1_2.shape[2])
            h2_0 = tf.layers.conv2d(h1_2, 1024, kernel_size, (1, 1), 'valid')
            h2_1 = tf.contrib.layers.batch_norm(h2_0, is_training=train_flag, updates_collections=None)
            h2_2 = tf.nn.leaky_relu(h2_1)

            h3 = tf.layers.conv2d(h2_2, 1, (1, 1), (1, 1), 'valid')

            raw_o = h3
            sig_o = tf.nn.sigmoid(h3)

        return raw_o, sig_o

    def _mnist_generator(self, name, input, train_flag):
        with tf.variable_scope(name) as scope:
            h0_0 = tf.layers.dense(input, 1024)
            h0_1 = tf.nn.relu(h0_0)

            h1_0 = tf.layers.dense(h0_1, 7*7*128)
            h1_1 = tf.nn.relu(h1_0)

            h2_0 = tf.reshape(h1_1, [-1, 7, 7, 128])
            h2_1 = tf.layers.conv2d_transpose(h2_0, 128, (5, 5), (2, 2), 'same')
            h2_2 = tf.nn.relu(h2_1)

            h3 = tf.layers.conv2d_transpose(h2_2, 1, (5, 5), (2, 2), 'same')

            raw_o = h3
            sig_o = tf.nn.sigmoid(h3)

        return raw_o, sig_o

    def _mnist_generator2(self, name, input, train_flag):
        with tf.variable_scope(name) as scope:
            h0_0 = tf.layers.dense(input, 1024)
            h0_1 = tf.contrib.layers.batch_norm(h0_0, is_training=train_flag, updates_collections=None)
            h0_2 = tf.nn.relu(h0_1)

            h1_0 = tf.layers.dense(h0_2, 7*7*128)
            h1_1 = tf.contrib.layers.batch_norm(h1_0, is_training=train_flag, updates_collections=None)
            h1_2 = tf.nn.relu(h1_1)

            h2_0 = tf.reshape(h1_2, [-1, 7, 7, 128])
            h2_1 = tf.layers.conv2d_transpose(h2_0, 128, (5, 5), (2, 2), 'same')
            h2_2 = tf.contrib.layers.batch_norm(h2_1, is_training=train_flag, updates_collections=None)
            h2_3 = tf.nn.relu(h2_2)

            h3 = tf.layers.conv2d_transpose(h2_3, 1, (5, 5), (2, 2), 'same')

            raw_o = h3
            sig_o = tf.nn.sigmoid(h3)

        return raw_o, sig_o

    def _init_loss(self):
        #self.loss_disc_real = tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=self.disc_real_raw_out, labels=tf.ones_like(self.disc_real_raw_out))
        #self.loss_disc_fake = tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=self.disc_fake_raw_out, labels=tf.zeros_like(self.disc_fake_raw_out))
        #self.loss_disc = tf.reduce_mean(self.loss_disc_real + self.loss_disc_fake)
        self.loss_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_real_raw_out, labels=tf.ones_like(self.disc_real_raw_out)))
        self.loss_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_fake_raw_out, labels=tf.zeros_like(self.disc_fake_raw_out)))
        self.loss_disc = self.loss_disc_real + self.loss_disc_fake

        self.loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_fake_raw_out, labels=tf.ones_like(self.disc_fake_raw_out)))

    def _init_trainable_vars(self):
        self.vars_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc')
        self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')

    def _init_optimizer(self, opt_params):
        self.opt_disc = tf.train.AdamOptimizer(opt_params['learning_rate'], opt_params['beta1']).minimize(self.loss_disc, var_list=self.vars_disc)
        self.opt_gen = tf.train.AdamOptimizer(opt_params['learning_rate'], opt_params['beta1']).minimize(self.loss_gen, var_list=self.vars_gen)
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #    self.opt_disc = tf.train.AdamOptimizer(opt_params['learning_rate'], opt_params['beta1']).minimize(self.loss_disc, var_list=self.vars_disc)
        #    self.opt_gen = tf.train.AdamOptimizer(opt_params['learning_rate'], opt_params['beta1']).minimize(self.loss_gen, var_list=self.vars_gen)

    def _init_tb(self, log_path):
        data_real = tf.summary.image('data_real_input', self.data_real_input)
        loss_disc = tf.summary.scalar('loss_disc', self.loss_disc)
        self.summary_disc = tf.summary.merge([data_real, loss_disc])

        data_fake = tf.summary.image('data_fake', self.gen_sig_out)
        loss_gen = tf.summary.scalar('loss_gen', self.loss_gen)
        self.summary_gen = tf.summary.merge([data_fake, loss_gen])

        #pre_data_real = tf.summary.image('pre_data_real', self.data_real_input)
        #pre_data_fake = tf.summary.image('pre_data_fake', self.data_fake_input)
        #pre_loss_disc = tf.summary.scalar('loss_pre_disc', self.loss_disc)
        #self.summary_pre_disc = tf.summary.merge([pre_data_real, pre_data_fake, pre_loss_disc])

        self.summary_writer = tf.summary.FileWriter(log_path, self.sess.graph)

    def _register_sample(self, data_real, noise):
        self.sample_data_real = data_real
        self.sample_noise = noise

    def _write_sample(self, epoch):
        gen_data, l_d, l_g = self.sess.run(
            [self.gen_sig_out, self.loss_disc, self.loss_gen], 
            feed_dict={self.data_real_input: self.sample_data_real,
                       self.noise_input: self.sample_noise,
                       self.flag_disc_train: False, self.flag_gen_train: False})
        tile_image = self.tile_image_maker.create(gen_data)
        image_path = os.path.normpath(os.path.join(self.sample_path, 'e%04d_ld%.8f_lg%.8f.png' % (epoch+1, l_d, l_g)))
        imageio.imwrite(image_path, tile_image)
        
    def train(self):
        tf.global_variables_initializer().run()

        iter_per_epoch = int(self.data_feeder.num_data // self.batch_size)

        start_time = time.time()

        for e in xrange(self.epoch):
            for i in xrange(0, iter_per_epoch):
                data_real = np.array(self.data_feeder.get(self.batch_size))
                noise = self.noise_feeder.get(self.batch_size)

                iter = e * iter_per_epoch + i
                if iter == 0:
                    self._register_sample(data_real, noise)
                    self._write_sample(-1)

                #_, loss_disc, summary_disc = self.sess.run(
                #    [self.opt_disc, self.loss_disc, self.summary_disc],
                #    feed_dict={self.data_real_input: data_real,
                #               self.noise_input: noise})

                #_, loss_gen, summary_gen = self.sess.run(
                #    [self.opt_gen, self.loss_gen, self.summary_gen],
                #    feed_dict={self.noise_input: noise})

                #_, loss_gen, summary_gen = self.sess.run(
                #    [self.opt_gen, self.loss_gen, self.summary_gen],
                #    feed_dict={self.noise_input: noise})
                _, loss_disc, summary_disc = self.sess.run(
                    [self.opt_disc, self.loss_disc, self.summary_disc],
                    feed_dict={self.data_real_input: data_real,
                               self.noise_input: noise,
                               self.flag_disc_train: True, self.flag_gen_train: True})

                _, loss_gen, summary_gen = self.sess.run(
                    [self.opt_gen, self.loss_gen, self.summary_gen],
                    feed_dict={self.noise_input: noise,
                               self.flag_disc_train: True, self.flag_gen_train: True})

                _, loss_gen, summary_gen = self.sess.run(
                    [self.opt_gen, self.loss_gen, self.summary_gen],
                    feed_dict={self.noise_input: noise,
                               self.flag_disc_train: True, self.flag_gen_train: True})

                self.summary_writer.add_summary(summary_disc, iter)
                self.summary_writer.add_summary(summary_gen, iter)

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss_d: %.8f, loss_g: %.8f" \
                    % (e+1, self.epoch, i+1, iter_per_epoch, time.time() - start_time, loss_disc, loss_gen))

            
            self._write_sample(e)
            
            
                