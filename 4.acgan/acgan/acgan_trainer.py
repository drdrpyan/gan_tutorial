import tile_image_maker
import tensorflow as tf
import numpy as np
from six.moves import xrange
import imageio
import time
import os

class ACGANTrainer(object):
    def __init__(self, sess, data_feeder, noise_feeder, opt_param, 
                 batch_size, epoch, log_path, num_sample_set, sample_path):
        self.sess = sess
        self.data_feeder = data_feeder
        self.noise_feeder = noise_feeder
        self.batch_size = batch_size
        self.epoch = epoch
        self.sample_path = sample_path

        self._init_input(data_feeder.data_shape, data_feeder.label_shape,
                         noise_feeder.shape)
        self._build_graph(self._mnist_disc_base, 
                          self._mnist_disc_decision,
                          self._mnist_classifier,
                          self._mnist_generator)
        #self._init_class_metric()
        self._init_loss()
        self._init_trainable_vars()
        self._init_optimizer(opt_param)    
        self._init_sample(num_sample_set)
        self._init_tb(log_path)

    def _init_input(self, data_shape, label_shape, noise_shape):
        self.data_real_input = tf.placeholder(tf.float32, [self.batch_size] + data_shape)
        self.label_input = tf.placeholder(tf.float32, [self.batch_size] + label_shape)
        self.noise_input = tf.placeholder(tf.float32, [self.batch_size] + noise_shape)

        self.flag_disc_train = tf.placeholder(tf.bool)
        self.flag_gen_train = tf.placeholder(tf.bool)
        self.flag_class_train = tf.placeholder(tf.bool)

    def _build_graph(self, disc_base, disc_decision, classifier, generator):
        self.gen_raw_out, self.gen_sig_out = \
            generator('gen', self.noise_input, self.label_input, self.flag_gen_train)

        disc_real_base = disc_base('disc-base', self.data_real_input, self.flag_disc_train)
        disc_fake_base = disc_base('disc-base', self.gen_sig_out, self.flag_disc_train, True)

        self.disc_real_raw_out, self.disc_real_sig_out = \
            disc_decision('disc-decision', disc_real_base, self.flag_disc_train)
        self.disc_fake_raw_out, self.disc_fake_sig_out = \
            disc_decision('disc-decision', disc_fake_base, self.flag_disc_train, True)

        self.class_real_raw_out, self.class_real_sm_out = \
            classifier('classifier', disc_real_base, self.flag_class_train)
        self.class_fake_raw_out, self.class_fake_sm_out = \
            classifier('classifier', disc_fake_base, self.flag_class_train, True)

    def _init_loss(self):
        loss_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_real_raw_out, labels=tf.ones_like(self.disc_real_raw_out)))
        loss_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_fake_raw_out, labels=tf.zeros_like(self.disc_fake_raw_out)))
        self.loss_disc = loss_disc_real + loss_disc_fake

        self.loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_fake_raw_out, labels=tf.ones_like(self.disc_fake_raw_out)))

        loss_class_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.class_real_raw_out, labels=self.label_input))
        loss_class_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.class_fake_raw_out, labels=self.label_input))
        self.loss_class = loss_class_real + loss_class_fake

    def _init_trainable_vars(self):
        vars_disc1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc-base')
        vars_disc2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc-decision')
        self.vars_disc = list(set(vars_disc1 + vars_disc2))
        self.vars_gen  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')
        self.vars_class = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'classifier')

    def _init_optimizer(self, opt_param):
         with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.opt_disc = tf.train.AdamOptimizer(opt_param['learning_rate']) \
                .minimize(self.loss_disc, var_list=self.vars_disc)
            self.opt_gen = tf.train.AdamOptimizer(opt_param['learning_rate']*opt_param['gen_lr_mult']) \
                .minimize(self.loss_gen, var_list=self.vars_gen)
            self.opt_class = tf.train.AdamOptimizer(opt_param['learning_rate']*opt_param['class_lr_mult']) \
                .minimize(self.loss_class, var_list=list(set(self.vars_disc+self.vars_gen+self.vars_class)))
        #self.opt_disc = tf.train.AdamOptimizer(opt_param['learning_rate']) \
        #    .minimize(self.loss_disc, var_list=self.vars_disc)
        #self.opt_gen = tf.train.AdamOptimizer(opt_param['learning_rate']*opt_param['gen_lr_mult']) \
        #    .minimize(self.loss_gen, var_list=self.vars_gen)
        #self.opt_class = tf.train.AdamOptimizer(opt_param['learning_rate']*opt_param['class_lr_mult']) \
        #    .minimize(self.loss_class, var_list=list(set(self.vars_disc+self.vars_gen+self.vars_class)))

    def _init_class_metric(self):
        labels = tf.reshape(self.label_input, self.class_real_sm_out.shape)
        acc_real, _ = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                          predictions=tf.argmax(self.class_real_sm_out, 1))
        acc_fake, _ = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                          predictions=tf.argmax(self.class_fake_sm_out, 1))
        self.acc_class = tf.reduce_mean((acc_real + acc_fake)/2.0)

    def _init_tb(self, log_path):
        data_real = tf.summary.image('data_real', self.data_real_input)
        loss_disc = tf.summary.scalar('loss_disc', self.loss_disc)
        self.disc_summary = tf.summary.merge([data_real, loss_disc])

        data_fake = tf.summary.image('data_fake', self.gen_sig_out)
        loss_gen = tf.summary.scalar('loss_gen', self.loss_gen)
        self.gen_summary = tf.summary.merge([data_fake, loss_gen])

        #loss_class = tf.summary.scalar('loss_class', self.loss_class)
        #acc_class = tf.summary.scalar('acc_class', self.acc_class)
        #self.class_summary = tf.summary.merge([loss_class, acc_class])
        loss_class = tf.summary.scalar('loss_class', self.loss_class)
        self.class_summary = tf.summary.merge([loss_class])

        self.summary_writer = tf.summary.FileWriter(log_path, self.sess.graph)

    def _init_sample(self, num_set):
        label_list = []
        sample_list = []
        cnt = self.data_feeder.num_class * num_set

        while cnt > 0:
            data, label = self.data_feeder.get(1)
            data = data[0]
            label = label[0]

            if label in label_list:
                idx = label_list.index(label)
                if len(sample_list[idx]) < num_set:
                    sample_list[idx].append(data)
                    cnt -= 1
            else:
                label_list.append(label)
                sample_list.append([data])
                cnt -= 1

        self.data_feeder.rewind()

        if self.data_feeder.one_hot:
            sorted_labels = sorted(label_list, reverse=True)
        else:
            sorted_labels = sorted(label_list)
        sample_data = []
        sample_label = []
        for i in sorted_labels:
            for j in sample_list[label_list.index(i)]:
                sample_label.append(i)
                sample_data.append(j)

        self.sample_data = np.array(sample_data)      
        self.sample_label = np.array(sample_label).reshape([num_set * self.data_feeder.num_class] + self.data_feeder.label_shape)  

        self.sample_noise = np.array(self.noise_feeder.get(num_set * self.data_feeder.num_class))

        self.sample_image_maker = tile_image_maker.TileImageMaker()

    def _one_hot_encoding(self, label, num_class):
        encoded = []
        for i in range(len(label)):
            vec = np.zeros([num_class], np.float32)
            vec[label[i]] = 1
            encoded.append(vec)

    def _write_sample(self, epoch):
        #gen_data, l_d, l_g, l_c, a_c = self.sess.run(
        #    [self.gen_sig_out, self.loss_disc, self.loss_gen,
        #     self.loss_class, self.acc_class], 
        #    feed_dict={self.data_real_input: self.sample_data,
        #               self.label_input: self.sample_label,
        #               self.noise_input: self.sample_noise,
        #               self.flag_disc_train: False, self.flag_gen_train: False,
        #               self.flag_class_train: False})
        #tile_image = self.sample_image_maker.create(gen_data, None, self.data_feeder.num_class)
        #image_path = os.path.normpath(os.path.join(self.sample_path, 'e%04d_ld%.8f_lg%.8f%_lc%.8f_ac%.8f.png' % (epoch+1, l_d, l_g, l_c, a_c)))
        #imageio.imwrite(image_path, tile_image)
        gen_data, l_d, l_g, l_c = self.sess.run(
            [self.gen_sig_out, self.loss_disc, self.loss_gen, self.loss_class], 
            feed_dict={self.data_real_input: self.sample_data,
                       self.label_input: self.sample_label,
                       self.noise_input: self.sample_noise,
                       self.flag_disc_train: False, self.flag_gen_train: False,
                       self.flag_class_train: False})
        tile_image = self.sample_image_maker.create(gen_data, None, self.data_feeder.num_class)
        image_path = os.path.normpath(os.path.join(self.sample_path, 'e%04d_ld%.8f_lg%.8f_lc%.8f.png' % (epoch+1, l_d, l_g, l_c)))
        imageio.imwrite(image_path, tile_image)
            

    def train(self):
        tf.global_variables_initializer().run()
        
        iter_per_epoch = int(self.data_feeder.num_data // self.batch_size)

        start_time = time.time()

        for e in xrange(self.epoch):
            for i in xrange(0, iter_per_epoch):
                data_real, label = self.data_feeder.get(self.batch_size)
                data_real = np.array(data_real)
                label = np.array(label).reshape([self.batch_size] + self.data_feeder.label_shape)
                
                noise = self.noise_feeder.get(self.batch_size)

                iter = e * iter_per_epoch + i

                if iter == 0:
                    self._write_sample(-1)

                _, loss_disc, summary = self.sess.run(
                    [self.opt_disc, self.loss_disc, self.disc_summary],
                    feed_dict={self.data_real_input: data_real,
                               self.noise_input: noise,
                               self.label_input: label,
                               self.flag_disc_train: True, self.flag_gen_train: True,
                               self.flag_class_train: True})
                self.summary_writer.add_summary(summary, iter)

                #_, loss_gen, summary_gen, _, loss_class, acc_class, summary_class = self.sess.run(
                #    [self.opt_gen, self.loss_gen, self.gen_summary, 
                #     self.opt_class, self.loss_class, self.acc_class, self.class_summary],
                #    feed_dict={self.data_real_input: data_real,
                #               self.noise_input: noise,
                #               self.label_input: label,
                #               self.flag_disc_train: True, self.flag_gen_train: True,
                #               self.flag_class_train: True})
                #self.summary_writer.add_summary(summary_gen, iter)
                #self.summary_writer.add_summary(summary_class, iter)

                #print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss_d: %.8f, loss_g: %.8f, loss_c: %.8f, acc: %.8f" \
                #    % (e+1, self.epoch, i+1, iter_per_epoch, time.time() - start_time, loss_disc, loss_gen, loss_class, acc_class))

                _, loss_gen, summary_gen, _, loss_class, summary_class = self.sess.run(
                    [self.opt_gen, self.loss_gen, self.gen_summary, 
                     self.opt_class, self.loss_class, self.class_summary],
                    feed_dict={self.data_real_input: data_real,
                               self.noise_input: noise,
                               self.label_input: label,
                               self.flag_disc_train: True, self.flag_gen_train: True,
                               self.flag_class_train: True})
                self.summary_writer.add_summary(summary_gen, iter)
                self.summary_writer.add_summary(summary_class, iter)

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss_d: %.8f, loss_g: %.8f, loss_c: %.8f" \
                    % (e+1, self.epoch, i+1, iter_per_epoch, time.time() - start_time, loss_disc, loss_gen, loss_class))

                

            self._write_sample(e)
        
        
        
        
    #def _mnist_discriminator(self, name, data_input, label_input, 
    #                         train_flag, reuse=False):
    #    with tf.variable_scope(name, reuse=reuse):
    #        #x = tf.reshape(data_input, [self.batch_size, 1, 1, -1])
    #        y = tf.reshape(label_input, [self.batch_size, 1, 1, -1])
    #        new_y_shape = [data_input.shape[0], data_input.shape[1], data_input.shape[2], y.shape[3]]
    #        xy = tf.concat(values=[data_input, y*tf.ones(new_y_shape)], axis=3)

    #        h0_0 = tf.layers.conv2d(xy, 64, [4, 4], [2, 2], 'same')
    #        h0_1 = tf.nn.leaky_relu(h0_0)
        
    #        h1_0 = tf.layers.conv2d(h0_1, 128, [4, 4], [2, 2], 'same')
    #        h1_1 = tf.contrib.layers.batch_norm(h1_0, is_training=train_flag, updates_collections=None)
    #        h1_2 = tf.nn.leaky_relu(h1_1)

    #        h2_0 = tf.reshape(h1_2, [self.batch_size, -1])
    #        h2_1 = tf.layers.dense(h2_0, 1024)
    #        h2_2 = tf.contrib.layers.batch_norm(h2_1, is_training=train_flag, updates_collections=None)
    #        h2_3 = tf.nn.leaky_relu(h2_2)

    #        h3_0_0 = tf.layers.dense(h2_3, 1)
    #        sig_o = tf.nn.sigmoid(h3_0_0)
    
    def _mnist_disc_base(self, name, data_input, train_flag, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            h0_0 = tf.layers.conv2d(data_input, 64, [4, 4], [2, 2], 'same')
            h0_1 = tf.nn.leaky_relu(h0_0)
        
            h1_0 = tf.layers.conv2d(h0_1, 128, [4, 4], [2, 2], 'same')
            h1_1 = tf.contrib.layers.batch_norm(h1_0, is_training=train_flag, updates_collections=None)
            h1_2 = tf.nn.leaky_relu(h1_1)

            h2_0 = tf.reshape(h1_2, [self.batch_size, -1])
            h2_1 = tf.layers.dense(h2_0, 1024)
            h2_2 = tf.contrib.layers.batch_norm(h2_1, is_training=train_flag, updates_collections=None)
            h2_3 = tf.nn.leaky_relu(h2_2)

        return h2_3

    def _mnist_disc_decision(self, name, disc_base_out, train_flag, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            h0_0 = tf.layers.dense(disc_base_out, 1)
            sig_o = tf.nn.sigmoid(h0_0)

        return h0_0, sig_o

    def _mnist_classifier(self, name, disc_base_out, train_flag, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            h0_0 = tf.layers.dense(disc_base_out, 128)
            h0_1 = tf.contrib.layers.batch_norm(h0_0, is_training=train_flag, updates_collections=None)
            h0_2 = tf.nn.leaky_relu(h0_1)

            h1 = tf.layers.dense(h0_2, self.data_feeder.num_class)
            softmax_o = tf.nn.softmax(h1)

        return h1, softmax_o        

    def _mnist_generator(self, name, noise_input, label_input,
                         train_flag):
        with tf.variable_scope(name):
            z = tf.reshape(noise_input, [self.batch_size, -1])
            y = tf.reshape(label_input, [self.batch_size, -1])
            zy = tf.concat(values=[z, y], axis=1)

            h0_0 = tf.layers.dense(zy, 1024)
            h0_1 = tf.contrib.layers.batch_norm(h0_0, is_training=train_flag, updates_collections=None)
            h0_2 = tf.nn.relu(h0_1)

            h1_0 = tf.layers.dense(h0_2, 7*7*128)
            h1_1 = tf.contrib.layers.batch_norm(h1_0, is_training=train_flag, updates_collections=None)
            h1_2 = tf.nn.relu(h1_1)
            
            h2_0 = tf.reshape(h1_2, [self.batch_size, 7, 7, 128])
            h2_1 = tf.layers.conv2d_transpose(h2_0, 64, [4, 4], [2, 2], 'same')
            h2_2 = tf.contrib.layers.batch_norm(h2_1, is_training=train_flag, updates_collections=None)
            h2_3 = tf.nn.relu(h2_2)

            h3 = tf.layers.conv2d_transpose(h2_3, 1, [4, 4], [2, 2], 'same')

            sig_o = tf.nn.sigmoid(h3)

        return h3, sig_o
            

    