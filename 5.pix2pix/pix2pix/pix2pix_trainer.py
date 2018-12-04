import seg2face_data_maker
import tile_image_maker

import numpy as np
import cv2
from six.moves import xrange
import tensorflow as tf

import math
import os
import time

class Pix2PixTrainer(object):
    def __init__(self, sess, data_feeder, img_size, landmark_dat, 
                 opt_param, batch_size, epoch, log_path, sample_path):
        self._sess = sess
        self._data_feeder = data_feeder
        self._seg_maker = seg2face_data_maker.Seg2FaceDataMaker(landmark_dat, img_size)

        self._batch_size = batch_size
        self._epoch = epoch
        self._sample_path = sample_path

        self._init_input(img_size)
        self._build_graph(self._discriminator, self._generator)
        self._init_loss()
        self._init_trainable_vars()
        self._init_optimizer(opt_param)
        self._init_sample()
        self._init_tb(log_path)

    def _init_input(self, img_size):
        self._input_img = tf.placeholder(tf.float32, [self._batch_size]+img_size+[1])
        self._target_img = tf.placeholder(tf.float32, [self._batch_size]+img_size+[3])

        self._disc_bn_flag = tf.placeholder(tf.bool)
        self._gen_bn_flag = tf.placeholder(tf.bool)

        self._gen_dropout_prop = tf.placeholder(tf.float32)

    def _build_graph(self, discriminator, generator):
        self._gen_out = generator(
            'gen', self._input_img, self._gen_bn_flag, self._gen_dropout_prop)

        disc_real_input = tf.concat(values=[self._target_img, self._input_img], axis=3)
        self._disc_real_out = discriminator(
            'disc', disc_real_input, self._disc_bn_flag, False)
        disc_fake_input = tf.concat(values=[self._gen_out, self._input_img], axis=3)
        self._disc_fake_out = discriminator(
            'disc', disc_fake_input, self._disc_bn_flag, True)

    def _init_loss(self):
        EPS = 0.0001
        self._loss_disc = tf.reduce_mean(
            -(tf.log(self._disc_real_out + EPS) + tf.log(1 - self._disc_fake_out + EPS)))

        self._loss_gen_gan = tf.reduce_mean(-tf.log(self._disc_fake_out + EPS))
        self._loss_gen_l1 = tf.reduce_mean(tf.abs(self._target_img - self._gen_out))
        self._loss_gen = self._loss_gen_gan + self._loss_gen_l1

    def _init_trainable_vars(self):
        self._var_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc')
        self._var_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')

    def _init_optimizer(self, opt_param):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._opt_disc = tf.train.AdamOptimizer(opt_param['learning_rate']) \
                .minimize(self._loss_disc, var_list=self._var_disc)
            self._opt_gen = tf.train.AdamOptimizer(opt_param['learning_rate']*opt_param['gen_lr_mult']) \
                .minimize(self._loss_gen, var_list=self._var_gen)

    def _init_sample(self):
        self._tile_image_maker = tile_image_maker.TileImageMaker()
        self._sample_target_img, self._sample_input_img = self._get_seg_img(self._batch_size, False)
        
    def _init_tb(self, log_path):
        target_img = tf.summary.image('target_img', self._target_img)
        input_img = tf.summary.image('input_img', self._input_img)
        loss_disc = tf.summary.scalar('loss_disc', self._loss_disc)
        self._disc_summary = tf.summary.merge([target_img, input_img, loss_disc])

        generated = tf.summary.image('generated', self._gen_out)
        loss_gen_gan = tf.summary.scalar('loss gen_GAN', self._loss_gen_gan)
        loss_gen_l1 = tf.summary.scalar('loss gen_L1', self._loss_gen_l1)
        loss_gen = tf.summary.scalar('loss_gen', self._loss_gen)
        self._gen_summary = tf.summary.merge([generated, loss_gen_gan, loss_gen_l1, loss_gen])

        self._tb_writer = tf.summary.FileWriter(log_path, self._sess.graph)

    def _get_seg_img(self, num, train_data=True):
        target_img = []
        seg_img = []
        while len(seg_img) < num:
            if train_data:
                img_list, bbox_list = self._data_feeder.get_train(1)
            else:
                img_list, bbox_list = self._data_feeder.get_validation(1)

            target, seg = self._seg_maker.make(img_list[0])
            if seg is not None:
                target_img.append(target)
                seg_img.append(seg)
        
        target_img = (np.array(target_img, np.float32) / 255.0 * 2.0) - 1.0
        seg_img = (np.array(seg_img, np.float32) / 4.0) - 1.0
        seg_img = seg_img.reshape(list(seg_img.shape) + [1])
        return target_img, seg_img

    def _generator(self, name, input, bn_train, dropout_prop):
        with tf.variable_scope(name):
            # encoder
            e0_0 = tf.layers.conv2d(input, 64, [4, 4], [2, 2], 'same')
            e0_1 = tf.nn.leaky_relu(e0_0)

            e1_0 = tf.layers.conv2d(e0_1, 128, [4, 4], [2, 2], 'same')
            e1_1 = tf.contrib.layers.batch_norm(e1_0, is_training=bn_train, updates_collections=None)
            e1_2 = tf.nn.leaky_relu(e1_1)

            e2_0 = tf.layers.conv2d(e1_2, 256, [4, 4], [2, 2], 'same')
            e2_1 = tf.contrib.layers.batch_norm(e2_0, is_training=bn_train, updates_collections=None)
            e2_2 = tf.nn.leaky_relu(e2_1)

            e3_0 = tf.layers.conv2d(e2_2, 512, [4, 4], [2, 2], 'same')
            e3_1 = tf.contrib.layers.batch_norm(e3_0, is_training=bn_train, updates_collections=None)
            e3_2 = tf.nn.leaky_relu(e3_1)

            e4_0 = tf.layers.conv2d(e3_2, 512, [4, 4], [2, 2], 'same')
            e4_1 = tf.contrib.layers.batch_norm(e4_0, is_training=bn_train, updates_collections=None)
            e4_2 = tf.nn.leaky_relu(e4_1)
            
            e5_0 = tf.layers.conv2d(e4_2, 512, [4, 4], [2, 2], 'same')
            e5_1 = tf.contrib.layers.batch_norm(e5_0, is_training=bn_train, updates_collections=None)
            e5_2 = tf.nn.leaky_relu(e5_1)

            e6_0 = tf.layers.conv2d(e5_2, 512, [4, 4], [2, 2], 'same')
            e6_1 = tf.contrib.layers.batch_norm(e6_0, is_training=bn_train, updates_collections=None)
            e6_2 = tf.nn.leaky_relu(e6_1)

            e7_0 = tf.layers.conv2d(e6_2, 512, [4, 4], [2, 2], 'same')
            e7_1 = tf.contrib.layers.batch_norm(e7_0, is_training=bn_train, updates_collections=None)
            e7_2 = tf.nn.relu(e7_1)

            # decoder
            d0_0 = tf.layers.conv2d_transpose(e7_2, 512, [4, 4], [2, 2], 'same')
            d0_1 = tf.contrib.layers.batch_norm(d0_0, is_training=bn_train, updates_collections=None)
            d0_2 = tf.nn.dropout(d0_1, dropout_prop)
            d0_3 = tf.nn.relu(d0_2)
            d0_4 = tf.concat([d0_3, e6_2], axis=3)

            d1_0 = tf.layers.conv2d_transpose(d0_4, 512, [4, 4], [2, 2], 'same')
            d1_1 = tf.contrib.layers.batch_norm(d1_0, is_training=bn_train, updates_collections=None)
            d1_2 = tf.nn.dropout(d1_1, dropout_prop)
            d1_3 = tf.nn.relu(d1_2)
            d1_4 = tf.concat([d1_3, e5_2], axis=3)

            d2_0 = tf.layers.conv2d_transpose(d1_4, 512, [4, 4], [2, 2], 'same')
            d2_1 = tf.contrib.layers.batch_norm(d2_0, is_training=bn_train, updates_collections=None)
            d2_2 = tf.nn.dropout(d2_1, dropout_prop)
            d2_3 = tf.nn.relu(d2_2)
            d2_4 = tf.concat([d2_3, e4_2], axis=3)

            d3_0 = tf.layers.conv2d_transpose(d2_4, 512, [4, 4], [2, 2], 'same')
            d3_1 = tf.contrib.layers.batch_norm(d3_0, is_training=bn_train, updates_collections=None)
            d3_2 = tf.nn.relu(d3_1)
            d3_3 = tf.concat([d3_2, e3_2], axis=3)

            d4_0 = tf.layers.conv2d_transpose(d3_3, 256, [4, 4], [2, 2], 'same')
            d4_1 = tf.contrib.layers.batch_norm(d4_0, is_training=bn_train, updates_collections=None)
            d4_2 = tf.nn.relu(d4_1)
            d4_3 = tf.concat([d4_2, e2_2], axis=3)

            d5_0 = tf.layers.conv2d_transpose(d4_3, 128, [4, 4], [2, 2], 'same')
            d5_1 = tf.contrib.layers.batch_norm(d5_0, is_training=bn_train, updates_collections=None)
            d5_2 = tf.nn.relu(d5_1)
            d5_3 = tf.concat([d5_2, e1_2], axis=3)

            d6_0 = tf.layers.conv2d_transpose(d5_3, 64, [4, 4], [2, 2], 'same')
            d6_1 = tf.contrib.layers.batch_norm(d6_0, is_training=bn_train, updates_collections=None)
            d6_2 = tf.nn.relu(d6_1)
            d6_3 = tf.concat([d6_2, e0_1], axis=3)

            d7_0 = tf.layers.conv2d_transpose(d6_3, 3, [4, 4], [2, 2], 'same')
            d7_1 = tf.nn.tanh(d7_0)
        return d7_1

    def _discriminator(self, name, input, bn_train, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            h0_0 = tf.layers.conv2d(input, 64, [4, 4], [2, 2], 'same')
            h0_1 = tf.nn.leaky_relu(h0_0)

            h1_1 = tf.layers.conv2d(h0_1, 128, [4, 4], [2, 2], 'same')
            h1_2 = tf.contrib.layers.batch_norm(h1_1, is_training=bn_train, updates_collections=None)
            h1_3 = tf.nn.leaky_relu(h1_2)

            h2_1 = tf.layers.conv2d(h1_3, 256, [4, 4], [2, 2], 'same')
            h2_2 = tf.contrib.layers.batch_norm(h2_1, is_training=bn_train, updates_collections=None)
            h2_3 = tf.nn.leaky_relu(h2_2)

            h3_filter = tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.02))
            h3_0 = tf.pad(h2_3, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            h3_1 = tf.nn.conv2d(h3_0, h3_filter, strides=[1, 1, 1, 1], padding='VALID')
            h3_2 = tf.contrib.layers.batch_norm(h3_1, is_training=bn_train, updates_collections=None)
            h3_3 = tf.nn.leaky_relu(h3_2)

            h4_filter = tf.Variable(tf.truncated_normal([4, 4, 512, 1], stddev=0.02))
            h4_0 = tf.pad(h3_3, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            h4_1 = tf.nn.conv2d(h4_0, h4_filter, strides=[1, 1, 1, 1], padding='VALID')

            h5 = tf.nn.sigmoid(h4_1)

        return h5

    def _write_sample(self, epoch):
        gen_data, l_d, l_g = self._sess.run(
            [self._gen_out, self._loss_disc, self._loss_gen],
            feed_dict={self._target_img: self._sample_target_img,
                       self._input_img: self._sample_input_img,
                       self._disc_bn_flag: False, self._gen_bn_flag: False,
                       self._gen_dropout_prop: 0.5})
        tile_image = self._tile_image_maker.create(gen_data, None, int(math.sqrt(self._batch_size)))
        tile_image = (tile_image + 1. / 2.)
        image_path = os.path.normpath(
            os.path.join(self._sample_path, 'e%04d_ld%.8f_lg%.8f.png' % (epoch+1, l_d, l_g)))
        cv2.imwrite(image_path, tile_image)

    def train(self):
        tf.global_variables_initializer().run()

        iter_per_epoch = int(self._data_feeder.num_train // self._batch_size)

        start_time = time.time()


        for e in xrange(self._epoch):
            for i in xrange(0, iter_per_epoch):
                iter = e * iter_per_epoch + i

                if iter == 0:
                    self._write_sample(-1)

                target_img, input_img = self._get_seg_img(self._batch_size)

                _, loss_disc, summary = self._sess.run(
                    [self._opt_disc, self._loss_disc, self._disc_summary],
                    feed_dict={self._target_img: target_img,
                               self._input_img: input_img,
                               self._disc_bn_flag: True, self._gen_bn_flag: True,
                               self._gen_dropout_prop: 0.5})
                self._tb_writer.add_summary(summary, iter)

                _, loss_gen, summary = self._sess.run(
                    [self._opt_gen, self._loss_gen, self._gen_summary],
                    feed_dict={self._target_img: target_img,
                               self._input_img: input_img,
                               self._disc_bn_flag: True, self._gen_bn_flag: True,
                               self._gen_dropout_prop: 0.5})
                self._tb_writer.add_summary(summary, iter)

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss_d: %.8f, loss_g: %.8f" \
                    % (e+1, self.epoch, i+1, iter_per_epoch, time.time() - start_time, loss_disc, loss_gen))

            self._write_sample(e)