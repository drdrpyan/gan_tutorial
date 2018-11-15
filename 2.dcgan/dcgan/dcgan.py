import image_loader
import sample_image_maker

import tensorflow as tf
#import tensorboard as tb
from glob import glob
import numpy as np
import scipy.misc
from six.moves import xrange
import imageio

import math
import os
import time

#from ops import *



class Summary(object):
    def __init__(self):
        try:
            self.image_summary = tf.image_summary
            self.scalar_summary = tf.scalar_summary
            self.histogram_summary = tf.histogram_summary
            self.merge_summary = tf.merge_summary
            self.SummaryWriter = tf.train.SummaryWriter
        except:
            self.image_summary = tf.summary.image
            self.scalar_summary = tf.summary.scalar
            self.histogram_summary = tf.summary.histogram
            self.merge_summary = tf.summary.merge
            self.SummaryWriter = tf.summary.FileWriter

class DCGAN(object):
    def __init__(self, sess, 
                 img_path, format='png', shuffle=True, resize=64, center_crop_size=64, color='color', 
                 batch_size=64, z_dim=100,
                 num_sample=64, sampling_iter=100, sample_z=None, sample_path='./sample'):
        self.sess = sess

        self.image_loader = image_loader.ImageLoader(img_path, format, shuffle, resize, center_crop_size, color)

        self.image_shape = self.image_loader.image_shape
        
        self.batch_size = batch_size
        self.z_dim = z_dim

        self._init_sampling_config(num_sample, sampling_iter, sample_path, sample_z)

        self.summary = Summary()
        self._init_summary(self.summary)

        
    def _init_sampling_config(self, num_sample, sampling_iter, sample_path, sample_z=None):
        self.num_sample = num_sample
        self.sampling_iter = sampling_iter
        
        if sample_z is None:
            self.sample_z = np.random.uniform(-1, 1, size=(self.num_sample, self.z_dim))
        else:
            self.sample_z = sample_z

        self.sample_maker = sample_image_maker.TileImageMaker()
        self.sample_cols = math.floor(math.sqrt(num_sample))
        self.sample_rows = math.ceil(num_sample / self.sample_cols)

        self.sample_path = sample_path

        self.sample_images = self.image_loader.load_by_idx(range(0, self.num_sample))

        self._build_model()


    def train(self, epoch, learning_rate=0.01, beta1=0.9, sampling=True):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')):
            g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.loss_G, var_list=self.vars_G)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')):
            d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.loss_D, var_list=self.vars_D)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        start_time = time.time()
        counter = 1

        batch_idx = self.image_loader.num_images // self.batch_size

        for e in xrange(epoch):
            for idx in xrange(0, int(batch_idx)):
                batch_img = self.image_loader.load_by_idx(range(idx*self.batch_size, (idx+1)*self.batch_size))
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                _, summary_str = self.sess.run([d_opt, self.summary_g],
                    feed_dict={self.input: batch_img, self.z: batch_z, 
                               self.G_train_flag: False, self.D_train_flag: True})
                self.summary_writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g_opt, self.summary_loss_g],
                    feed_dict={self.z: batch_z, self.G_train_flag: True, self.D_train_flag: False})
                self.summary_writer.add_summary(summary_str, counter)

                err_D_real = self.loss_D_real.eval({self.input: batch_img, self.D_train_flag: False})
                err_D_fake = self.loss_D_fake.eval({self.z: batch_z, self.G_train_flag: False, self.D_train_flag: False})
                err_g = self.loss_G.eval({self.z: batch_z, self.G_train_flag: False, self.D_train_flag: False})

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss_d: %.8f, loss_g: %.8f" \
                    % (e+1, epoch, idx+1, batch_idx, time.time() - start_time, err_D_real+err_D_fake, err_g))

                if sampling and (counter % self.sampling_iter == 0):
                    img_name = '{}/train_{:02d}_{:04d}.png'.format(self.sample_path, e+1, idx+1)
                    self._save_sample(img_name)

                counter += 1
        


        

    def _init_summary(self, summary):
        self.summary_z = summary.histogram_summary('z', self.z)
        
        self.summary_d_real = summary.histogram_summary('D_real', self.D_real)
        self.summary_d_fake = summary.histogram_summary('D_fake', self.D_fake)

        self.summary_g = summary.image_summary('G', self.G)

        self.summary_loss_d_real = summary.scalar_summary('loss_d_real', self.loss_D_real)
        self.summary_loss_d_fake = summary.scalar_summary('loss_d_fake', self.loss_D_fake)
        self.summary_loss_d = summary.scalar_summary('loss_d', self.loss_D)
        
        self.summary_loss_g = summary.scalar_summary('loss_g', self.loss_G)

        self.summary_g = summary.merge_summary(
            [self.summary_z, self.summary_d_fake, self.summary_g, 
                self.summary_loss_d_fake, self.summary_loss_g])
        self.summary_d = summary.merge_summary(
            [self.summary_z, self.summary_d_real, self.summary_loss_d_real,
                self.summary_loss_d])

        self.summary_writer = summary.SummaryWriter('./logs', self.sess.graph)
        


    def _build_model(self):
        self.input = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G_train_flag = tf.placeholder(tf.bool, name='G_train_flag')
        self.D_train_flag = tf.placeholder(tf.bool, name='D_train_flag')
        
        self.G = self._build_generator(self.z, self.G_train_flag)

        self.D_real = self._build_discriminator(self.input, self.D_train_flag)
        self.D_fake = self._build_discriminator(self.G, self.D_train_flag, reuse_var=True)

        self.loss_G = tf.reduce_mean(
            self._sigmoid_cross_entropy_with_logits(self.D_fake, tf.ones_like(self.D_real)))
        self.loss_D_real = tf.reduce_mean(
            self._sigmoid_cross_entropy_with_logits(self.D_real, tf.ones_like(self.D_real)))
        self.loss_D_fake = tf.reduce_mean(
            self._sigmoid_cross_entropy_with_logits(self.D_fake, tf.zeros_like(self.D_fake)))
        self.loss_D = self.loss_D_real + self.loss_D_fake

        #trainable_vars = tf.trainable_variables()
        #self.vars_G = [var for var in trainable_vars if 'generator' in var.name]
        #self.vars_D = [var for var in trainable_vars if 'discriminator' in var.name]
        self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

        self.saver = tf.train.Saver()


    def _build_generator(self, input, train_flag, name='generator'):
        with tf.variable_scope(name) as scope:
            h0_0 = tf.reshape(input, [-1, 1, 1, self.z_dim])
            h0_1 = tf.layers.conv2d_transpose(h0_0, 1024, [4, 4], [4, 4], 'same', name='deconv0')
            h0_2 = tf.layers.batch_normalization(h0_1, training=train_flag, name='deconv0_bn')
            h0_3 = tf.nn.relu(h0_2, name='deconv0_relu')

            h1_0 = tf.layers.conv2d_transpose(h0_1, 512, [4, 4], [2, 2], 'same', name='deconv1')
            h1_1 = tf.layers.batch_normalization(h1_0, training=train_flag, name='deconv1_bn')
            h1_2 = tf.nn.relu(h1_1, name='deconv1_relu')

            h2_0 = tf.layers.conv2d_transpose(h1_1, 256, [4, 4], [2, 2], 'same', name='deconv2')
            h2_1 = tf.layers.batch_normalization(h2_0, training=train_flag, name='deconv2_bn')
            h2_2 = tf.nn.relu(h2_1, name='deconv2_relu')

            h3_0 = tf.layers.conv2d_transpose(h2_2, 128, [4, 4], [2, 2], 'same', name='deconv3')
            h3_1 = tf.layers.batch_normalization(h3_0, training=train_flag, name='deconv3_bn')
            h3_2 = tf.nn.relu(h3_1, name='deconv3_relu')

            #h4 = tf.layers.conv2d_transpose(h3_2, 3, [4, 4], [2, 2], 'same', name='deconv4')

            if self.image_shape[0] < h3_2.shape[1] or self.image_shape[0] % h3_2.shape[1] != 0:
                raise Exception('(image height)/(feature map height)=C, ' 
                                + 'but (image height)=%d, (feature map height)=%d' 
                                % (img_shape[0], h3_2.shape[1]))
            if self.image_shape[1] < h3_2.shape[2] or self.image_shape[1] % h3_2.shape[2] != 0:
                raise Exception('(image width)/(feature map width)=C, ' 
                                + 'but (image width)=%d, (feature map width)=%d' 
                                % (img_shape[1], h3_2.shape[2]))
            block_size = self.image_shape[0] / int(h3_2.shape[1])
            if block_size != self.image_shape[1] < h3_2.shape[2]:
                raise Exception('(image height)/(feature map height) != (image width)/(feature map width)')
                
            ch = block_size * block_size * self.image_shape[2]
            h4_0 = tf.layers.conv2d(h3_2, ch, [3, 3], [1, 1], 'same', name='conv0')
            #h4_1 = tf.reshape(h4_0, self.image_shape, name='reshape')
            h4_1 = tf.depth_to_space(h4_0, block_size, name='pix_shuffle')

            o = tf.nn.tanh(h4_1, name='tanh_out')

        return o


    def _build_discriminator(self, input, train_flag, name='discriminator', reuse_var=False):
        with tf.variable_scope(name) as scope:
            if reuse_var:
                scope.reuse_variables()
    
            h0_0 = tf.layers.conv2d(input, 128, [4, 4], (2, 2), 'same', name='conv0')
            h0_1 = tf.layers.batch_normalization(h0_0, training=train_flag, name='conv0_bn')
            h0_2 = tf.nn.leaky_relu(h0_1, name='conv0_lrelu')
            
            h1_0 = tf.layers.conv2d(h0_2, 256, [4, 4], (2, 2), 'same', name='conv1')
            h1_1 = tf.layers.batch_normalization(h1_0, training=train_flag, name='conv1_bn')
            h1_2 = tf.nn.leaky_relu(h1_1, name='conv1_lrelu')

            h2_0 = tf.layers.conv2d(h1_2, 512, [4, 4], (2, 2), 'same', name='conv2')
            h2_1 = tf.layers.batch_normalization(h2_0, training=train_flag, name='conv2_bn')
            h2_2 = tf.nn.leaky_relu(h2_1, name='conv2_lrelu')

            h3_0 = tf.layers.conv2d(h2_2, 1024, [4, 4], (2, 2), 'same', name='conv3')
            h3_1 = tf.layers.batch_normalization(h3_0, training=train_flag, name='conv3_bn')
            h3_2 = tf.nn.leaky_relu(h3_1, name='conv3_lrelu')

            pool_size = [int(h3_2.shape[1]), int(h3_2.shape[2])]
            h4 = tf.layers.average_pooling2d(h3_2, pool_size, (1, 1), name='avg_pool')
            #h4 = tf.reduce_mean(h3_2, axis=[1, 2], name='avg_pool')

            h5 = tf.layers.conv2d(h4, 1, [1, 1], name='conv4')

            o = tf.sigmoid(h5, name='sig_out')

        return o


    def _sigmoid_cross_entropy_with_logits(self, x, y):
        try:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        except:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    def _save_sample(self, path):
        samples, loss_d, loss_g = self.sess.run(
            [self.G, self.loss_D, self.loss_G], 
            feed_dict={self.z: self.sample_z, self.input: self.sample_images, 
                       self.G_train_flag: False, self.D_train_flag: False})
        tile_image = self.sample_maker.create(samples, cols=np.floor(np.sqrt(self.num_sample)))
        tile_image = np.squeeze(tile_image)

        #img_name = './{}/train_{:02d}_{:04d}.png'.format(path, epoch, iter)

        #scipy.misc.imsave(path, tile_image)

        pixel_range = self.image_loader.norm_range
        tile_image = ((tile_image - pixel_range[0]) / (pixel_range[1]-pixel_range[0])) * 255
        tile_image = tile_image.astype(np.uint8)

        imageio.imwrite(path, tile_image)


#s_h, s_w = self.output_height, self.output_width
#s_h2, s_h4 = int(s_h/2), int(s_h/4)
#s_w2, s_w4 = int(s_w/2), int(s_w/4)

## yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
#yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
#z = concat([z, y], 1)

#h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
#h0 = concat([h0, y], 1)

#h1 = tf.nn.relu(self.g_bn1(
#    linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
#h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
#h1 = conv_cond_concat(h1, yb)

#h2 = tf.nn.relu(self.g_bn2(
#    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
#h2 = conv_cond_concat(h2, yb)

#return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))