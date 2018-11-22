import tensorflow as tf
import numpy as np
from six.moves import xrange
import time

class DCGANTrainer3(object):
    def __init__(self, sess, discriminator, generator, 
                 data_feeder, noise_feeder, log_path='./logs/'):
        self.sess = sess
        self.data_feeder = data_feeder
        self.noise_feeder = noise_feeder

        self._init_input(data_feeder.shape, noise_feeder.shape)    
        self._build_graph(discriminator, generator)
        self._init_output()
        self._init_loss()
        self._init_trainable_vars()
        self._init_tb_summary(log_path)

    def train(self, batch_size, epoch, opt_param):
        self._init_optimizer(opt_param)

        tf.global_variables_initializer().run()
        #tf.initialize_all_variables().run()

        
        iter_per_epoch = int(self.data_feeder.num_data // batch_size)

        label = self._make_label(batch_size, batch_size)

        start_time = time.time()

        #self._pretrain_disc(batch_size, start_time)        

        for e in xrange(epoch):
            for i in xrange(0, iter_per_epoch):
                data_real = np.array(self.data_feeder.get(batch_size))
                noise = self.noise_feeder.get(batch_size)

                _, loss_gen, data_fake, summary_gen_str = self.sess.run(
                    [self.opt_gen, self.loss_gen, self.gen_sig_out, self.summary_gen],
                    feed_dict={self.noise_input: noise, 
                               self.flag_disc_train: False, self.flag_gen_train: True})

                _, loss_disc, summary_disc_str = self.sess.run(
                    [self.opt_disc, self.loss_disc, self.summary_disc],
                    feed_dict={self.data_real_input: data_real, 
                               self.noise_input: noise,
                               #self.data_fake_input: data_fake,
                               #self.label_input: label,
                               self.flag_disc_train: True, self.flag_gen_train: False})

                summary_iter = e * iter_per_epoch + i
                self.summary_writer.add_summary(summary_disc_str, summary_iter)
                self.summary_writer.add_summary(summary_gen_str, summary_iter)

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss_d: %.8f, loss_g: %.8f" \
                    % (e+1, epoch, i+1, iter_per_epoch, time.time() - start_time, loss_disc, loss_gen))


            

    def _init_input(self, data_shape, noise_shape):
        self.data_real_input = tf.placeholder(tf.float32, [None] + data_shape)
        self.data_fake_input = tf.placeholder(tf.float32, [None] + data_shape)
        self.noise_input = tf.placeholder(tf.float32, [None] + noise_shape)

        self.label_input = tf.placeholder(tf.float32, [None, 1, 1, 1])

        self.flag_disc_train = tf.placeholder(tf.bool)
        self.flag_gen_train = tf.placeholder(tf.bool)

    def _build_graph(self, discriminator, generator):
        #self.data_input = tf.concat([self.data_real_input, self.data_fake_input], 0)

        #self.label_real = tf.ones([self.data_real_input.shape[0], 1, 1, 1])
        #self.label_fake = tf.zeros([self.data_fake_input.shape[0], 1, 1, 1])
        #self.label = tf.concat([self.label_real, self.label_fake], 0)

        self.gen = generator('gen', self.noise_input, self.flag_gen_train)
        #self.disc = discriminator('disc', self.data_input, self.label_input, self.flag_disc_train)
        self.disc_real = discriminator('disc', self.data_real_input, self.label_input, self.flag_disc_train)
        self.disc_fake = discriminator('disc', self.gen.outputs[1], self.label_input, self.flag_disc_train, True)

    def _init_output(self):
        #self.disc_raw_out = self.disc.outputs[0]
        #self.disc_sig_out = self.disc.outputs[1]
        self.gen_raw_out = self.gen.outputs[0]
        self.gen_sig_out = self.gen.outputs[1]
        self.disc_real_raw_out = self.disc_real.outputs[0]
        self.disc_real_sig_out = self.disc_real.outputs[1]
        self.disc_fake_raw_out = self.disc_fake.outputs[0]
        self.disc_fake_sig_out = self.disc_fake.outputs[1]

    def _init_loss(self):
        #self.loss_disc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=self.disc_raw_out, labels=self.label_input))
        self.loss_disc_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_real_raw_out, labels=tf.ones_like(self.disc_real_raw_out))
        self.loss_disc_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_fake_raw_out, labels=tf.zeros_like(self.disc_fake_raw_out))
        self.loss_disc = tf.reduce_mean(self.loss_disc_real + self.loss_disc_fake)

        self.loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_fake_raw_out, labels=tf.ones_like(self.disc_fake_raw_out)))

    def _init_trainable_vars(self):
        self.vars_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.disc_real.name)
        self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.gen.name)

    def _init_optimizer(self, opt_param):
        self.opt_disc = tf.train.AdamOptimizer(opt_param['learning_rate'], opt_param['beta1']).minimize(self.loss_disc, var_list=self.vars_disc)
        self.opt_gen = tf.train.AdamOptimizer(opt_param['learning_rate'], opt_param['beta1']).minimize(self.loss_gen, var_list=self.vars_gen)

    def _init_tb_summary(self, log_path):
        data_real = tf.summary.image('data_real_input', self.data_real_input)
        data_fake = tf.summary.image('data_fake_input', self.data_fake_input)
        loss_disc = tf.summary.scalar('loss_disc', self.loss_disc)
        self.summary_disc = tf.summary.merge([data_real, data_fake, loss_disc])

        data_fake = tf.summary.image('data_fake', self.gen_sig_out)
        loss_gen = tf.summary.scalar('loss_gen', self.loss_gen)
        self.summary_gen = tf.summary.merge([data_fake, loss_gen])

        pre_data_real = tf.summary.image('pre_data_real', self.data_real_input)
        pre_data_fake = tf.summary.image('pre_data_fake', self.data_fake_input)
        pre_loss_disc = tf.summary.scalar('loss_pre_disc', self.loss_disc)
        self.summary_pre_disc = tf.summary.merge([pre_data_real, pre_data_fake, pre_loss_disc])

        self.summary_writer = tf.summary.FileWriter(log_path, self.sess.graph)        
        
    def _pretrain_disc(self, batch_size, start_time):
        num_fetch = int(batch_size/2)
        num_iter = int(self.data_feeder.num_data // num_fetch)

        label = self._make_label(num_fetch, num_fetch)

        for i in xrange(0, num_iter):
            x_real = np.array(self.data_feeder.get(num_fetch))
            x_fake = np.random.uniform(size=x_real.shape)

            _, loss, summary_str = self.sess.run(
                [self.opt_disc, self.loss_disc, self.summary_pre_disc],
                feed_dict={self.data_real_input: x_real, 
                           self.data_fake_input: x_fake,
                           self.label_input: label,
                           self.flag_disc_train: True,
                           self.flag_gen_train: False})

            self.summary_writer.add_summary(summary_str, i)
            print("Pretrain: [%2d/%2d] time: %4.4f, loss_pre_d: %.8f" \
                % (i+1, num_iter, time.time() - start_time, loss))

    #def _pretrain_gen(self, batch_size, start_time):
    #    num_iter = int(self.data_feeder.num_data // batch_size)
        
    #    for i in xrange(0, num_iter):
    #        noise = self.noise_feeder.get(batch_size)
    #        _, loss, summary_str = self.sess.run
        


    def _make_label(self, num_real, num_fake):
        #label_real = np.ones([num_real, 1, 1, 1], np.float32)
        #label_fake = np.zeros([num_fake, 1, 1, 1], np.float32)
        #label = np.concatenate(label_real, label_fake)
        label = np.zeros([num_real+num_fake, 1, 1, 1], np.float32)
        label[:num_real] = 1.0

        return label