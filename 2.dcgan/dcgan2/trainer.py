import tensorflow as tf

import time
#import abc

#class Trainer(abc.ABC):
#    @abc.abstractclassmethod
#    def train(self):
#        raise Exception("Implement 'train(self)'")

#class DCGANTrainer(Trainer):
#    def __init__(self, data_feeder, noise_feeder, 
#                 generator, discriminator):
#        self._init_graph(generator, discriminator)

#    def _init_loss(self):
#        pass

#    def _init_graph(self, data_feeder, noise_feeder, 
#                    generator, discriminator):
#        disc_real_input = tf.placeholder(tf.float32, [None] + data_feeder.shape[0])
#        gen_noise_input = tf.placeholder(tf.float32, [None] + noise_feeder.shape[0])
        
#        disc_real = discriminator('disc_real', disc_real_input, 
        
                 
class DCGANTrainer(object):
    def __init__(self, sess, dcgan, data_feeder, noise_feeder, 
                 epoch, learning_rate, log_path='./logs/'):
        self.sess = sess
        self.dcgan = dcgan

        self._init_opt(dcgan, learning_rate, beta1)

        self.data_feeder = data_feeder
        self.noise_feeder = noise_feeder

    def _init_opt(self, learning_rate, beta1):
        with tf.control_dependencies(dcgan.ops_all):
            self.g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_gen, var_list=self.vars_gen)
            self.d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_disc, var_list=self.vars_disc)

    def _init_tb_summary(self, log_path):
        data_real = tf.summary.image('data_real', self.dcgan.disc_real.inputs[0])
        loss_disc = tf.summary.scalar('loss_disc', self.dcgan.loss_disc)
        self.summary_disc = tf.summary.merge([data_real, loss_disc])

        gen = tf.summary.image('gen', self.dcgan.gen.outputs[0])
        loss_gen = tf.summary.scalar('loss_gen', self.dcgan.loss_gen)
        self.summary_disc = tf.summary.merge([gen, loss_gen])

        self.summary_writer = tf.train.SummaryWriter(log_path, self.sess.graph)

    def train(self, batch_size, epoch):
        tf.global_variables_initializer().run()
    
        start_time = time.time()
        iter = 1
        batch_per_epoch = int(self.data_feeder.num_data // batch_size)

        for e in xrange(epoch):
            for b in xrange(0, batch_per_epoch):
                x = self.data_feeder.get(batch_size)
                z = self.noise_feeder.get(batch_size)

                _, summary_str = self.sess.run([self.d_opt, self.summary_disc],
                    feed_dict={self.dcgan.disc.inputs[0]: x, self.dcgan.disc.inputs[1]: True,
                               self.dcgan.gen.inputs[0]: z, self.dcgan.disc.inputs[1]: False})

    def _train_disc(self, data_real, noise):
        _, summary_str = self.sess.run([self.d_opt, self.summary_disc],
                    feed_dict={self.dcgan.disc.inputs[0]: data_real, self.dcgan.disc.inputs[1]: True,
                               self.dcgan.gen.inputs[0]: noise, self.dcgan.disc.inputs[1]: False})
        self.summary_writer.add_summary(summary_str, 
    

