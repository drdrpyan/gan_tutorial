import tensorflow as tf
from six.moves import xrange
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
                 log_path='./logs/'):
        self.sess = sess
        self.dcgan = dcgan

        #self._init_opt(dcgan, learning_rate, beta1)

        self.data_feeder = data_feeder
        self.noise_feeder = noise_feeder

        self._init_tb_summary(log_path)

    def _init_opt(self, learning_rate, beta1):
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #    self.g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_gen, var_list=self.dcgan.vars_gen)
        #    self.d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_disc, var_list=self.dcgan.vars_disc)
            
        self.g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_gen, var_list=self.dcgan.vars_gen)
        self.d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_disc, var_list=self.dcgan.vars_disc)

        #with tf.control_dependencies(self.dcgan.ops_gen):
        #    self.g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_gen, var_list=self.dcgan.vars_gen)
        #with tf.control_dependencies(self.dcgan.ops_disc):
        #    self.d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.dcgan.loss_disc, var_list=self.dcgan.vars_disc)

    def _init_tb_summary(self, log_path):
        data_real = tf.summary.image('data_real', self.dcgan.disc_real.inputs[0])
        loss_disc = tf.summary.scalar('loss_disc', self.dcgan.loss_disc)
        disc_real_result = tf.summary.histogram('disc_real_result', self.dcgan.disc_real.outputs[0])
        disc_fake_result = tf.summary.histogram('disc_fake_result', self.dcgan.disc_fake.outputs[0])
        self.summary_disc = tf.summary.merge([data_real, loss_disc, disc_real_result, disc_fake_result])

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

                self._train_disc(x, z, iter)
                self._train_gen(x, z, iter)

                #err_disc_real = self.dcgan.loss_disc_real.eval({self.dcgan.disc_real.inputs[0]: x, 
                #                                                self.dcgan.disc_real.inputs[1]: False})
                #err_disc_fake = self.dcgan.loss_disc_fake.eval({self.dcgan.gen.inputs[0]: z, 
                #                                                self.dcgan.gen.inputs[1]: False,
                #                                                self.dcgan.disc_fake.inputs[1]: False})
                err_disc = self.dcgan.loss_disc.eval({self.dcgan.disc_real.inputs[0]: x, 
                                                      self.dcgan.gen.inputs[0]: z, 
                                                      self.dcgan.gen.inputs[1]: False,
                                                      self.dcgan.disc_fake.inputs[1]: False})

                err_gen = self.dcgan.loss_gen.eval({self.dcgan.gen.inputs[0]: z, 
                                                    self.dcgan.gen.inputs[1]: False,
                                                    self.dcgan.disc_fake.inputs[1]: False})
                #print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss_d_real: %.8f, loss_d_fake: %.8f, loss_d: %.8f, loss_g: %.8f" \
                #    % (e+1, epoch, b+1, batch_per_epoch, time.time() - start_time, err_disc_real, err_disc_fake, err_disc_real+err_disc_fake, err_gen))
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss_d: %.8f, loss_g: %.8f" \
                    % (e+1, epoch, b+1, batch_per_epoch, time.time() - start_time, err_disc, err_gen))

                iter += 1

    def _train_disc(self, data_real, noise, iter):
        #_, summary_str = self.sess.run([self.d_opt, self.summary_disc],
        #    feed_dict={self.dcgan.disc_real.inputs[0]: data_real, self.dcgan.disc_real.inputs[1]: True,
        #               self.dcgan.gen.inputs[0]: noise, self.dcgan.gen.inputs[1]: False})
        _, _, summary_str = self.sess.run([self.d_opt, self.dcgan.loss_disc, self.summary_disc],
            feed_dict={self.dcgan.disc_real.inputs[0]: data_real,
                       self.dcgan.gen.inputs[0]: noise})
        self.summary_writer.add_summary(summary_str, iter)

    def _train_gen(self, data_real, noise, iter):
        #_, summary_str = self.sess.run([self.g_opt, self.summary_gen],
        #    feed_dict={self.dcgan.disc_real.inputs[0]: data_real, self.dcgan.disc_real.inputs[1]: False,
        #               self.dcgan.gen.inputs[0]: noise, self.dcgan.gen.inputs[1]: True})
        _, _, summary_str = self.sess.run([self.g_opt, self.dcgan.loss_gen, self.summary_gen],
            feed_dict={self.dcgan.gen.inputs[0]: noise, self.dcgan.disc_real.inputs[0]: data_real})
        self.summary_writer.add_summary(summary_str, iter)

    

