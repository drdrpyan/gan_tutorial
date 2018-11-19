import tensorflow as tf

class DCGAN(object):
    def __init__(self, data_shape, noise_shape, 
                 discriminator, generator, name=None):
        self.name = name

        if self.name is not None:
            #with tf.variable_scope(self.name) as scope:
            #    self._init_DCGAN(data_shape, noise_shape,
            #                     discriminator, generator)
            self._init_DCGAN(data_shape, noise_shape, discriminator, generator)
        else:
            self._init_DCGAN(data_feeder.shape[0], noise_feeder.shape[0],
                             discriminator, generator)


    def _init_DCGAN(self, data_shape, noise_shape, discriminator, generator):
        self._build_graph(data_shape, noise_shape,
                          discriminator, generator)
        self._init_loss()
        self._init_trainable_vars()
        self._init_update_ops()


    def _build_graph(self, data_shape, noise_shape, 
                     discriminator, generator):
        self.noise = tf.placeholder(tf.float32, [64] + noise_shape)
        self.gen_train_flag = tf.placeholder(tf.bool)

        self.data_real = tf.placeholder(tf.float32, [64] + data_shape)
        self.disc_train_flag = tf.placeholder(tf.bool)

        self.gen = generator('gen', self.noise, self.gen_train_flag)
        self.disc_real = discriminator('disc', self.data_real, self.disc_train_flag)
        self.disc_fake = discriminator('disc', self.gen.outputs[0], self.disc_train_flag, True)


    def _init_loss(self):
        #self.loss_gen = tf.reduce_mean(
        #    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake.outputs[0], 
        #                                            labels=tf.ones_like(self.disc_fake.outputs[0])))
        #self.loss_disc_real = tf.reduce_mean(
        #    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_real.outputs[0], 
        #                                            labels=tf.ones_like(self.disc_real.outputs[0])))
        #self.loss_disc_fake = tf.reduce_mean(
        #    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake.outputs[0], 
        #                                            labels=tf.zeros_like(self.disc_fake.outputs[0])))     
        #self.loss_disc = self.loss_disc_real + self.loss_disc_fake
        EPS = 1e-2
        
        self.loss_gen = tf.reduce_mean(-tf.log(self.disc_fake.outputs[0] + EPS))
        self.loss_disc_real = -tf.log(self.disc_real.outputs[0] + EPS)
        self.loss_disc_fake = -tf.log(1 - self.disc_fake.outputs[0] + EPS)
        self.loss_disc = tf.reduce_mean(self.loss_disc_real + self.loss_disc_fake)

        #self.loss_disc = tf.reduce_mean(
        #    -tf.log(self.disc_real.outputs[0] + EPS) - tf.log(1 - self.disc_fake.outputs[0] + EPS)) 


    def _init_trainable_vars(self):
        #test1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #test5 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name + '/' + self.disc_real.name)

        #with tf.variable_scope(self.name):
        #    test2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #    test3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.disc_real.name)
        #    test4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.gen.name)
        #self.vars_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.disc_real.name)
        #self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.gen.name)

        #if not self.name:
        #    self.vars_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.disc_real.name)
        #    self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.gen.name)
        #else:
        #    self.vars_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
        #                                       self.name + '/' + self.disc_real.name)
        #    temp = tf.trainable_variables()
        #    self.vars_disc2 = [v for v in temp if 'disc' in v.name]
        #    self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
        #                                      self.name + '/' + self.gen.name)

        self.vars_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.disc_real.name)
        self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.gen.name)
                
    def _init_update_ops(self):
        #self.ops_disc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.disc_real.name)
        #self.ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.gen.name)
        if not self.name:
            self.ops_disc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.disc_real.name)
            self.ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.gen.name)
        else:
            self.ops_disc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.name + '/' + self.disc_real.name)
            self.ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.name + '/' + self.gen.name)
        self.ops_all = self.ops_disc + self.ops_gen