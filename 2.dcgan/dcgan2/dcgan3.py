import tensorflow as tf

class DCGAN2(object):
    def __init__(self, name, data_shape, 
                 noise_shape, discriminator, generator):
        self.name = name

        if self.name is not None:
            with tf.variable_scope(self.name) as scope:
                self._build_graph(data_shape, noise_shape, discriminator, generator)
        else:
            self._build_graph(data_shape, noise_shape, discriminator, generator)

        self._init_loss()
        self._init_trainable_vars()
        self._init_update_ops()

        
    def _build_graph(self, data_shape, noise_shape, 
                     discriminator, generator):
        self.noise = tf.placeholder(tf.float32, [None] + noise_shape)
        self.gen_train_flag = tf.placeholder(tf.bool)

        self.data = tf.placeholder(tf.float32, [None] + data_shape)
        self.label = tf.placeholder(tf.float32, [None] + [1, 1, 1])
        self.disc_train_flag = tf.placeholder(tf.bool)

        self.gen = generator('gen', self.noise, self.gen_train_flag)
        self.disc = discriminator('disc', self.data, self.label, self.disc_train_flag)

        self.gen_raw_out = self.gen.outputs[0]
        self.disc_raw_out = self.disc.outputs[0]

    def _init_loss(self):
        self.loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc.raw_out, labels=tf.ones_like(self.disc_raw_out)))
        self.loss_disc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_raw_out, labels=self.label))

    def _init_trainable_vars(self):
        if not self.name:
            self.vars_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.disc.name)
            self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.gen.name)
        else:
            self.vars_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                               self.name + '/' + self.disc.name)
            self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              self.name + '/' + self.gen.name)

    def _init_update_ops(self):
        if not self.name:
            self.ops_disc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.disc_real.name)
            self.ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.gen.name)
        else:
            self.ops_disc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.name + '/' + self.disc_real.name)
            self.ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.name + '/' + self.gen.name)
        self.ops_all = self.ops_disc + self.ops_gen