import tensorflow as tf
import tensorboard as tb

class Summary(object):
    def __init__(self):
        try:
            image_summary = tf.image_summary
            scalar_summary = tf.scalar_summary
            histogram_summary = tf.histogram_summary
            merge_summary = tf.merge_summary
            SummaryWriter = tf.train.SummaryWriter
        except:
            image_summary = tf.summary.image
            scalar_summary = tf.summary.scalar
            histogram_summary = tf.summary.histogram
            merge_summary = tf.summary.merge
            SummaryWriter = tf.summary.FileWriter

class DCGAN(object):
    def __init__(self, batch_size=64, z_dim=100):
        self.img_shape = [64, 64, 3]

        self.batch_size = batch_size
        self.z_dim = z_dim

    def train(self, learning_rate=0.01, beta1=0.9):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.loss_G, var_list=self.vars_G)
        d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.loss_D, var_list=self.vars_D)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = tf.merge_s

    def _init_summary(self, summary):
        self.summary_z = summary.histogram_summary('z', self.z)
        
        self.summary_d_real = summary.histogram_summary('D_real', self.D_real)
        self.summary_d_fake = summary.histogram_summary('D_fake', self.D_fake)

        self.summary_g = summary.image_summary('G', self.G)

        self.summary_loss_d_real = summary.scalar_summary('loss_d_real', self.loss_D_real)
        self.summary_loss_d_fake = summary.scalar_summary('loss_d_fake', self.loss_D_fake)
        self.summary_loss_d = summary.scalar_summary('loss_d', self.loss_D)
        
        self.summary_loss_g = summary.scalar_summary('loss_g', self.loss_G)

        self.g_sum
        


    def _build_model(self):
        self.input = tf.placeholder(tf.float32, [self.batch_size] + self.img_shape, name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        
        self.G = self._build_generator(self.z)

        self.D_real = self._build_discriminator(self.input)
        self.D_fake = self._build_discriminator(self.G, reuse_var=True)

        self.loss_G = tf.reduce_mean(
            _sigmoid_cross_entropy_with_logits(self.D_real, tf.ones_like(self.D_real)))
        self.loss_D_real = tf.reduce_mean(
            _sigmoid_cross_entropy_with_logits(self.D_real, tf.ones_like(self.D_real)))
        self.loss_D_fake = tf.reduce_mean(
            _sigmoid_cross_entropy_with_logits(self.D_fake, tf.zeros_like(self.D_fake)))
        self.loss_D = self.loss_D_real + self.loss_D_fake

        trainable_vars = tf.trainable_variables()
        self.vars_G = [var for var in trainable_vars if 'generator' in var.name]
        self.vars_D = [var for var in trainable_vars if 'discriminator' in var.name]

        self.saver = tf.train.Saver()


    def _build_generator(self, input, name='generator'):
        with tf.variable_scope(name) as scope:
            h0_0 = tf.layers.conv2d_transpose(input, 1024, [4, 4], [4, 4], 'same', name='deconv0')
            h0_1 = tf.layers.batch_normalization(h0_0, name='deconv0_bn')
            h0_2 = tf.nn.relu(h0_1, name='deconv0_relu')

            h1_0 = tf.layers.conv2d_transpose(h0_1, 512, [4, 4], [2, 2], 'same', name='deconv1')
            h1_1 = tf.layers.batch_normalization(h1_0, name='deconv1_bn')
            h1_2 = tf.nn.relu(h1_1, name='deconv1_relu')

            h2_0 = tf.layers.conv2d_transpose(h1_1, 256, [4, 4], [2, 2], 'same', name='deconv2')
            h2_1 = tf.layers.batch_normalization(h2_0, name='deconv2_bn')
            h2_2 = tf.nn.relu(h2_1, name='deconv2_relu')

            h3_0 = tf.layers.conv2d_transpose(h2_2, 128, [4, 4], [2, 2], 'same', name='deconv3')
            h3_1 = tf.layers.batch_normalization(h3_0, name='deconv3_bn')
            h3_2 = tf.nn.relu(h3_1, name='deconv3_relu')

            h4 = tf.layers.conv2d_transpose(h3_2, 3, [4, 4], [2, 2], 'same', name='deconv4')

            o = tf.nn.sigmoid(h4) * 255

        return o


    def _build_discriminator(self, input, name='discriminator', reuse_var=False):
        with tf.variable_scope(name) as scope:
            if reuse_var:
                scope.reuse_variables()
    
            h0_0 = tf.layers.conv2d(input, 128, [4, 4], (2, 2), 'same', name='conv0')
            h0_1 = tf.layers.batch_normalization(h0_0, name='conv0_bn')
            h0_2 = tf.nn.leaky_relu(h0_1, name='conv0_lrelu')
            
            h1_0 = tf.layers.conv2d(h0_2, 256, [4, 4], (2, 2), 'same', name='conv1')
            h1_1 = tf.layers.batch_normalization(h1_0, name='conv1_bn')
            h1_2 = tf.nn.leaky_relu(h1_1, name='conv1_lrelu')

            h2_0 = tf.layers.conv2d(h1_2, 512, [4, 4], (2, 2), 'same', name='conv2')
            h2_1 = tf.layers.batch_normalization(h2_0, name='conv2_bn')
            h2_2 = tf.nn.leaky_relu(h2_1, name='conv2_lrelu')

            h3_0 = tf.layers.conv2d(h2_2, 1024, [4, 4], (2, 2), 'same', name='conv3')
            h3_1 = tf.layers.batch_normalization(h3_0, name='conv3_bn')
            h3_2 = tf.nn.leaky_relu(h3_1, name='conv3_lrelu')

            h4 = tf.reduce_mean(h3_2, axis=[1, 2], name='avg_pool')

            o = tf.sigmoid(h4, name='sig_out')

        return o


    def _sigmoid_cross_entropy_with_logits(x, y):
        try:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        except:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
