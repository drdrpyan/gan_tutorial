import tensorflow as tf

#class Net(object):
#    def __init__(self, name='', inputs=[], outputs=[]):
#        self.name = name
#        self.inputs = inputs
#        self.outputs = outputs

#class MNISTDisc(Net):
class MNISTDisc(object):
    def __init__(self, name, data_input, bn_flag_input=tf.placeholder(tf.bool), reuse=False):
        self._BN_EPS = 1e-5
        self._BN_MOMENTUM = 0.9

        #super().__init__(name)
        #Net.__init__(name)
        self.name = name
        self.inputs = []
        self.outputs = []

        self._check_data_input(data_input)
        self._check_bn_flag_input(bn_flag_input)

        self.data_input = data_input
        self.bn_flag_input = bn_flag_input
        self.inputs = [self.data_input, self.bn_flag_input]

        self.disc_conf = self._build(reuse)
        self.outputs.append(self.disc_conf)
        

    def _check_data_input(self, data_input):
        if data_input.shape[1:] != [28, 28, 1]:
            raise Exception('Illegal MNIST data')
        if data_input.dtype != tf.float32:
            raise Exception('data_input is not tf.float32 type')

    def _check_bn_flag_input(self, bn_flag_input):
        if bn_flag_input.dtype != tf.bool:
            raise Exception('bn_flag_input is not tf.bool')

    def _build(self, reuse):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            h0_0 = tf.layers.conv2d(self.data_input, 64, (5, 5), (2, 2), 'same')
            h0_1 = tf.layers.batch_normalization(
                h0_0, momentum=self._BN_MOMENTUM, epsilon=self._BN_EPS,
                training=self.bn_flag_input)
            h0_2 = tf.nn.leaky_relu(h0_1)

            h1_0 = tf.layers.conv2d(h0_2, 128, (5, 5), (2, 2), 'same')
            h1_1 = tf.layers.batch_normalization(
                h1_0, momentum=self._BN_MOMENTUM, epsilon=self._BN_EPS,
                training=self.bn_flag_input)
            h1_2 = tf.nn.leaky_relu(h1_1)

            kernel_size = (h1_2.shape[1], h1_2.shape[2])
            h2_0 = tf.layers.conv2d(h1_2, 1024, kernel_size, (1, 1), 'valid')
            h2_1 = tf.layers.batch_normalization(
                h2_0, momentum=self._BN_MOMENTUM, epsilon=self._BN_EPS,
                training=self.bn_flag_input)
            h2_2 = tf.nn.leaky_relu(h2_1)

            h3 = tf.layers.conv2d(h2_2, 1, (1, 1), (1, 1), 'valid')

            o = tf.nn.sigmoid(h3)

            #h2_0 = tf.layers.dense(h1_2, 1024)
            #h2_1 = tf.layers.batch_normalization(
            #    h2_0, momentum=self._BN_MOMENTUM, epsilon=self._BN_EPS,
            #    training=self.bn_flag_input)
            #h2_2 = tf.nn.leaky_relu(h2_1)

            #h3 = tf.layers.dense(h2_2, 1)

            #o = tf.nn.sigmoid(h3)

            #h2_0 = tf.layers.conv2d(h1_2, 256, (5, 5), (2, 2), 'same')
            #h2_1 = tf.layers.batch_normalization(
            #    h2_0, momentum=self._BN_MOMENTUM, epsilon=self._BN_EPS,
            #    training=self.bn_flag_input)
            #h2_2 = tf.nn.leaky_relu(h2_1)

            #h3_0 = tf.layers.conv2d(h2_2, 512, (5, 5), (2, 2), 'same')
            #h3_1 = tf.layers.batch_normalization(
            #    h3_0, momentum=self._BN_MOMENTUM, epsilon=self._BN_EPS,
            #    training=self.bn_flag_input)
            #h3_2 = tf.nn.leaky_relu(h3_1)

            #h4 = tf.layers.conv2d(h3_2, 1, (4, 4), padding='same')

            #o = tf.nn.sigmoid(h4)

        return o

#class MNISTGen(Net):
class MNISTGen(object):
    def __init__(self, name, noise_input, bn_flag_input=tf.placeholder(tf.bool)):
        self._BN_EPS = 1e-5
        self._BN_MOMENTUM = 0.9

        #super().__init__(name)
        #Net.__init__(name)
        self.name = name
        self.inputs = []
        self.outputs = []

        self.noise_input = noise_input
        self.bn_flag_input = bn_flag_input
        self.inputs = [self.noise_input, self.bn_flag_input]

        self.gen = self._build()
        self.outputs.append(self.gen)

    def _build(self):
        with tf.variable_scope(self.name) as scope:
            #num_elem = 1
            #for dim in self.noise_input.shape[1:]:
            #    num_elem *= int(dim)

            #h0_0 = tf.reshape(self.noise_input, [-1, 1, 1, num_elem])
            #h0_1 = tf.layers.conv2d(h0_0, 1024, (1, 1), (1, 1), 'valid')
            #h0_2 = tf.nn.relu(h0_1)

            #h0_1 = tf.layers.conv2d(self.noise_input, 1024, (1, 1), (1, 1), 'valid')
            #h0_2 = tf.nn.relu(h0_1)
            h0_0 = tf.layers.dense(self.noise_input, 1024)
            h0_1 = tf.nn.relu(h0_0)

            #h1_0 = tf.layers.conv2d(h0_2, 7*7*128, (1, 1), (1, 1), 'same')
            #h1_1 = tf.nn.relu(h1_0)
            h1_0 = tf.layers.dense(h0_1, 7*7*128)
            h1_1 = tf.nn.relu(h1_0)

            h2_0 = tf.reshape(h1_1, [-1, 7, 7, 128])
            h2_1 = tf.layers.conv2d_transpose(h2_0, 128, (5, 5), (2, 2), 'same')
            #h2_2 = tf.layers.batch_normalization(
            #    h2_1, momentum=self._BN_MOMENTUM, epsilon=self._BN_EPS,
            #    training=self.bn_flag_input)
            h2_2 = tf.layers.batch_normalization(
                h2_1, momentum=self._BN_MOMENTUM, epsilon=self._BN_EPS)
            h2_3 = tf.nn.relu(h2_2)

            h3 = tf.layers.conv2d_transpose(h2_3, 1, (5, 5), (2, 2), 'same')

            o = tf.nn.sigmoid(h3)

        return o
            

class MNISTDisc2(MNISTDisc):
    def _build(self, reuse):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            h0_0 = tf.layers.conv2d(self.data_input, 64, (5, 5), (2, 2), 'same')
            h0_1 = tf.nn.leaky_relu(h0_0)

            h1_0 = tf.layers.conv2d(h0_1, 128, (5, 5), (2, 2), 'same')
            h1_1 = tf.nn.leaky_relu(h1_0)

            kernel_size = (h1_1.shape[1], h1_1.shape[2])
            h2_0 = tf.layers.conv2d(h1_1, 1024, kernel_size, (1, 1), 'valid')
            h2_1 = tf.nn.leaky_relu(h2_0)

            h3 = tf.layers.conv2d(h2_1, 1, (1, 1), (1, 1), 'valid')

            o = tf.nn.sigmoid(h3)

            return o

class MNISTGen2(MNISTGen):
    def _build(self):
        with tf.variable_scope(self.name) as scope:
            h0_0 = tf.layers.dense(self.noise_input, 1024)
            h0_1 = tf.nn.relu(h0_0)

            h1_0 = tf.layers.dense(h0_1, 7*7*128)
            h1_1 = tf.nn.relu(h1_0)

            h2_0 = tf.reshape(h1_1, [-1, 7, 7, 128])
            h2_1 = tf.layers.conv2d_transpose(h2_0, 128, (5, 5), (2, 2), 'same')
            h2_2 = tf.nn.relu(h2_1)

            h3 = tf.layers.conv2d_transpose(h2_2, 1, (5, 5), (2, 2), 'same')

            o = tf.nn.sigmoid(h3)

            return o

class MNISTDisc3(object):
    def __init__(self, name, data_input, label_input, train_flag, reuse=False):
        self.name = name
        self.inputs = [data_input, label_input, train_flag]
        raw_o, sig_o = self._build(reuse)
        self.outputs = [raw_o, sig_o]

    def _build(self, reuse):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            h0_0 = tf.layers.conv2d(self.inputs[0], 64, (5, 5), (2, 2), 'same')
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

class MNISTGen3(object):
    def __init__(self, name, noise_input, train_flag):
        self.name = name
        self.inputs = [noise_input, train_flag]
        raw_o, sig_o = self._build()
        self.outputs = [raw_o, sig_o]

    def _build(self):
        with tf.variable_scope(self.name) as scope:
            h0_0 = tf.layers.dense(self.inputs[0], 1024)
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