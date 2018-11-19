import tensorflow as tf
from six.moves import xrange
import data_feeder
import numpy as np
import os

def gen(name, input):
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

        return h3, o

def disc(name, input, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        h0_0 = tf.layers.conv2d(input, 64, (5, 5), (2, 2), 'same')
        h0_1 = tf.nn.leaky_relu(h0_0)

        h1_0 = tf.layers.conv2d(h0_1, 128, (5, 5), (2, 2), 'same')
        h1_1 = tf.nn.leaky_relu(h1_0)

        kernel_size = (h1_1.shape[1], h1_1.shape[2])
        h2_0 = tf.layers.conv2d(h1_1, 1024, kernel_size, (1, 1), 'valid')
        h2_1 = tf.nn.leaky_relu(h2_0)

        h3 = tf.layers.conv2d(h2_1, 1, (1, 1), (1, 1), 'valid')

        o = tf.nn.sigmoid(h3)
    return h3, o


def main():
    mnist_feeder = data_feeder.MNISTFeeder(os.path.normpath('D:/dataset/mnist/train-images.idx3-ubyte'))

    img_input = tf.placeholder(tf.float32, [64, 28, 28, 1])
    label_input = tf.placeholder(tf.float32, [64, 1, 1, 1])
    d_real_out1, d_real_out2 = disc('disc', img_input)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out1, labels=label_input))
    train_vars= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc')
    opt = tf.train.AdamOptimizer().minimize(loss, var_list=train_vars)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    data_real = mnist_feeder.get(32)
    data_real = np.reshape(data_real, [32, 28, 28, 1])
    data_fake = np.random.uniform(0, 1, size=(32, 28, 28, 1))

    dr_summary = tf.summary.image('data_real', img_input)
    #df_summary = tf.summary.image('data_fake', data_fake)
    loss_summary = tf.summary.scalar('loss', loss)
    summary_merge = tf.summary.merge([dr_summary, loss_summary])
    summary_writer = tf.summary.FileWriter('./logs/', sess.graph)

    for i in xrange(10000):
        data_real = np.array(mnist_feeder.get(32))
        data_real = np.reshape(data_real, [32, 28, 28, 1])
        data_fake = np.random.uniform(0, 1, size=(32, 28, 28, 1))
        data = np.zeros([64, 28, 28, 1])
        data[0:32] = data_real
        data[32:] = data_fake

        #label_real = np.ones((32, 1, 1, 1))
        #label_fake = np.zeros((32, 1, 1, 1))
        label = np.ones((64, 1, 1, 1))
        label[32:] = np.zeros((32, 1, 1, 1))
    
        _, loss_result, summary_str = sess.run([opt, loss, summary_merge],
                                               feed_dict={img_input:data, label_input:label})

        print('iter:%d, loss:%f' % (i, loss_result))

        
        


if __name__ == '__main__':
    main()