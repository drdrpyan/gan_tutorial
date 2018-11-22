import data_feeder
import trainer4
import net
import dcgan2

import tensorflow as tf

import os

#mnist_config = {'data_shape':[28, 28, 1], 
#                'noise_shape':[1, 1, 100],
#                'data_path':os.path.normpath('D:/dataset/mnist/train-images.idx3-ubyte'),
#                'epoch':100,
#                'learning_rate':0.01,
#                'beta1':0.9,
#                'log_path':'./logs/',
#                'batch_size':64}

#def main():
#    d_feeder = data_feeder.MNISTFeeder(mnist_config['data_path'])
#    n_feeder = data_feeder.NoiseFeeder(mnist_config['noise_shape'])
    
#    discriminator = net.MNISTDisc2
#    generator = net.MNISTGen2

#    dcgan = dcgan2.DCGAN(mnist_config['data_shape'],
#                         mnist_config['noise_shape'],
#                         discriminator, generator, 'mnist-dcgan')
#    #dcgan = dcgan2.DCGAN(mnist_config['data_shape'],
#    #                     mnist_config['noise_shape'],
#    #                     discriminator, generator, None)
    
#    sess = tf.InteractiveSession()
    
#    dcgan_trainer = trainer.DCGANTrainer(sess, dcgan, d_feeder, n_feeder,
#                                         mnist_config['log_path'])

#    dcgan_trainer.train(mnist_config['batch_size'], 
#                        mnist_config['epoch'],
#                        mnist_config['learning_rate'],
#                        mnist_config['beta1'])

mnist_config = {'data_path': os.path.normpath('D:/dataset/mnist/train-images.idx3-ubyte'),
                'noise_shape': [1, 1, 100],
                'log_path': './logs',
                'sample_path': './samples',
                'batch_size': 64,
                'epoch': 100,
                'learning_rate': 0.00001,
                'beta1': 0.9}
#def main():
#    d_feeder = data_feeder.MNISTFeeder(mnist_config['data_path'])
#    n_feeder = data_feeder.NoiseFeeder(mnist_config['noise_shape'])

#    discriminator = net.MNISTDisc3
#    generator = net.MNISTGen3

#    sess = tf.InteractiveSession()

#    dcgan_trainer = trainer3.DCGANTrainer3(sess, discriminator, generator,
#                                            d_feeder, n_feeder, mnist_config['log_path'])
#    adam_opt_param = {'learning_rate': mnist_config['learning_rate'],
#                      'beta1': mnist_config['beta1']}
#    dcgan_trainer.train(mnist_config['batch_size'], 
#                        mnist_config['epoch'], adam_opt_param)

def main():
    d_feeder = data_feeder.MNISTFeeder(mnist_config['data_path'])
    n_feeder = data_feeder.NoiseFeeder(mnist_config['noise_shape'])

    sess = tf.InteractiveSession()

    adam_opt_param = {'learning_rate': mnist_config['learning_rate'],
                      'beta1': mnist_config['beta1']}
    dcgan_trainer = trainer4.DCGANTrainer4(sess, d_feeder, n_feeder, 
                                           mnist_config['batch_size'],
                                           mnist_config['epoch'],
                                           adam_opt_param,
                                           mnist_config['log_path'],
                                           mnist_config['sample_path'])
    dcgan_trainer.train()
                                           

if __name__ == '__main__':
    main()