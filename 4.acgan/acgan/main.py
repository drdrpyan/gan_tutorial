import data_feeder
import acgan_trainer

import tensorflow as tf

import os

acgan_config = {'data_path': os.path.normpath('D:/dataset/mnist/train-images.idx3-ubyte'),
               'label_path': os.path.normpath('D:/dataset/mnist/train-labels.idx1-ubyte'),
               'noise_shape': [1, 1, 100],
               'log_path': './logs',
               'sample_path': './samples',
               'batch_size': 50,
               'epoch': 100,
               'learning_rate': 0.0001,
               'gen_lr_mult': 5,
               'class_lr_mult': 5,
               'num_sample_set': 5}

def main():
    d_feeder = data_feeder.MNISTFeeder(acgan_config['data_path'],
                                       acgan_config['label_path'])
    n_feeder = data_feeder.NoiseFeeder(acgan_config['noise_shape'])

    opt_param = {'learning_rate': acgan_config['learning_rate'],
                 'gen_lr_mult': acgan_config['gen_lr_mult'],
                 'class_lr_mult': acgan_config['class_lr_mult']}

    sess = tf.InteractiveSession()

    trainer = acgan_trainer.ACGANTrainer(
        sess, d_feeder, n_feeder, opt_param,
        acgan_config['batch_size'], acgan_config['epoch'],
        acgan_config['log_path'],
        acgan_config['num_sample_set'], acgan_config['sample_path'])

    trainer.train()


    pass

if __name__ == '__main__':
    main()