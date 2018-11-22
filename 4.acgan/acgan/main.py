import data_feeder
import cgan_trainer

import tensorflow as tf

import os

cgan_config = {'data_path': os.path.normpath('D:/dataset/mnist/train-images.idx3-ubyte'),
               'label_path': os.path.normpath('D:/dataset/mnist/train-labels.idx1-ubyte'),
               'noise_shape': [1, 1, 100],
               'log_path': './logs',
               'sample_path': './samples',
               'batch_size': 50,
               'epoch': 100,
               'learning_rate': 0.0001,
               'gen_lr_mult': 5,
               'num_sample_set': 5}

def main():
    d_feeder = data_feeder.MNISTFeeder(cgan_config['data_path'],
                                       cgan_config['label_path'])
    n_feeder = data_feeder.NoiseFeeder(cgan_config['noise_shape'])

    opt_param = {'learning_rate': cgan_config['learning_rate'],
                 'gen_lr_mult': cgan_config['gen_lr_mult']}

    sess = tf.InteractiveSession()

    trainer = cgan_trainer.CGANTrainer(
        sess, d_feeder, n_feeder, opt_param,
        cgan_config['batch_size'], cgan_config['epoch'],
        cgan_config['log_path'],
        cgan_config['num_sample_set'], cgan_config['sample_path'])

    trainer.train()


    pass

if __name__ == '__main__':
    main()