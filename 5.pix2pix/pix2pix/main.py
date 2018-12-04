import data_feeder
import pix2pix_trainer

import tensorflow as tf

import os

pix2pix_config = {'data_path': os.path.normpath('D:/dataset/celebA/image'),
                  'label_path': os.path.normpath('D:/dataset/celebA/list_bbox_celeba.csv'),
                  'img_size': [256, 256],
                  'landmark_dat': './shape_predictor_68_face_landmarks.dat',
                  'log_path': './logs',
                  'sample_path': './samples',
                  'batch_size': 16,
                  'epoch': 100,
                  'learning_rate': 0.0001,
                  'gen_lr_mult': 5}

def main():
    d_feeder = data_feeder.CelebABboxFeeder(pix2pix_config['data_path'],
                                            pix2pix_config['label_path'], 0.1)

    opt_param = {'learning_rate': pix2pix_config['learning_rate'],
                 'gen_lr_mult': pix2pix_config['gen_lr_mult']}

    sess = tf.InteractiveSession()

    trainer = pix2pix_trainer.Pix2PixTrainer(
        sess, d_feeder, pix2pix_config['img_size'], 
        pix2pix_config['landmark_dat'], opt_param,
        pix2pix_config['batch_size'], pix2pix_config['epoch'],
        pix2pix_config['log_path'], pix2pix_config['sample_path'])

    trainer.train()


    pass

if __name__ == '__main__':
    main()