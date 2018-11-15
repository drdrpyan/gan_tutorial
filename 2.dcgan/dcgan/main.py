import dcgan
import tensorflow as tf

def main():
    sess = tf.InteractiveSession()
    train_data_pass = 'D:/dataset/celebA'
    train_data_format = 'jpg'
    epoch = 20

    model = dcgan.DCGAN(sess, train_data_pass, train_data_format)
    model.train(epoch)    


if __name__ == '__main__':
    main()