import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from abc import *
import os
import sys
  
        
class Sample(ABC):
    @abstractmethod
    def sample(self, num):
        raise NotImplementedError('define sample()')


class Gaussian1D(Sample):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    
    def sample(self, num, sort=False):        
        samples = np.random.normal(self.mu, self.sigma, num)
        if sort:
            samples.sort()
        return samples


class Noise(Sample):
    def __init__(self, noise_range):
        self.noise_range = noise_range
    

    def sample(self, num):
        offset = np.random.random(num) * (float(self.noise_range[1] - self.noise_range[0]) / num)
        samples = np.linspace(self.noise_range[0], self.noise_range[1], len(offset)) + offset
        return samples

class ResultPlot(object):
    def __init__(self, x_range, num_points, num_bins, mu, sigma):
        self.x_range = x_range

        self.num_points = num_points
        self.num_bins = num_bins

        self.mu = mu
        self.sigma = sigma

        self.xs = np.linspace(x_range[0], x_range[1], num_points)
        self.bins = np.linspace(x_range[0], x_range[1], num_bins)

    #def show_results(self, db_init, db_pre_trained, db_trained, p_d, p_g, save_img=True):
    #    #db_x = np.linspace(self.x_range[0], self.x_range[1], 100)
    #    #p_x = np.linspace(self.x_range[0], self.x_range[1], 100)
    #    #f, ax = plt.subplots(1)        
    #    #ax.plot(db_x, db_init, 'g--', linewidth=2, label='db_init')

    #    db_x = np.linspace(self.x_range[0], self.x_range[1], len(db_trained))
    #    p_x = np.linspace(self.x_range[0], self.x_range[1], len(p_d))
    #    f, ax = plt.subplot(1)
    #    ax.plot(db_x, db_init, 'g--', linewidth=2, label='db_init')
    #    ax.plot(db_x, db_pre_trained, 'c--', linewidth=2, label='db_pre_trained')
    #    ax.plot(db_x, db_trained, 'g-', linewidth=2, label='db_trained')
    #    ax.set_ylim(0, max(1, np.max(p_d) * 1.1))
    #    ax.set_xlim(max(self.mu - self.sigma * 3, self.x_range[0] * 0.9), 
    #                min(self.mu + self.sigma*3, self.x_range[1] * 0.9))
    #    plt.plot(p_x, p_d, 'b-', linewidth=2, label='real data')
    #    plt.plot(p_x, p_g, 'r-', linewidth=2, label='generated data')
    #    plt.title('1D Generative Adversarial Network: ' + '(mu : %3g. ' % self.mu + ' sigma : %3g)' % self.sigma)
    #    plt.xlabel('Data values')
    #    plt.ylabel('Probability density')
    #    plt.legend()
    #    plt.grid(True)

    #    if save_img:
    #        plt.savefig('GAN_1D_Gaussian' + '_mu_%g' % self.mu + '_sigma_%g' % self.sigma + '.png')

    #    plt.show()

    def show(self, db, real_data, gen_data, d_iter=None, g_iter=None, save_prefix=None, out_name=None):
        #p_real = self._data_to_pdf(real_data)
        #p_gen = self._data_to_pdf(gen_data)
        p_real, _ = np.histogram(real_data, bins=self.bins, density=True)
        p_gen, _ = np.histogram(gen_data, bins=self.bins, density=True)

        db_x = np.linspace(self.x_range[0], self.x_range[1], len(db))
        p_x = np.linspace(self.x_range[0], self.x_range[1], len(p_real))

        f, ax = plt.subplots(1)
        ax.set_ylim(0, max(max(1, np.max(p_real) * 1.1), np.max(p_gen) * 1.1))
        ax.set_xlim(max(self.mu - self.sigma * 3, self.x_range[0] * 0.9), min(self.mu + self.sigma * 3, self.x_range[1] * 0.9))

        plt.plot(db_x, db, 'g--', linewidth=2, label='decision boundary')

        
        plt.plot(p_x, p_real, 'b-', linewidth=2, label='real data')
        plt.plot(p_x, p_gen, 'r-', linewidth=2, label='generated data')
    
        plt.title('1D Generative Adversarial Network: ' + '(mu : %3g,' % self.mu + ' sigma : %3g)' % self.sigma)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.grid(True)

        #if out_name is None:
        #    plt.savefig(os.path.normpath(save_prefix + 'D%d_' %  d_iter + 'G%d.png' % g_iter))
        #else:
        #    plt.savefig(out_name)
        
        #plt.show()
        plt.pause(0.5)
        plt.show()
        
        


    def _data_to_pdf(self, data):
        pdf, _ = np.histogram(data, bins=self.bins, density=True)
        return pdf



class NetFactory(ABC):
    def __init__(self, default_args_dict=None):
        self.default_args_dict = default_args_dict


    def create(self, name, input, reuse_var=False, args_dict=None):
        with tf.variable_scope(name) as scope:
            if reuse_var:
                scope.reuse_variables()
    
            if args_dict is None:
                net_args = self.default_args_dict                
            else:
                net_args = args_dict
        
            net_out = self._create_net(input, net_args)
            net = Net(name, input, reuse_var, net_out)

        return net


    @abstractmethod
    def _create_net(self, input, args_dict):
        raise NotImplementedError('define _create_net()')


class Gaussian1DGeneraterFactory(NetFactory):
    def _create_net(self, input, args_dict):
        w_init = tf.truncated_normal_initializer(stddev=2)
        b_init = tf.constant_initializer(0.)
        
        w0 = tf.get_variable('w0', 
                             [input.get_shape()[1], args_dict['num_hidden']], 
                             initializer=w_init)
        b0 = tf.get_variable('b0', [args_dict['num_hidden']], 
                                initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(input, w0) + b0)

        w1 = tf.get_variable('w1', [h0.get_shape()[1], 1], initializer=w_init)
        b1 = tf.get_variable('b1', [1], initializer=b_init)
            
        o = tf.matmul(h0, w1) + b1

        return o

class Gaussian1DDiscriminatorFactory(NetFactory):
    def _create_net(self, input, args_dict):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        w0 = tf.get_variable('w0', 
                             [input.get_shape()[1], args_dict['num_hidden']], 
                             initializer=w_init)
        b0 = tf.get_variable('b0', [args_dict['num_hidden']], 
                                initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(input, w0) + b0)

        w1 = tf.get_variable('w1', [h0.get_shape()[1], 1], initializer=w_init)
        b1 = tf.get_variable('b1', [1], initializer=b_init)
            
        o =  tf.sigmoid(tf.matmul(h0, w1) + b1)

        return o


class Net(object):
    def __init__(self, name, input, shared=False, output=None):
        self.name = name
        self.input = input
        self.output = output
        self.shared = shared
    

class GANTrainer(object):
    def __init__(self, x_sampler, z_sampler, 
                 generator_factory, discriminator_factory,
                 batch_size=150, learning_rate=0.03, train_iters=3000):
        self._eps = 1e-2

        self.x_sampler = x_sampler
        self.z_sampler = z_sampler

        self.g_factory = generator_factory
        self.d_factory = discriminator_factory

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.G_z_input, self.G_z = self._init_G_z_net()        
        self.D_pre_input, self.D_pre_label, self.D_pre = self._init_D_pre_net()
        self.D_real_input, self.D_real, self.D_fake = self._init_D_net()

        #d_pre_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_pre.name)
        #d_real_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_real.name)
        #d_fake_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_fake.name)

        self.loss_g, self.opt_g = self._init_G_loss_opt()
        self.loss_d_pre, self.opt_d_pre = self._init_D_pre_loss_opt()
        self.loss_d, self.opt_d = self._init_D_loss_opt()

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        self.d_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_real.name))
        self.g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.G_z.name))

        #d_pre_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_pre.name)
        #d_real_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_real.name)
        #d_fake_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_fake.name)


    def get_decision_boundary(self, plot_xs):
        b = self.batch_size

        db = np.zeros((len(plot_xs), 1))
        for i in range(len(plot_xs) // b):
            db[b * i:b * (i+1)] = self.sess.run(self.D_real.output, 
                {self.D_real_input: np.reshape(plot_xs[b*i:b*(i+1)], (b, 1))})
            #db[b * i:b * (i+1)] = self.sess.run(self.D_pre.output, 
            #    {self.D_pre_input: np.reshape(plot_xs[b*i:b*(i+1)], (b, 1))})

        return db

    
    def pretrain_discriminator(self, num_iters, num_data=1000, num_bins=100):
        for iter in range(num_iters):
            print('pre-training : %d/%d' % (iter + 1, num_iters))

            sample = self.x_sampler.sample(num_data)
            histc, edges = np.histogram(sample, num_bins, density=True)

            max_histc = np.max(histc)
            min_histc = np.min(histc)
            labels = (histc - min_histc) / (max_histc - min_histc)
            x = edges[1:]

            self.sess.run([self.loss_d_pre, self.opt_d_pre],
                {self.D_pre_input: np.reshape(x, (num_bins, 1)), 
                    self.D_pre_label: np.reshape(labels, (num_bins, 1))})
        
        # store pre-trained variables
        d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_pre.name)
        d_weights = self.sess.run(d_pre_params)
        # copy weights from pre-training over to new D network
        d_real_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_real.name)
        tf.global_variables_initializer().run()
        for i, v in enumerate(d_real_params):
            self.sess.run(v.assign(d_weights[i]))
        
        
        print('pre-training finished!')

    
    def train_discriminator(self):
        #np.random.seed(np.random.randint(0, num_iters))
        x = self.x_sampler.sample(self.batch_size)
        z = self.z_sampler.sample(self.batch_size)

        loss_d, _ = self.sess.run([self.loss_d, self.opt_d], 
            {self.D_real_input: np.reshape(x, (self.batch_size, 1)), 
                self.G_z_input: np.reshape(z, (self.batch_size, 1))})

        return loss_d

        
    def train_generator(self):
        z = self.z_sampler.sample(self.batch_size)
        loss_g, _ = self.sess.run([self.loss_g, self.opt_g], 
            {self.G_z_input: np.reshape(z, (self.batch_size, 1))})

        return loss_g


    def pick_best_g_param(self, ckpt_list):
        num_sample = 1000
        num_bin = 100
        bins = np.linspace(self.z_sampler.noise_range[0], self.z_sampler.noise_range[1], num_bin)

        p_real, _ = np.histogram(self.get_real_data(num_sample), bins, density=True)
        
        best_idx = 0
        best_kl = sys.float_info.max

        for i in range(len(ckpt_list)):
            self.g_saver.restore(self.sess, ckpt_list[i])

            p_gen, _ = np.histogram(self.get_gen_data(num_sample), bins, density=True)

            kl = 0
            for j in range(len(p_real)):
                kl += (p_real[j] * np.log((p_real[j] + self._eps) / (p_gen[j] + self._eps)))
            if kl < best_kl:
                best_idx = i
                best_kl = kl

        return ckpt_list[best_idx]           


    def get_real_data(self, num):
        return self.x_sampler.sample(num)

    def get_gen_data(self, num):
        zs = np.linspace(self.z_sampler.noise_range[0], self.z_sampler.noise_range[1], num)
        gen = np.zeros((num, 1))
        for i in range(num // self.batch_size):
            gen[self.batch_size * i:self.batch_size * (i+1)] = self.sess.run(self.G_z.output, 
                {self.G_z_input: np.reshape(zs[self.batch_size*i:self.batch_size*(i+1)], (self.batch_size, 1))})
        
        return gen      


    def save_generator(self, path, step=None):
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.G_z.name))
        saved_path = saver.save(self.sess, path, step)
        return saved_path
        

    def save_discriminator(self, path):
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.D_real.name))
        saver.save(self.sess, path)

    def load_generator(self, path):
        pass

        
        

    def _init_G_z_net(self):
        G_z_input = tf.placeholder(tf.float32, shape=(None, 1))
        G_z = self.g_factory.create('G-z', G_z_input)

        return G_z_input, G_z


    def _init_D_pre_net(self):
        D_pre_input = tf.placeholder(tf.float32, shape=(None, 1))
        D_pre_label = tf.placeholder(tf.float32, shape=(None, 1))
        D_pre = self.d_factory.create('D-pre', D_pre_input)

        return D_pre_input, D_pre_label, D_pre


    def _init_D_net(self):
        D_real_input = tf.placeholder(tf.float32, shape=(None, 1))
        D_real = self.d_factory.create('D-x', D_real_input)
        D_fake = self.d_factory.create('D-x', self.G_z.output, True)
        
        return D_real_input, D_real, D_fake

    
    def _init_G_loss_opt(self):
        loss_g = tf.reduce_mean(-tf.log(self.D_fake.output + self._eps))
        params_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                     scope=self.G_z.name)
        opt_g = self._get_default_optimizer(
            loss_g, params_g, initial_learning_rate=self.learning_rate/2)

        return loss_g, opt_g


    def _init_D_pre_loss_opt(self):
        loss_d_pre = tf.reduce_mean(tf.square(self.D_pre.output - self.D_pre_label))
        params_d_pre = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.D_pre.name)
        opt_d_pre = self._get_default_optimizer(
            loss_d_pre, params_d_pre, initial_learning_rate=self.learning_rate)

        return loss_d_pre, opt_d_pre

    def _init_D_loss_opt(self):
        loss_d = tf.reduce_mean(
            -tf.log(self.D_real.output + self._eps) - tf.log(1-self.D_fake.output + self._eps))
        params_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.D_real.name)
        opt_d = self._get_default_optimizer(
            loss_d, params_d, initial_learning_rate=self.learning_rate)

        return loss_d, opt_d


    def _get_default_optimizer(self, loss, var_list, 
                               decay=0.95, num_decay_steps=400,
                               initial_learning_rate=0.03):
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, batch, num_decay_steps, decay,
            staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=batch, var_list=var_list)

        return optimizer


        
def train_gaussian_1D_GAN():
    mu = 0
    sigma = 1

    z_range = [-5, 5]

    batch_size = 150
    learning_rate = 0.03
    train_iters = 100

    num_hidden = 32

    num_train_step = 20000
    
    num_g_train_iter = 100
    g_train_checkpt = 10

    num_d_train_iter = 100

    num_d_pretrain_ter = 1000


    g_ckpt_prefix = './ckpt/G'
    d_ckpt_prefix = './ckpt/D'
    result_path = './result'

    x_sampler = Gaussian1D(mu, sigma)
    z_sampler = Noise(z_range)

    g_factory = Gaussian1DGeneraterFactory({'num_hidden':num_hidden})
    d_factory = Gaussian1DDiscriminatorFactory({'num_hidden':num_hidden})

    gan_trainer = GANTrainer(x_sampler, z_sampler, 
                             g_factory, d_factory, 
                             batch_size, learning_rate, train_iters)

    plot = ResultPlot(z_range, 10000, 20, mu, sigma)
    
    # initial state
    db_init = gan_trainer.get_decision_boundary(plot.xs)
    data_real = gan_trainer.get_real_data(plot.num_points)
    data_gen = gan_trainer.get_gen_data(plot.num_points)
    plot.show(db_init, data_real, data_gen, out_name=os.path.join(result_path, 'initial_state.png'))

    # pretrain discriminator
    gan_trainer.pretrain_discriminator(num_d_pretrain_ter)
    db_pretrain = gan_trainer.get_decision_boundary(plot.xs)
    plot.show(db_pretrain, data_real, data_gen, out_name=os.path.join(result_path, 'pretrained_d.png'))

    # train
    for s in range(num_train_step):
        #g_ckpt_list = []
        #for i_g in range(num_g_train_iter):
        #    gan_trainer.train_generator()
            
        #    if i_g % g_train_checkpt == 0:
        #        g_ckpt_path = gan_trainer.save_generator(g_ckpt_prefix, s*num_g_train_iter + i_g)
        #        g_ckpt_list.append(g_ckpt_path)        
        
        #gan_trainer.pick_best_g_param(g_ckpt_list)

        #for i_d in range(num_d_train_iter):
        #    gan_trainer.train_discriminator()
        print('training : %d/%d' % (s+ 1, num_train_step))
        gan_trainer.train_generator()
        gan_trainer.train_discriminator()

    data_real = gan_trainer.get_real_data(plot.num_points)
    data_gen = gan_trainer.get_gen_data(plot.num_points)
    db = gan_trainer.get_decision_boundary(plot.xs)
    plot.show(db, data_real, data_gen, out_name='./result.png')
        

def net_test(x, n_hidden=32):

    # initializers
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
    b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # output layer
    w1 = tf.get_variable('w1', [h0.get_shape()[1], 1], initializer=w_init)
    b1 = tf.get_variable('b1', [1], initializer=b_init)
    o = tf.sigmoid(tf.matmul(h0, w1) + b1)

    return o

def main():
    #with tf.variable_scope('f') as scope:
    #    x0 = tf.placeholder(tf.float32, shape=(None, 1))
    #    n0 = net_test(x0)

    #with tf.variable_scope('t') as scope:
    #    x1 = tf.placeholder(tf.float32, shape=(None, 1))
    #    #x2 = tf.placeholder(tf.float32, shape=(None, 1))
    #    n1 = net_test(x1)
    #    #scope.reuse_variables()
    #    #n2 = net_test(x2)

    #with tf.variable_scope('t', reuse=True) as scope:
    #    x3 = tf.placeholder(tf.float32, shape=(None, 1))
    #    n3 = net_test(x3)

    #n0_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'f')
    #n1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 't')
    #n3_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 't_1')
    
    #sess = tf.InteractiveSession()
    #tf.global_variables_initializer().run()

    #n0_weights = sess.run(n0_params)
    #n1_weights = sess.run(n1_params)

    #n1_enum = enumerate(n1_params)


    train_gaussian_1D_GAN()


if __name__ == '__main__':
    main()