import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from abc import *

##class Gaussian1D(object):
##    def __init__(self, mu, sigma):
##        self.mu = mu
##        self.sigma = sigma

##    def sample(self, num, sort=False):
##        samples = np.random.normal(self.mu, self.sigma, num)
##        if sort:
##            samples.sort()
##        return samples

##class NoiseDist(object):
##    def __init__(self, range):
##        self.range = range

##    # ����

##class Generator(object):
##    def __init__(self, z, num_hidden=32, name='G', optimzer=None):
##        self.name = name

##        w_init = tf.truncated_normal_initializer(stddev=2)
##        b_init = tf.constant_initializer(0.)

##        with tf.variable_scope(name):
##            w0 = tf.get_variable('w0', [z.get_shape()[1], num_hidden], initializer=w_init)
##            b0 = tf.get_variable('b0', [num_hidden], initializer=b_init)
##            h0 = tf.nn.relu(tf.matmul(z, w0) + b0)

##            w1 = tf.get_variable('w1', [h0.get_shape()[1], 1], initializer=w_init)
##            b1 = tf.get_variable('b1', [1], initializer=b_init)
            
##        self.graph = tf.matmul(h0, w1) + b1

##        self.optimzer = optimzer

##class Discriminator(object):
##    def __init__(self, x, num_hidden=32, name='D', optimzer=None):
##        self.name = name

##        w_init = tf.contrib.layers.variance_scaling_initializer()
##        b_init = tf.constant_initializer(0.)

##        with tf.variable_scope(name):
##            w0 = tf.get_variable('w0', [x.get_shape()[1], num_hidden], initializer=w_init)
##            b0 = tf.get_variable('b0', [num_hidden], initializer=b_init)
##            h0 = tf.nn.relu(tf.matmul(x, w0) + b0)            

##            w1 = tf.get_variable('w1', [h0.get_shape()[1], 1], initializer=w_init)
##            b1 = tf.get_variable('b1', [1], initializer=b_init)

##        self.graph = tf.sigmoid(tf.matmul(h1, w1) + b1)

##        self.optimzer = optimzer

###class Optimizer(object):
###    def __init__(self, decay=0.95, num_decay_steps=400, initial_learning_rate=0.03):
###        self.decay = decay
###        self.num_decay_steps = num_decay_steps
###        self.initial_learning_rate = initial_learning_rate

###    def get_optimizer(self, loss, var_list):
###        batch = tf.Variable(0)
###        learning_rate = tf.train.exponential_decay(
###            self.initial_learning_rate, batch, 
###            self.num_decay_steps, self.decay, staircase=True
###        )

###        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
###            loss, global_step=batch, var_list=var_list
###        )

###        return optimizer

##class ClassicGAN(object):
##    def __init__(self, generator, discriminator, pre_discriminator=None, pre_d_opt=None, g_opt=None, d_opt=None):
##        self.generator = generator
##        self.discriminator = discriminator

##        self.pre_discriminator = pre_discriminator
##        if self.pre_discriminator == None:
##            self.pre_discriminator = discriminator

##        self.g_opt = g_opt
##        self.d_opt = d_opt
##        self.pre_d_opt = pre_d_opt

##        self._eps = 1e-2


##    def _get_default_optimizer(self, loss, var_list, decay=0.95, num_decay_steps=400, initial_learning_rate=0.03):
##        batch = tf.Variable(0)
##        learning_rate = tf.train.exponential_decay(
##            initial_learning_rate, batch, 
##            num_decay_steps, decay, staircase=True
##        )

##        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
##            loss, global_step=batch, var_list=var_list
##        )

##        return optimizer

##    def init_pretrain_D(self, pre_input, pre_labels, pre_d_opt=None):
##        opt = None
##        if pre_d_opt != None:
##            opt = pre_d_opt
##        elif (pre_d_opt == None) and (self.pre_d_opt != None):
##            opt = self.pre_d_opt
##        else:
##            pre_d_loss = tf.reduce_mean(tf.square(self.pre_discriminator.discrimination - pre_labels))
##            pre_d_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=pre_discriminator.name)
##            opt = _get_default_optimizer(pre_d_loss, pre_d_param)

##    def pretrain_D_onestep(self, tf_session):
        

        

##def main():
##    mu = 0
##    sigma = 1

##    batch_size = 150
##    learning_rate = 0.03
##    train_iters = 3000

##    num_hidden = 32

##    generator = Generator(z, num_hidden, shape(None, 1), 'G_z')
    
        
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
    def __init__(self, range):
        self.range = range
    

    def sample(self, num):
        offset = np.random.random(num) * (float(self.range[1] - self.range[0]) / num)
        samples = np.linspace(range[0], range[1]) + offset
        return samples

class NetFactory(ABC):
    def __init__(self, default_args_dict=None):
        self.default_args_dict = default_args_dict


    def create(self, name, input, reuse_var=False, args_dict=None):
        with tf.variable_scope(name) as scope:
            scope.reuse_variables()
    
            if args_dict is None:
                net_args = self.default_args_dict                
            else:
                net_args = args_dict
        
            net_out = _create_net(input, net_args)
            net = Net(name, input, False, net_out)

        return net


    @abstractmethod
    def _create_net(self, input, args_dict):
        raise NotImplementedError('define _create_net()')


class Gaussian1DGeneraterFactory(NetFactory):
    def _create_net(self, input, args_dict=None):
        w0 = tf.get_variable('w0', 
                             [input.get_shape()[1], kwargs['num_hidden']], 
                             initializer=w_init)
        b0 = tf.get_variable('b0', [kwargs['num_hidden']], 
                                initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(z, w0) + b0)

        w1 = tf.get_variable('w1', [h0.get_shape()[1], 1], initializer=w_init)
        b1 = tf.get_variable('b1', [1], initializer=b_init)
            
        o = tf.matmul(h0, w1) + b1

        return o

class Gaussian1DDiscriminatorFactory(NetFactory):
    def _create_net(self, input, args_dict=None):
        w0 = tf.get_variable('w0', 
                             [input.get_shape()[1], kwargs['num_hidden']], 
                             initializer=w_init)
        b0 = tf.get_variable('b0', [kwargs['num_hidden']], 
                                initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(z, w0) + b0)

        w1 = tf.get_variable('w1', [h0.get_shape()[1], 1], initializer=w_init)
        b1 = tf.get_variable('b1', [1], initializer=b_init)
            
        o = tf.matmul(h0, w1) + b1

        return o


class Net(object):
    def __init__(self, name, input, shared=False, output=None):
        self.name = name
        self.input = input
        self.output = output
        self.shared = shared


#class Graph(metaclass=ABCMeta):
#    #def __init__(self, name, input, share=False, **kwargs):
#    #    self.name = name
#    #    self.input = input
#    #    self.output = None
#    #    self.shared = share

#    #    _create_graph(input, kwargs)

#    #@abc.abstractmethod
#    #def _create_graph(input, **kwargs):
#    #    raise NotImplementedError('define _create_graph()')


#    #def clone(self, new_name, new_input, **kwargs):
#    #    new_graph = Graph(new_name, new_input, False, kwargs)
        
#    #    return new_graph

#    def __init__(self, name, input, share=False):
#        self.name = name
#        self.input = input
#        self.output = None
#        self.shared = share


#    def share(self, new_input):
#        self.shared = True
        
#        shared_graph = Graph(self.name, new_input, True)
#        shared_graph.output = self.output

#        return shared_graph

#class GraphFactory(metaclass=ABCMeta):
#    @abc.abstractmethod
#    def create(self, name, input, **kwargs):
#        raise NotImplementedError('define create()')

#class Gussian1DGeneratorFactory(GraphFactory):
#    def create(self, name, input, **kwargs):
#        graph = Graph(name, input, False, kwargs)


#class Gaussian1DGenerator(Graph):
#    def __init__(self, name, z, share=False, num_hidden=32):
#        Graph.__init__(self, name, z, share, 
#                       {'num_hidden' : num_hidden})

#    def _create_graph(input, **kwargs):
#        w_init = tf.truncated_normal_initializer(stddev=2)
#        b_init = tf.constant_initializer(0.)

#        with tf.variable_scope(Graph.name) as scope:
#            if Graph.shared:
#                scope.reuse_variables()

#            w0 = tf.get_variable('w0', 
#                                 [input.get_shape()[1], kwargs['num_hidden']], 
#                                 initializer=w_init)
#            b0 = tf.get_variable('b0', [kwargs['num_hidden']], 
#                                 initializer=b_init)
#            h0 = tf.nn.relu(tf.matmul(z, w0) + b0)

#            w1 = tf.get_variable('w1', [h0.get_shape()[1], 1], initializer=w_init)
#            b1 = tf.get_variable('b1', [1], initializer=b_init)
            
#        Graph.output = tf.matmul(h0, w1) + b1
        


#class Gaussian1DDiscriminator(Graph):
#    def __init__(self, name, x, share=False, num_hidden=32):
#        Graph.__init__(self, name, x, share,
#                       {'num_hidden':num_hidden})

#    def _create_graph(input, **kwargs):
#        w_init = tf.contrib.layers.variance_scaling_initializer()
#        b_init = tf.constant_initializer(0.)

#        with tf.variable_scope(name) as scope:
#            if Graph.shared:
#                scope.reuse_variables()

#            w0 = tf.get_variable('w0', 
#                                 [input.get_shape()[1], kwargs['num_hidden']], 
#                                 initializer=w_init)
#            b0 = tf.get_variable('b0', [kwargs['num_hidden']], 
#                                 initializer=b_init)
#            h0 = tf.nn.relu(tf.matmul(z, w0) + b0)

#            w1 = tf.get_variable('w1', [h0.get_shape()[1], 1], initializer=w_init)
#            b1 = tf.get_variable('b1', [1], initializer=b_init)

#        Graph.output = tf.sigmoid(tf.matmul(h1, w1) + b1)

##class Gaussian1DDiscriminatorPre(Gaussian1DDiscriminator):
##    def __init__(self, pre_input, pre_label, 
##                 num_hidden=32, name='D_pre'):
##        Gaussian1DDiscriminator.__init__(self, pre_input, 
##                                         num_hidden, name)

#class GAN(object):
#    def __init__(self, generator, discriminator, 
#                 discriminator_pre=None):
#        self.generator = generator
#        self.discriminator = discriminator
#        self.discriminator_pre = discriminator_pre

#        if self.discriminator_pre is None:
#            self.discriminator_pre = self.discriminator



##class OptimizerFactory(metaclass=ABCMeta):
##    @abc.abstractmethod
##    def create(self, graph):
##        raise NotImplementedError('define create()')

##class DefaultOptimizerFactory(OptimizerFactory):
##    def create(self, graph, num_

##class DiscriminatorPreOptFactory(OptimizerFactory):
##    pass

##class DiscriminatorOptFactory(OptimizerFactory):
##    #def create(self, 
##    pass

##class GeneratorOptFactory(OptimizerFactory):
##    pass

#def get_default_optimizer(loss, var_list, 
#                          decay=0.95, num_decay_steps=400,
#                          initial_learning_rate=0.03):
#    batch = tf.Variable(0)
#    learning_rate = tf.train.exponential_decay(
#        initial_learning_rate, batch, num_decay_steps, decay,
#        staircase=True)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
#        loss, global_step=batch, var_list=var_list)

#    return optimizer

#def get_pre_d_optimizer(graph, base_opt):
#    graph.graph
    

class GANTrainer(object):
    def __init__(self, x_sampler, z_sampler, 
                 generator_factory, discriminator_factory,
                 batch_size=150, learning_rate=0.03, train_iters=3000):
        self._eps = 1e-2

        self.x_sampler = x_sampler
        self.z_sampler = z_sampler

        self.g_factory = generator_factory
        self.d_factory = discriminator_factory

        self.G_z_input, self.G_z = self._init_G_z_net()        
        self.D_pre_input, self.D_pre_label, self.D_pre = _init_D_pre_net()
        self.D_real_input, self.D_real, self.D_fake = _init_D_net()

        self.loss_g, self.opt_g = _init_G_loss_opt()
        self.loss_d_pre, self.opt_d_pre = _init_D_pre_loss_opt()
        self.loss_d, self.opt_d = _init_D_loss_opt()   

    def _init_G_z_net(self):
        G_z_input = tf.placeholder(tf.float32, shape=(None, 1))
        G_z = self.g_factory.create('G_z', G_z_input)

        return G_z_input, G_z


    def _init_D_pre_net(self):
        D_pre_input = tf.placeholder(tf.float32, shape=(None, 1))
        D_pre_label = tf.placeholder(tf.float32, shape=(None, 1))
        D_pre = self.d_factory.create('D_pre', D_pre_input)

        return D_pre_input, D_pre_label, D_pre


    def _init_D_net(self):
        D_real_input = tf.placeholder(tf.float32, shape=(None, 1))
        D_real = self.discriminator_factory.create('D', D_real_input)
        D_fake = self.discriminator_factory.create('D', self.G_z.output, True)
        
        return D_real_input, D_real, D_fake

    
    def _init_G_loss_opt(self):
        loss_g = tf.reduce_mean(-tf.log(self.D_fake.output + _eps))
        params_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                     scope=self.G_z.name)
        opt_g = _get_default_optimizer(
            loss_g, params_g, initial_learning_rate=self.learning_rate/2)

        return loss_g, opt_g


    def _init_D_pre_loss_opt(self):
        loss_d_pre = tf.reduce_mean(tf.square(self.D_pre.output - self.D_pre_label))
        params_d_pre = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.D_pre.name)
        opt_d_pre = _get_default_optimizer(
            loss_d_pre, params_d_pre, initial_learning_rate=self.learning_rate)

        return loss_d_pre, opt_d_pre

    def _init_D_loss_opt(self):
        loss_d = tf.reduce_mean(
            -tf.log(self.D_real.output + _eps) - tf.log(1-self.D_fake.output + _eps))
        params_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.D_real.name)
        opt_d = _get_default_optimizer(
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
    train_iters = 3000

    num_hidden = 32


    x_sampler = Gaussian1D(mu, sigma)
    z_sampler = Noise(z_range)

    g_factory = Gaussian1DGeneraterFactory({'num_hidden':num_hidden})
    d_factory = Gaussian1DDiscriminatorFactory({'num_hidden':num_hidden})

    gan_trainer = GANTrainer(x_sampler, z_sampler, 
                             g_factory, d_factory, 
                             batch_size, learning_rate, train_iters)


def main():
    train_gaussian_1D_GAN()

    
    

#class GANTrainer(object):
#    def __init__(self, x_sampler, z_sampler,
#                 generator, discriminator):
#        self._eps = 1e-2

#        self.x_sampler = x_sampler 
#        self.z_sampler = z_sampler

#        self.g = generator
#        self.d_real = discriminator
#        self.d_fake = discriminator.share()

#        self.g_loss, self.g_opt = _init_g()

        

#        self.pre_d_loss = None
#        self.pre_d_params = None
#        self.pre_d_opt = None
        
#        self.d_loss = None
#        self.d_params = None
#        self.d_opt = None

#        self.g_loss = None
#        self.g_params = None
#        self.g_opt = None

#    def _init_g(self, generator):
#        loss_g = 
#    def _init_pre_d(self, pre_d):
#        self.pre_d_loss = tf.reduce_mean(tf.square(pre_d.output - 
#    def init_pretraining_D(self, pre_input, pre_label, discriminator, optimizer=None):
#        self.pre_d_loss = tf.reduce_mean(tf.square(discriminator.graph - pre_label))
#        self.pre_d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = discriminator.name)
        

#    def _get_default_optimizer(self, loss, params, decay=0.95, num_decay_steps=400, initial_learning_rate=0.03):
#        pass
        

        

        
        
        

        

#def main():
#    g_sampler = Gaussian1D(0, 0.1)
#    #s = g_sampler.sample(10)

if __name__ == '__main__':
    main()