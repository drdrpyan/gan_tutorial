import tensorflow as tf

#def deconv2d(input, filters, kernel_size, stride, pad, batch_norm=None, actvation=None, name='deconv2d'):
#    if len(kernel_size) == 1:
#        k_w = kernel_size
#        k_h = kernel_size
#    elif len(kernel_size) == 2:
#        k_w = kernel[0]
#        k_h = kernel[1]
#    else:
#        raise Exception('kernel_size needs one or two positive integers')
    
#    #with tf.variable_scope(name):
#    #    w = tf.get_variable('w', 
#    #pass
#    tf.layers.conv2d_transpose(input, filters, kernel_size, 

def conv2d(input, filters, kernel_size, stride, padding, batch_norm=None, activation=None, name=None):
    if len(kernel_size) == 1:
        k_w = kernel_size
        k_h = kernel_size
    elif len(kernel_size) == 2:
        k_w = kernel_size[0]
        k_h = kernel_size[1]
    else:
        raise Exception('kernel_size needs one or two positive integers')

    if len(stride) == 1:
        s_vertical = stride
        s_horizontal = stride
    elif len(stride) == 2:
        s_vertical = stride[0]
        s_horizontal = stride[1]
    else:
        raise Exception('stride needs one or two positive integers')
        
    w = tf.Variable(tf.random_normal([k_w, k_h, input.get_shape()[0],  filters])
    #h = tf.nn.conv2d(input, w, (1, s_vertical, s_horizontal, 1), 
    #w = tf.get_variable('conv_w', 

def leaky_relu(input, leak=0.2):
    return tf.maximum(input, leak*x)