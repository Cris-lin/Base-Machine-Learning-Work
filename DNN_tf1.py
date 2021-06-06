"""
Created on 20210309

@author: CrisLin
"""

import numpy as np
import tensorflow as tf

class Dense_layer(object):

    def __init__(self, last_width, width, activate, index):
        self.W = tf.get_variable( name = 'parameter_W'+str(index), shape = (last_width, width), initializer =tf.random_normal_initializer(mean=0, stddev=1), dtype = tf.float32)
        self.b = tf.get_variable( name = 'parameter_b'+str(index), shape = (width), initializer =tf.random_normal_initializer(mean=0, stddev=1), dtype = tf.float32)
        self.activation = activate
    
    def forward(self, x):
        if self.activation == 'relu':
            return tf.nn.relu( tf.matmul(x, self.W) + self.b )
        if self.activation == 'sigmoid':
            return tf.nn.sigmoid( tf.matmul(x, self.W) + self.b )
        if self.activation == 'softmax':
            return tf.nn.softmax(tf.matmul(x, self.W) + self.b )
        if self.activation == 'normal':
            return tf.matmul(x, self.W) + self.b

class DNN(object):

    def __init__(self, input_shape, output_shape, depth, net_width, activations, output_activation ):
        
        self.layers = []
        self.layers.append(Dense_layer(input_shape, net_width[0], activations[0], 0))

        for layer_index in range(1, depth):
            self.layers.append(Dense_layer(net_width[layer_index-1], net_width[layer_index], activations[layer_index], layer_index))

        if output_activation == 'softmax':
            self.layers.append(Dense_layer(net_width[-1], output_shape, 'softmax', depth))
        if output_activation == 'sigmoid':
            self.layers.append(Dense_layer(net_width[-1], output_shape, 'sigmoid', depth))
        if output_activation == 'normal':
            self.layers.append(Dense_layer(net_width[-1], output_shape, 'normal', depth))

    def get_output(self, x):

        for layer in self.layers:
            x = layer.forward(x)
        
        return x
            

def main():
    print('define DNN')

if __name__ == '__main__':
    main()
