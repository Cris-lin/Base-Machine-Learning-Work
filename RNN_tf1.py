'''

Created at 2021-03-16
author@CrisLin

'''

from typing import Sequence, final
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

class RNN_layer(object):

    def __init__(self, input_shape, output_shape, latent_shape, activate ):
        self.W = tf.get_variable( name = 'transfer_weight', shape = (latent_shape, latent_shape), initializer =tf.random_normal_initializer(mean=0, stddev=1), dtype = tf.float32 )
        self.U = tf.get_variable( name = 'encoding_parameter', shape = (input_shape, latent_shape), initializer =tf.random_normal_initializer(mean=0, stddev=1), dtype = tf.float32 )
        self.V = tf.get_varibale( name = 'decoding_paraqmeter', shape = (latent_shape, output_shape), initializer =tf.random_normal_initializer(mean=0, stddev=1), dtype = tf.float32 )
        self.b = tf.get_variable( name = 'bias', shape = (latent_shape), initializer =tf.random_normal_initializer(mean=0, stddev=1), dtype = tf.float32 )
        self.c = tf.get_variable( name = 'latent_bias', shape = (output_shape), initializer =tf.random_normal_initializer(mean=0, stddev=1), dtype = tf.float32 )
        self.latent_shape = latent_shape
        self.activation = activate

    def forward(self, x):
        self.latent_code = tf.get_variable( name = 'latent_code', shape = (tf.shape(x)[0], self.latent_shape), initializer =tf.random_normal_initializer(mean=0, stddev=1), dtype = tf.float32)
        output = []
        for token in x:
            if self.activation == 'relu':
                self.latent_code = tf.nn.relu( tf.matmul(token, self.U) + tf.matmul(self.latent_code, self.W) + self.b )
                output.append( tf.nn.relu(tf.matmul( self.latent_code, self.V ) + self.c ) )
            if self.activation == 'sigmoid':
                self.latent_code = tf.nn.sigmoid( tf.matmul(token, self.U) + tf.matmul(self.latent_code, self.W) + self.b )
                output.append( tf.nn.sigmiod(tf.matmul( self.latent_code, self.V ) + self.c ) )
            if self.activation == 'normal':
                self.latent_code = tf.matmul(token, self.U) + tf.matmul(self.latent_code, self.W) + self.b
                output.append( tf.matmul( self.latent_code, self.V ) + self.c  )
            
        return output

class RNN(object):

    def __init__(self, input_shape, output_shape, feature_shape, latent_shape, rnn_activate, dense_depth, dense_width, dense_activations, output_activation  ) -> None:
        
        self.layers = []
        self.layers.append( RNN_layer( input_shape, feature_shape, latent_shape, rnn_activate) )

        self.layers.append( Dense_layer( feature_shape, dense_width[0], dense_activations[0], 0 ) )
        for dense_idx in range(1, dense_depth):
            self.layers.append( Dense_layer( dense_width[dense_idx-1], dense_width[dense_idx], dense_activations[dense_idx], dense_idx ) )

        self.layers.append( Dense_layer( dense_width[-1], output_shape, output_activation, dense_depth+1))

    def forward(self, x):
        final_output = []
        for sample in x:
            rnn_output = self.layers[0].forward(sample)
            net_output = []
            for token in rnn_output:
                y = token
                for layer in self.layers[1:]:
                    y = layer.forward(y)
                net_output.append(y)
            final_output.append(tf.sigmoid(tf.reduce_sum(net_output)))

        return final_output
        
        
