'''

Created at 2021-03-14
author@CrisLin

'''

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


class Conv_layer(object):

    def __init__(self, kernel_size, input_channels, output_channels, conv_stride, conv_padding, index, pool_type, pool_size, pool_stride, pool_padding):
        self.W = tf.get_variable( name = 'convolution_kernel'+str(index), shape = (kernel_size, kernel_size, input_channels, output_channels), \
        initializer =tf.random_normal_initializer(mean=0, stddev=1), dtype = tf.float32)
        self.stride = conv_stride
        self.padding = conv_padding
        self.pooling_type = pool_type
        self.pooling_size = pool_size
        self.pooling_stride = pool_stride
        self.pooling_padding = pool_padding

    def forward(self, x):
        y = tf.nn.conv2d( x, self.W, self.stride, self.padding)
        if self.pooling_type == 'max_pooling':
            return tf.nn.max_pool(y, [1, self.pooling_size, self.pooling_size, 1], self.pooling_stride, self.pooling_padding)
        if self.pooling_type == 'avg_pooling':
            return tf.nn.avg_pool(y, [1, self.pooling_size, self.pooling_size, 1], self.pooling_stride, self.pooling_padding)


class CNN(object):

    def __init__(self, input_shape, input_channels, output_shape, output_activation, conv_depth, conv_kernel_size, output_channels, conv_strides, conv_paddings, pooling_types, pooling_sizes, \
        pooling_strides, pooling_paddings, dense_depth, dense_width, activations):
        
        self.layers = []
        self.conv_dep = conv_depth
        self.dense_dep = dense_depth
        
        conv_output_shape = input_shape

        for layer_idx in range(0, conv_depth):
            if conv_paddings[layer_idx] == 'VALID':
                conv_output_shape = (conv_output_shape - 2*conv_kernel_size[layer_idx] + 1)/conv_strides[layer_idx] + 1
            if conv_paddings[layer_idx] == 'SAME':
                conv_output_shape = conv_output_shape/conv_strides[layer_idx]
            if pooling_paddings[layer_idx] == 'VALID':
                conv_output_shape = (conv_output_shape - 2*pooling_sizes[layer_idx] + 1)/pooling_strides[layer_idx] + 1
            if pooling_paddings[layer_idx] == 'SAME':
                conv_output_shape = conv_output_shape/pooling_strides[layer_idx]
        
        conv_output_shape = int(conv_output_shape)

        self.output_features = conv_output_shape*conv_output_shape*output_channels[-1]

        self.layers.append( Conv_layer( conv_kernel_size[0], input_channels, output_channels[0], conv_strides[0], conv_paddings[0], 0, pooling_types[0], \
            pooling_sizes[0], pooling_strides[0], pooling_paddings[0]))

        for conv_idx in range(1, conv_depth):
            self.layers.append( Conv_layer( conv_kernel_size[conv_idx], output_channels[conv_idx-1], output_channels[conv_idx], conv_strides[conv_idx], \
                conv_paddings[conv_idx], conv_idx, pooling_types[conv_idx], pooling_sizes[conv_idx], pooling_strides[conv_idx], pooling_paddings[conv_idx]) )
        
        self.layers.append( Dense_layer( self.output_features, dense_width[0], activations[0], 0))

        for dense_idx in range(1, dense_depth):
            self.layers.append( Dense_layer( dense_width[dense_idx-1], dense_width[dense_idx], activations[dense_idx], dense_idx))

        if output_activation == 'softmax':
            self.layers.append(Dense_layer(dense_width[-1], output_shape, 'softmax', dense_depth))
        if output_activation == 'sigmoid':
            self.layers.append(Dense_layer(dense_width[-1], output_shape, 'sigmoid', dense_depth))
        if output_activation == 'normal':
            self.layers.append(Dense_layer(dense_width[-1], output_shape, 'normal', dense_depth))

    
    def get_output(self, x):
        
        for conv_idx in range(0, self.conv_dep):
            x = self.layers[conv_idx].forward(x)
 
        x = tf.reshape( x, (-1, self.output_features))

        for dense_idx in range( self.conv_dep, self.conv_dep+self.dense_dep+1):
            x = self.layers[dense_idx].forward(x)

        return x

def main():
    print('Runing CNN file')

if __name__ == '__main__':
    main()