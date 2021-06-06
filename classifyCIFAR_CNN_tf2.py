'''

Created at 2021-03-15
@author: CrisLin

'''

import os
import numpy as np
import tensorflow as tf
import CNN_tf2 as cnntf2
from tensorflow.keras.datasets import cifar10

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    #meta data
    minibatch_size = 25
    learning_rate = 5e-5
    epochs = 300
    training_size = 3000
    testing_size = 100
    input_shape = [32,32,3]
    output_shape = 10

    conv_depth = 2
    conv_kernel_size = [3, 3]
    conv_channels = [8, 4]
    conv_strides = [1, 1]
    conv_paddings = ['same', 'same']

    pooling_types = ['max_pooling', 'avg_pooling']
    pooling_sizes = [2, 2]
    pooling_strides = [2, 2]
    pooling_paddings = ['valid', 'valid']

    dense_depth = 5
    dense_width = [1000, 500, 500, 500, 500]
    activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']

    #import data
    (training_data, training_label),(testing_data, testing_label) = cifar10.load_data()

    training_data = training_data/255*2.0 - 1.0
    training_label = tf.keras.utils.to_categorical(training_label)

    classifier = cnntf2.CNN( input_shape[0], input_shape[2], output_shape, 'softmax', conv_depth, conv_kernel_size, conv_channels, conv_strides, conv_paddings, \
        pooling_types, pooling_sizes, pooling_strides, pooling_paddings, dense_depth, dense_width, activations)

    classifier.fit( training_data[0:training_size], training_label[0:training_size], epochs = 50, batch_size = 25)

    classifier.summary()

if __name__ == '__main__':
    main()