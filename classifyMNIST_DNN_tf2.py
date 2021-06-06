'''
created at 2021 03 13

author@CrisLin
'''
import os
import numpy as np
import tensorflow as tf
import time
import DNN_tf2 as nntf2
from tensorflow.keras.datasets import mnist

def main():

    #meta data
    training_size = 10000
    input_shape = 784
    output_shape = 1
    depth = 3
    net_width = [1000, 500, 500]
    activations = ['sigmoid', 'sigmoid', 'sigmoid']

    #import data
    (training_data, training_label),(testing_data, testing_label) = mnist.load_data()

    training_data = tf.reshape(training_data, (-1,784))
    classifier = nntf2.DNN(input_shape, output_shape, depth, net_width, activations)
    classifier.summary()

    classifier.fit( training_data[0:training_size], training_label[0:training_size], epochs = 50, batch_size = 25)


if __name__ == '__main__':
    main()
