'''

Created at 2021-03-14
@author: CrisLin

'''

import os
import numpy as np
import tensorflow as tf
import time
import pickle
import CNN_tf1 as cnntf
import matplotlib.pyplot as plt

def main():

    time_start = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    ## meta_data
    minibatch_size = 25
    learning_rate = 5e-5
    epochs = 300
    training_size = 3000
    testing_size = 100
    input_shape = [32,32,3]
    output_shape = 1

    conv_depth = 2
    conv_kernel_size = [3, 3]
    conv_channels = [8, 4]
    conv_strides = [1, 1]
    conv_paddings = ['SAME', 'SAME']

    pooling_types = ['max_pooling', 'avg_pooling']
    pooling_sizes = [2, 2]
    pooling_strides = [2, 2]
    pooling_paddings = ['SAME', 'SAME']

    dense_depth = 5
    dense_width = [1000, 500, 500, 500, 500]
    activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']

    with open('./cifar10/cifar-10-python/cifar-10-batches-py/data_batch_1', 'rb') as f:
        training_dict = pickle.load(f, encoding='bytes')
    
    training_data = np.reshape(training_dict[b'data'][0:training_size], (training_size, input_shape[0], input_shape[1], input_shape[2]))/255*2.0 - 1.0
    training_label = np.reshape(training_dict[b'labels'][0:training_size], (training_size, 1) )/10.0

    classifier = cnntf.CNN( input_shape[0], input_shape[2], output_shape, 'sigmoid', conv_depth, conv_kernel_size, conv_channels, conv_strides, conv_paddings, \
        pooling_types, pooling_sizes, pooling_strides, pooling_paddings, dense_depth, dense_width, activations)

    data = tf.placeholder( name = 'input_data', dtype = tf.float32, shape = (None, input_shape[0], input_shape[1], input_shape[2]))
    label = tf.placeholder( name = 'predict_output', dtype = tf.float32, shape = (None, output_shape))

    predict = classifier.get_output(data)
    loss = tf.reduce_mean( tf.square(predict - label))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_list = []
        for step in range(0, epochs+1):
            cur_time = time.time()
            loss_val = sess.run(loss, feed_dict = {data:training_data, label:training_label})
            if step%10 == 0 :
                print('第', step, '次迭代, Loss =', loss_val, ',已运行%.6fs' % (cur_time - time_start))
            for index in range(0, training_size, minibatch_size):
                sess.run(train_op, feed_dict = {data:training_data[index:index+minibatch_size], label:training_label[index:index+minibatch_size]})
            loss_list.append(loss_val)

            if step >= epochs:
                plt.figure()
                plt.xlabel("iterations") 
                plt.ylabel("log-loss") 
                plt.plot(range(0,epochs+1), np.log(loss_list))
                plt.savefig(f'./result/CNN/loss.png')

if __name__ == '__main__':
    main()

