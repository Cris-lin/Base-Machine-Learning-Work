"""
Created on 20210310

@author: CrisLin
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import time
import DNN_tf1 as nntf1
import matplotlib.pyplot as plt

def main():

    time_start = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ## meta_data
    minibatch_size = 25
    learning_rate = 5e-5
    epochs = 100
    training_size = 10000
    testing_size = 100
    input_shape = 784
    output_shape = 10
    depth = 3
    net_width = [1000, 500, 500]
    activations = ['sigmoid', 'sigmoid', 'sigmoid']


    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    #mnist=tf.keras.datasets.mnist
    mnist = input_data.read_data_sets("./MNIST", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    #(trX, trY), (teX, teY) = mnist.load_data()
    training_data = np.reshape(trX, (-1, 784))
    training_label = np.reshape(trY, (-1,10))
    testing_data = np.reshape(teX, (-1,784))
    testing_label = np.reshape(teY, (-1,10))

    classifier = nntf1.DNN( input_shape, output_shape, depth, net_width, activations)

    data = tf.placeholder(name = 'data', dtype = tf.float32, shape = (None, input_shape))
    label = tf.placeholder(name = 'label', dtype = tf.float32, shape = (None, output_shape))

    predict = classifier.get_output(data)
    loss = tf.reduce_mean( tf.square(label - predict ))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_list = []
        for step in range(0, epochs+1):
            cur_time = time.time()
            loss_val = sess.run(loss, feed_dict = {data:training_data[0:training_size], label:training_label[0:training_size]})
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
                plt.savefig(f'./result/loss.png')

        with open('./result/output.txt', 'w') as f:
            for idx in range(0,testing_size):
                pic = np.reshape(testing_data[idx], (1, 784))
                lab = testing_label[idx]
                output = sess.run(predict, feed_dict = {data:pic})
                f.write(str(output)+"\n")
                #f.write("///")
                f.write(str(lab)+"\n")


if __name__ == '__main__':
    main()


