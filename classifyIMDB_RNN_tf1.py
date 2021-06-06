'''

Created at 2021-03-18
@author:CrisLin

'''

import os
import tensorflow as tf
from keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np
import time
import RNN_tf1 as nntf1

def main():

    time_start = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ## meta_data
    minibatch_size = 25
    learning_rate = 5e-5
    epochs = 300
    training_size = 3000
    testing_size = 100
    
    input_shape = 1
    output_shape = 1
    feature_shape = 100
    latent_shape = 100
    rnn_activate = 'sigmoid'
    dense_depth = 3
    dense_width = [1000, 500, 500]
    dense_activations = ['relu', 'relu', 'relu']
    output_activation = 'sigmoid'

    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="./imdb/imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

    training_data = x_train[0:training_size]
    training_label = y_train[0:training_size]

    classifier = nntf1.RNN(input_shape, output_shape, feature_shape, latent_shape, rnn_activate, dense_depth, dense_width, dense_activations, output_activation)

    data = tf.placeholder( name = 'input_data', dtype = tf.float32, shape = (None, None, input_shape))
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
                plt.savefig(f'./result/RNN/loss.png')

if __name__ == '__main__':
    main()