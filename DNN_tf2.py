'''
created at 2021 02 13

author@CrisLin

'''

import tensorflow as tf
from tensorflow.keras import layers,models

def DNN( in_shape, out_shape, depth, net_width, activations):

    model = models.Sequential()
    model.add(layers.Dense(net_width[0], activation = activations[0], input_shape = (in_shape,)))

    for idx in range(1, depth-1):
        model.add(layers.Dense(net_width[idx], activation = activations[idx]))

    model.add(layers.Dense( out_shape ))

    model.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=['mse'])

    return model




def main():
    print('running DNN_tf2')

if __name__ == '__main__':
    main()