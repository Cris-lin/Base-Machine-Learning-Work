'''

Created at 2021-03-15

@author: CrisLin

'''

import tensorflow as tf
from tensorflow.keras import layers,models

def CNN(input_shape, input_channels, output_shape, output_activation, conv_depth, conv_kernel_size, output_channels, conv_strides, conv_paddings, pooling_types, pooling_sizes, \
        pooling_strides, pooling_paddings, dense_depth, dense_width, activations):
    
    model = models.Sequential()

    for conv_idx in range(0, conv_depth):
        model.add( layers.Conv2D( output_channels[conv_idx], conv_kernel_size[conv_idx], conv_strides[conv_idx], conv_paddings[conv_idx], 'channels_last', activation = 'relu') )
        if pooling_types[conv_idx] == 'max_pooling':
            model.add( layers.MaxPooling2D( pooling_sizes[conv_idx], pooling_strides[conv_idx], pooling_paddings[conv_idx], 'channels_last') )
        if pooling_types[conv_idx] == 'avg_pooling':
            model.add( layers.AveragePooling2D( pooling_sizes[conv_idx], pooling_strides[conv_idx], pooling_paddings[conv_idx], 'channels_last') )
    
    model.add( layers.Flatten('channels_last') )

    for dense_idx in range(0, dense_depth):
        model.add( layers.Dense( dense_width[dense_idx], activation = activations[dense_idx]))

    model.add( layers.Dense( output_shape, activation = output_activation) )

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

