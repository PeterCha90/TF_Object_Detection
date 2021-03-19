"""MobilenNet v2 models for tf.keras

# Reference
This file contains building code for MobileNetV2, based on MobileNetV2: 
[Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

and retranslates old
[mobilenet_v2_keras](https://github.com/JonathanCMitchell/mobilenet_v2_keras)
keras code into tensorflow version 2.4.1 format. 
"""

import os
import warnings
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.Add as Add


def inverted_res_block(inputs, expansion, out_channels,  
                       stride, block_id):
    in_channels = inputs.shape[-1]

    if expansion != 1:
        # 1x1 conv2d, ReLU6 -> expansion
        x = layers.Conv2D(in_channels * expansion, 
                        kernel_size=1, padding='same', 
                        use_bias=False)(inputs)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                      name='bn%d_conv_bn_expand' %
                                            block_id)(x)
        x = tf.nn.relu6(x, name='conv_%d_relu' % block_id)
    
    # 3x3 depthwise s=s, ReLU6
        x = layers.DepthwiseConv2D(kernel_size=3, strides=stride,
                                   activation=None, use_bias=False,
                                   padding='same',
                                   name='mobl%d_conv_depthwise' 
                                         % block_id)(x)
    else:
        x = layers.DepthwiseConv2D(kernel_size=3, strides=stride,
                                   activation=None, use_bias=False,
                                   padding='same',
                                   name='mobl%d_conv_depthsiwe' 
                                         % block_id)(inputs)

    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name='bn%d_conv_depthwise' 
                                        % block_id)(x)
    x = tf.nn.relu6(x, name='conv_dw_%d_relu' % block_id)
    
    # linear 1x1 conv2d
    x = layers.Conv2D(out_channels, kernel_size=1,
                      padding='same', use_bias=False,
                      activation=None,
                      name='bn%d_conv_bn_project' % block_id)(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name='bn%d_conv_bn_project' 
                                  % block_id)(x)
    
    if in_channels == out_channels and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x



def MobileNetV2(input_shape=None, input_tensor=None,
                classes=1000, include_top=True,
                weights=None):
    """Instantiates the MobileNetV2 architecture.
    """
    # weights path check 
    if not os.pah.exists(weights):
        raise ValueError('The weights file is not found.')

    # input channel position check
    if K.image_data_format() != 'channels_last':
        raise ValueError('for the input data format "channels_last" '
                         '(width, height, channels). ')    
    # input tensor
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor,
                              shape=input_shape)
        else:
            img_input = input_tensor
    # model 
    """Inverted res block configuration.
    Each tuple consists of 4 elements like following, 
    (expansion factor, out_channels, num_of_repeat, stride)
    as written in the paper Table 2.
    """
    configs = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2),
              (6, 64, 4, 2), (6, 92, 3, 1), (6, 160, 3, 2),
              (6, 320, 1, 1)]

    x = layers.Conv2D(32, kernel_size=3, strides=(2, 2), 
                      padding='same', use_bias=False, 
                      name='Conv1')(img_input)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name='bn_Conv1')(x)
    x = tf.nn.relu6(x, name='Conv1_relu')

    for i, config in enumerate(configs):
        for j in config[2]:
            stride = 1 if j != 0 else config[3]
            x = inverted_res_block(x, 
                                   config[0], 
                                   config[1],
                                   stride, 
                                   block_id=i+j)

    x = layers.Conv2D(1280, kernel_size=1, 
                      use_bias=False, name='Conv_last')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name='Conv_last_bn')(x)
    x = tf.nn.relu6(x, name='out_relu')
    
    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(classes, activation='softmax',
                         name='Logits')(x)
    # Create model.
    model = tf.keras.Model(inputs, x, name='MoblieNet_v2')

    # load weights
    if weights is not None:
        model.load_weights(weights)

    return model
