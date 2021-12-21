# -*- coding:UTF-8 -*-
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, DepthwiseConv2D
from keras.layers import Input, Dense, Activation, Reshape, Concatenate
from keras.layers import BatchNormalization, Dropout
from keras.layers.merge import add, concatenate
from keras.layers.core import Lambda
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
import keras
import math
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# SRNet
def SRNet(input_shape=(512, 512, 1)):
    # 初始化器
    variance_scaling_initializer = keras.initializers.VarianceScaling()
    # Layer type 1
    def Layer_typ1_1(x, nb_filter, kernel_size=(3, 3), padding='same', strides=(1, 1)):
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides,
                   kernel_initializer=variance_scaling_initializer,
                   bias_initializer=keras.initializers.Constant(0.2),
                   # kernel_regularizer=keras.layers.regularizers.l2(2e-4),
                   bias_regularizer=None)(x)  # conv
        x = BatchNormalization(momentum=0.9)(x)  # BN
        x = Activation(activation='relu')(x)  # RuLU
        return x

    # Layer type 2
    def Layer_type_2(inpt, nb_filter, kernel_size=(5, 5), padding='same', strides=(1, 1)):
        x = Layer_typ1_1(inpt, nb_filter, kernel_size, padding, strides)  # Layer type 1
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides,
                   kernel_initializer=variance_scaling_initializer,
                   bias_initializer=keras.initializers.Constant(0.2),
                   # kernel_regularizer=keras.layers.regularizers.l2(2e-4),
                   bias_regularizer=None
                   )(x)  # conv
        x = BatchNormalization(momentum=0.9)(x)  # BN
        x = add([inpt, x])  # add
        return x

    # Layer type 3
    def Layer_type_3(inpt, nb_filter, kernel_size=(3, 3), padding='same', strides=(1, 1)):
        x1 = Layer_typ1_1(inpt, nb_filter, kernel_size, padding, strides)  # Layer type 1
        x1 = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides,
                    kernel_initializer=variance_scaling_initializer,
                    bias_initializer=keras.initializers.Constant(0.2),
                    # kernel_regularizer=keras.layers.regularizers.l2(2e-4),
                    bias_regularizer=None
                    )(x1)  # conv
        x1 = BatchNormalization(momentum=0.9)(x1)  # BN
        x1 = AveragePooling2D(pool_size=(3, 3), padding=padding, strides=(2, 2))(x1)

        x2 = Conv2D(nb_filter, kernel_size=(1, 1), padding=padding, strides=(2, 2),
                    kernel_initializer='he_normal',
                    bias_initializer=keras.initializers.Constant(0.2),
                    # kernel_regularizer=keras.layers.regularizers.l2(2e-4),
                    bias_regularizer=None
                    )(inpt)  # conv
        x2 = BatchNormalization(momentum=0.9)(x2)  # BN
        x = add([x2, x1])  # add
        return x

    # Layer type 4
    def Layer_type_4(inpt, nb_filter, kernel_size=(3, 3), padding='same', strides=(1, 1)):
        x = Layer_typ1_1(inpt, nb_filter, kernel_size, padding, strides)  # Layer type 1
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides,
                   kernel_initializer=variance_scaling_initializer,
                   bias_initializer=keras.initializers.Constant(0.2),
                   # kernel_regularizer=keras.layers.regularizers.l2(2e-4),
                   bias_regularizer=None
                   )(x)  # conv
        x = BatchNormalization(momentum=0.9)(x)  # BN
        x = GlobalAveragePooling2D()(x)  # Global avgpooling
        # x = Flatten()(x)
        return x

    # bulid model
    input = Input(name='the_input', shape=input_shape, dtype='float32')


    # Two layers of type 1 L1-L2
    x = DCT_layer(input, conv_filters=36, kernel_size=(6, 6))
    x = Layer_typ1_1(x, nb_filter=2, kernel_size=(3, 3))
    x = Layer_typ1_1(x, nb_filter=2, kernel_size=(3, 3))


    # Five layers of type 2 L3-L7
    x = Layer_type_2(x, nb_filter=2, kernel_size=(3, 3))
    x = Layer_type_2(x, nb_filter=2, kernel_size=(3, 3))
    x = Layer_type_2(x, nb_filter=2, kernel_size=(3, 3))
    x = Layer_type_2(x, nb_filter=2, kernel_size=(3, 3))
    x = Layer_type_2(x, nb_filter=2, kernel_size=(3, 3))

    # Four layers of type 3 L8-L11
    x = Layer_type_3(x, nb_filter=2, kernel_size=(3, 3))
    x = Layer_type_3(x, nb_filter=2, kernel_size=(3, 3))
    x = Layer_type_3(x, nb_filter=2, kernel_size=(3, 3))
    x = Layer_type_3(x, nb_filter=2, kernel_size=(3, 3))


    # One layers of type 4 L12
    x = Layer_type_4(x, nb_filter=64, kernel_size=(3, 3))

    # Fully connecte
    x = Dense(2, kernel_initializer=keras.initializers.random_normal(mean=0., stddev=0.01),
              bias_initializer=keras.initializers.Constant(0.))(x)
    # Softmax
    x = Activation('softmax')(x)

    model = Model(inputs=[input], outputs=x)
    model.summary()

    return model

# 求绝对值
def ABS(x):
    x = K.abs(x)
    return x

# Group for xu_net
def Conv2d_BN(x, nb_filter, kernel_size=(3, 3), padding='same', conv_strides=(1, 1), isABS=0, isTanh=0, pool_size=(5, 5), pool_strides=(2, 2)):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=conv_strides)(x)
    if isABS == 1:
        x = Lambda(ABS)(x) # 借用Lambda层
    x = BatchNormalization(axis=-1, scale=False)(x)
    if isTanh == 1:
        x = Activation('tanh')(x)
    else:
        x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=pool_size, padding=padding, strides=pool_strides)(x)
    return x

#  high pass filter kernel initializer
def init_HPF(shape, dtype=None):
    HPF_kernel = np.zeros(shape, dtype=np.float32) # [height,width,input,output] shape=(5, 5, 1, 1)
    kn = np.array([[-1 / 12., 2 / 12., -2 / 12., 2 / 12., -1 / 12.],
              [2 / 12., -6 / 12., 8 / 12., -6 / 12., 2 / 12.],
              [-2 / 12., 8 / 12., -12 / 12., 8 / 12., -2 / 12.],
              [2 / 12., -6 / 12., 8 / 12., -6 / 12., 2 / 12.],
              [-1 / 12., 2 / 12., -2 / 12., 2 / 12., -1 / 12.]], dtype=np.float32) # (5, 5)
    for i in range(shape[2]):
        HPF_kernel[:, :, i, 0] = kn

    return HPF_kernel

# xu_Net
def xu_Net(input_shape=(512, 512, 1)):
    input = Input(name='the_input', shape=input_shape, dtype='float32')

    # HPH Layer
    x = Conv2D(1, kernel_size=(5, 5), padding='same', strides=(1, 1), kernel_initializer=init_HPF, trainable=False)(input)

    # group 1
    x = Conv2d_BN(x, nb_filter=8, kernel_size=(5, 5), isABS=1, isTanh=1, pool_size=(5, 5), pool_strides=(2, 2))
    # group 2
    x = Conv2d_BN(x, nb_filter=16, kernel_size=(5, 5), isTanh=1, pool_size=(5, 5), pool_strides=(2, 2))
    # group 3
    x = Conv2d_BN(x, nb_filter=32, kernel_size=(1, 1), pool_size=(5, 5), pool_strides=(2, 2))
    # group 4
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(1, 1), pool_size=(5, 5), pool_strides=(2, 2))
    # group 5
    x = Conv2d_BN(x, nb_filter=128, kernel_size=(1, 1), pool_size=(5, 5), pool_strides=(2, 2))

    # fully connected
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[input], outputs=[x])
    model.summary()

    return model

# DCT Keras
def init_DCT(shape, dtype=None):
    PI = math.pi
    DCT_kernel = np.zeros(shape, dtype=np.float32) # [height,width,input,output], shape=(4, 4, 1, 16)
    u = np.ones([4], dtype=np.float32) * math.sqrt(2.0 / 4.0)
    u[0] = math.sqrt(1.0 / 4.0)
    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                for l in range(0, 4):
                    DCT_kernel[i, j, :, k * 4 + l] = u[k] * u[l] * math.cos(PI / 8.0 * k * (2 * i + 1)) * math.cos(
                        PI / 8.0 * l * (2 * j + 1))

    return DCT_kernel

# DCT Keras, 能够适应各种shape
def init_DCT_new(shape, dtype=None):
    PI = math.pi
    DCT_kernel = np.zeros(shape, dtype=np.float32) # [height,width,input,output], shape=(4, 4, 1, 16)
    u = np.ones([shape[0]], dtype=np.float32) * math.sqrt(2.0 / 4.0)
    u[0] = math.sqrt(1.0 / 4.0)
    for i in range(0, shape[0]):
        for j in range(0, shape[0]):
            for k in range(0, shape[0]):
                for l in range(0, shape[0]):
                    DCT_kernel[i, j, :, k * shape[0] + l] = u[k] * u[l] * math.cos(PI / (2.0 * shape[0]) * k * (2 * i + 1)) * math.cos(
                        PI / (2.0 * shape[0]) * l * (2 * j + 1))

    return DCT_kernel

# Trancation operation for DCT
def DCT_Trunc(x):
    trunc = -(K.relu(-x + 8) - 8)
    return trunc

# DCT layer
def DCT_layer(x, conv_filters=16, kernel_size=(4, 4), padding='same', strides=(1, 1)):
    x = Conv2D(filters=conv_filters, kernel_size=kernel_size, padding=padding, strides=strides, trainable=False, kernel_initializer=init_DCT_new)(x)
    x = Lambda(ABS)(x) # 借用Lambda层
    x = Lambda(DCT_Trunc)(x)

    return x

# shortcut connection
def Conv_block(x, conv1_filters, conv2_filters,  kernel_size=(3, 3), padding='same', is_direct=1, isDropout=True):
    if is_direct:
        # without subsampling
        x1 = Conv2D(filters=conv1_filters, kernel_size=kernel_size, padding=padding, strides=(1, 1))(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation(activation='relu')(x1)
        x1 = Conv2D(filters=conv2_filters, kernel_size=kernel_size, padding=padding, strides=(1, 1))(x1)
        x1 = BatchNormalization()(x1)

        x = add([x, x1])
    else:
        # with subsampling
        x1 = Conv2D(filters=conv1_filters, kernel_size=kernel_size, padding=padding, strides=(1, 1))(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation(activation='relu')(x1)
        x1 = Conv2D(filters=conv2_filters, kernel_size=kernel_size, padding=padding, strides=(2, 2))(x1)
        x1 = BatchNormalization()(x1)

        x2 = Conv2D(filters=conv2_filters, kernel_size=kernel_size, padding=padding, strides=(2, 2))(x)
        x2 = BatchNormalization()(x2)

        x = add([x2, x1])

    x = Activation('relu')(x)
    if isDropout:
        x = Dropout(0.1)(x)
    return x

# xu_JNet
def xu_JNet(input_shape=(512, 512, 1)):
    input = Input(name='the_input', shape=input_shape, dtype='float32')
    # DCT
    x = DCT_layer(input, conv_filters=16, kernel_size=(4, 4))
    # block1
    x = Conv_block(x, conv1_filters=12, conv2_filters=24, is_direct=0)
    # # # block2
    x = Conv_block(x, conv1_filters=24, conv2_filters=24, is_direct=1)
    # block3
    x = Conv_block(x, conv1_filters=24, conv2_filters=48, is_direct=0)
    # block4
    x = Conv_block(x, conv1_filters=48, conv2_filters=48, is_direct=1)
    # block5
    x = Conv_block(x, conv1_filters=48, conv2_filters=96, is_direct=0)
    # block6
    x = Conv_block(x, conv1_filters=96, conv2_filters=96, is_direct=1)
    # block7
    x = Conv_block(x, conv1_filters=96, conv2_filters=192, is_direct=0, isDropout=True)
    # block8
    x = Conv_block(x, conv1_filters=192, conv2_filters=192, is_direct=1, isDropout=True)
    # block9
    x = Conv_block(x, conv1_filters=192, conv2_filters=384, is_direct=0, isDropout=True)
    # block10
    x = Conv_block(x, conv1_filters=384, conv2_filters=384, is_direct=1, isDropout=True)
    # block11
    # x = Conv_block(x, conv1_filters=384, conv2_filters=512, is_direct=0)
    # block12
    # x = Conv_block(x, conv1_filters=512, conv2_filters=512, is_direct=1)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[input], outputs=[x])
    model.summary()

    return model

def srm_init(shape, dtype=None):
    hpf = np.zeros(shape)
    kn = np.load('/data1/cby/py_project/ALASKA2/network/SRM_Kernels.npy')
    for i in range(shape[2]):
        hpf[:, :, i:i+1, :] = kn
    return hpf

# TLU激活函数
def TLU(x, tlu_threshold=3):
    tlu = K.clip(x, min_value=-tlu_threshold, max_value=tlu_threshold)
    return tlu

# ye_Net
def ye_Net(input_shape=(512, 512, 1), isReshape=False):
    if isReshape:
        input_shape = (int(input_shape[0] / 2), int(input_shape[1] / 2), int(input_shape[2] * 4))
    input = Input(name='the_input', shape=input_shape, dtype='float32')
    # Layer1 SRM preprocess
    x = Conv2D(30, kernel_size=(5, 5), padding='valid', strides=(1, 1), kernel_initializer=srm_init, trainable=False)(input)
    x = Lambda(TLU, arguments={'tlu_threshold': 3})(x)
    # Layer2
    x = Conv2D(filters=30, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    # Layer3
    x = Conv2D(filters=30, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    # Layer4
    x = Conv2D(filters=30, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2))(x)
    # Layer5
    x = Conv2D(filters=32, kernel_size=(5, 3), padding='valid', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2))(x)
    # Layer6
    x = Conv2D(filters=32, kernel_size=(5, 5), padding='valid', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2))(x)
    # Layer7
    x = Conv2D(filters=32, kernel_size=(5, 5), padding='valid', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), padding='valid', strides=(2, 2))(x)
    # Layer8
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    # Layer9
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    # Layer10
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[input], outputs=[x])
    model.summary()
    return model

# WiserNet
def WISERNet(input_shape=(1024, 1024, 3)):
    # 卷积计算并对层次命名
    def Conv2d_BN(x, nb_filter, kernel_size=(3, 3), padding='same', conv_strides=(1, 1), isANB=False, pool_size=(3, 3),
                  pool_strides=(2, 2)):
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=conv_strides)(x)
        if isANB:
            x = Lambda(ABS)(x)  # 借用Lambda层
        x = BatchNormalization(axis=-1, scale=False)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=pool_size, padding=padding, strides=pool_strides)(x)
        return x

    # 求绝对值
    def ABS(x):
        x = K.abs(x)
        return x

    def expand_dims(x, dim=0):
        x = K.expand_dims(x[:, :, :, dim], axis=-1)
        return x

    input = Input(shape=input_shape)

    inputR = Lambda(expand_dims, arguments={'dim': 0})(input)
    inputG = Lambda(expand_dims, arguments={'dim': 1})(input)
    inputB = Lambda(expand_dims, arguments={'dim': 2})(input)

    xR = Conv2D(filters=30, kernel_size=(5, 5), strides=(1, 1), kernel_initializer=srm_init, trainable=False )(inputR)
    xG = Conv2D(filters=30, kernel_size=(5, 5), strides=(1, 1), kernel_initializer=srm_init, trainable=False)(inputG)
    xB = Conv2D(filters=30, kernel_size=(5, 5), strides=(1, 1), kernel_initializer=srm_init, trainable=False)(inputB)
    x = concatenate([xR, xG, xB], axis=-1)

    x = Conv2d_BN(x, 72, kernel_size=(5, 5), conv_strides=(2, 2), isANB=True, pool_size=(3, 3), pool_strides=(2, 2))
    x = Conv2d_BN(x, 144, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(3, 3), pool_strides=(2, 2))
    x = Conv2d_BN(x, 288, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(5, 5), pool_strides=(4, 4))
    x = Conv2d_BN(x, 1152, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(16, 16), pool_strides=(32, 32))

    x = Flatten()(x)
    x = Dense(800, activation='relu')(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[input], outputs=[x])
    model.summary()
    return model




if __name__ == '__main__':
    # print(init_DCT(shape=(4, 4, 3, 16))[0, 0, 0, 0:16])
    # print(init_DCT_new(shape=(8, 8, 3, 64)))
    # SRM_Kernels = np.load('SRM_Kernels.npy')
    # print(SRM_Kernels.shape)
    # SRNet(input_shape=(512, 512, 3))
    # xu_Net(input_shape=(1024, 1024, 3))
    xu_JNet(input_shape=(512, 512, 3))
    # ye_Net(input_shape=(1024, 1024, 1), isReshape=False)
    # ye_JNet()
    # WISERNet(input_shape=(1024, 1024, 3))

    pass