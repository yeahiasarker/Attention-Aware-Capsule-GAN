import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, Flatten
from keras.layers import Activation, Multiply, Input, Lambda
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from attention import self_attention


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / \
        K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


def discriminator_func(shape):
    img = Input(shape=(shape[0], shape[1], shape[2]))
    x = Conv2D(
        filters=16,
        kernel_size=3,
        strides=1,
        kernel_initializer=init,
        name='conv1')(img)
    x = SelfAttention(ch=16)(x)
    x = LeakyReLU()(x)
    x = Conv2D(
        filters=16,
        kernel_size=3,
        strides=1,
        kernel_initializer=init,
        name='primarycap_conv2')(x)
    x = SelfAttention(ch=16)(x)
    x = LeakyReLU()(x)

    x = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(x)
    x = Lambda(squash, name='primarycap_squash')(x)

    x = Flatten()(x)
    uhat = Dense(
        128,
        kernel_initializer=init,
        bias_initializer='zeros',
        name='uhat_digitcaps')(x)
    c = Activation('softmax', name='softmax_digitcaps1')(uhat)
    c = Dense(128)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    c = Activation('softmax', name='softmax_digitcaps2')(s_j)
    c = Dense(128)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    c = Activation('softmax', name='softmax_digitcaps3')(s_j)
    c = Dense(128)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    pred = Dense(1, activation='tanh')(s_j)
    return Model(img, pred)
