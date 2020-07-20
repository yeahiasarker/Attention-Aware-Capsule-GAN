import tensorflow as tf
from keras.layers import Conv2DTranspose, Dense, Reshape, Flatten
from keras.layers import Activation, Multiply, Input
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU


def discriminator_func():
    img = Input(shape=(shape[0], shape[1], shape[2]))
    x = Conv2D(
        filters=16,
        kernel_size=5,
        strides=1,
        kernel_initializer=init,
        name='conv1')(img)
    x = SelfAttention(ch=16)(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(
        filters=32,
        kernel_size=5,
        strides=1,
        padding='valid',
        kernel_initializer=init,
        name='primarycap_conv2')(x)
    x = SelfAttention(ch=32)(x)
    x = LeakyReLU()(x)
    x = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(x)
    x = Lambda(squash, name='primarycap_squash')(x)

    x = Flatten()(x)
    uhat = Dense(
        100,
        kernel_initializer=init,
        bias_initializer='zeros',
        name='uhat_digitcaps')(x)
    c = Activation('softmax', name='softmax_digitcaps1')(uhat)
    c = Dense(100)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    c = Activation('softmax', name='softmax_digitcaps2')(s_j)
    c = Dense(100)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    c = Activation('softmax', name='softmax_digitcaps3')(s_j)
    c = Dense(100)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    pred = Dense(1, activation='sigmoid')(s_j)
    return Model(img, pred)
