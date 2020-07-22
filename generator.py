from keras.layers import Conv2D, Conv2DTranspose, Dense
from keras.layers import Dense, Reshape, GaussianNoise, GaussianDropout
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from attention import self_attention
from layers import pixel_normalization


def generator_func(latent_dim):

    init = RandomNormal(stddev=0.02)
    model = Sequential()
    n_nodes = 256 * 8 * 8
    model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 256)))
    model.add(GaussianNoise(0.3))
    model.add(
        Conv2DTranspose(
            16, (1, 1), strides=(
                2, 2), padding='same', kernel_initializer=init))
    model.add(SelfAttention(ch=16))
    model.add(PixelNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2DTranspose(
            16, (3, 3), strides=(
                2, 2), padding='same', kernel_initializer=init))
    model.add(SelfAttention(ch=16))
    model.add(PixelNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2D(3,(3,3),
            activation='tanh',
            padding='same',
            kernel_initializer=init))
    return model
