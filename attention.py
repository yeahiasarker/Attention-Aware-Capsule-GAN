from keras.layers import Layer, InputSpec
from keras import backend as K
from keras.initializers import RandomNormal



class self_attention(Layer):

    def __init__(self, ch, **kwargs):
        super(self_attention, self).__init__(**kwargs)
        self.channels = ch
        self.init = RandomNormal(stddev = 0.02)
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        self.gamma = self.add_weight(name = 'gamma', shape = [1], initializer = self.init, trainable = True)
        self.kernel_f = self.add_weight(shape = kernel_shape_f_g,
                                        initializer = self.init,
                                        name = 'kernel_f',
                                        trainable = True)
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer = self.init,
                                        name = 'kernel_g',
                                        trainable = True)
        self.kernel_h = self.add_weight(shape = kernel_shape_h,
                                        initializer = self.init,
                                        name = 'kernel_h',
                                        trainable = True)

        super(self_attention, self).build(input_shape)
        self.input_spec = InputSpec(ndim = 4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):

        def hw_flatten(x):
            return K.reshape(x, shape = [K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])

        f = K.conv2d(x,
                     kernel = self.kernel_f,
                     strides = (1, 1), padding = 'same')
        g = K.conv2d(x,
                     kernel = self.kernel_g,
                     strides = (1, 1), padding = 'same')
        h = K.conv2d(x,
                     kernel = self.kernel_h,
                     strides = (1, 1), padding = 'same')

        s = K.batch_dot(hw_flatten(g), K.permute_dimensions(hw_flatten(f), (0, 2, 1)))

        beta = K.softmax(s, axis = -1)

        o = K.batch_dot(beta, hw_flatten(h))

        o = K.reshape(o, shape = K.shape(x))

        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape
