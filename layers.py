
from keras.layers import Layer
from keras import backend as k


class mini_batch_stdev(Layer):
	def __init__(self, **kwargs):
		super(mini_batch_stdev, self).__init__(**kwargs)
 
	def call(self, inputs):
		mean = k.mean(inputs, axis = 0, keepdims = True)

		squ_diffs = k.square(inputs - mean)

		mean_sq_diff = k.mean(squ_diffs, axis = 0, keepdims = True)

		mean_sq_diff += 1e-8

		stdev = k.sqrt(mean_sq_diff)

		mean_pix = k.mean(stdev, keepdims = True)

		shape = k.shape(inputs)

		output = k.tile(mean_pix, (shape[0], shape[1], shape[2], 1))

		combined = k.concatenate([inputs, output], axis = -1)

		return combined
 
	def compute_output_shape(self, input_shape):

		input_shape = list(input_shape)

		input_shape[-1] += 1

		return tuple(input_shape)


class pixel_normalization(Layer):

	def __init__(self, **kwargs):
		super(pixel_normalization, self).__init__(**kwargs)
 
	def call(self, inputs):

		values = inputs**2.0

		mean_values = k.mean(values, axis=-1, keepdims=True)

		mean_values += 1.0e-8

		l2 = k.sqrt(mean_values)

		normalized = inputs / l2

		return normalized
 
	def compute_output_shape(self, input_shape):
		return input_shape
