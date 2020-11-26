import tensorflow as tf
from layers import convolution, get_num_channels


class q_net():
	def __init__(self, scope):
		self.name = 'q_net'
		self.train_phase = tf.placeholder(tf.bool,name="train_phase_placeholder")
		self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
		self.scope = scope
	def base_CNN(self, state):
		with tf.variable_scope(self.scope):
			num_channels = get_num_channels(state)
			with tf.variable_scope('conv_1'):
				x = convolution(state, [4,4,num_channels,8], strides=2, padding="VALID")
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
				x = tf.nn.relu(x)
			with tf.variable_scope('conv_2'):
				x = convolution(x, [3,3,8,16], strides=2, padding="VALID")
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
				x = tf.nn.relu(x)
			with tf.variable_scope('conv_3'):
				x = convolution(x, [3,3,16,32], strides=2, padding="VALID")
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
				x = tf.nn.relu(x)
			with tf.variable_scope('conv_4'):
				x = convolution(x, [3,3,32,64], strides=2, padding="VALID")
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
				x = tf.nn.relu(x)
			with tf.variable_scope('conv_5'):
				x = convolution(x, [3,3,64,128], strides=1, padding="VALID")
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
				x = tf.nn.relu(x)
			#with tf.variable_scope('conv_6'):
			#	x = convolution(x, [3,3,128,256], strides=1, padding="VALID")
			#	x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
			#	x = tf.nn.relu(x)
			x = tf.reduce_mean(x, axis=[1,2])
			x = tf.keras.layers.Dense(units=128)(x)
			#x = tf.nn.dropout(x, self.keep_prob)
			logits = tf.keras.layers.Dense(units=6)(x)
			return logits


