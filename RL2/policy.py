import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils import *

class LSTMPolicy:
    def __init__(self,
	    scope,
		inputs,
		action_dim,
		hidden_dim=256,
		activation=tf.nn.leaky_relu):

		# parameters
		self.scope = scope
		self.hidden_dim = hidden_dim
		self.activation_fn = activation
		self.action_dim = action_dim

		with tf.variable_scope(self.scope):
			# initialize lstm cell
			gru = tf.contrib.rnn.GRUCell(self.hidden_dim, activation=self.activation_fn)
			self.h_init = np.zeros((1, gru.state_size), np.float32)

			self.h_in = tf.placeholder(tf.float32, [1, gru.state_size])

			lstm_out, lstm_state = self.gru_unit(input=inputs,
											hold=self.h_in,
											activation=self.activation_fn)

			# extract lstm internal state
			lstm_h = lstm_state
			self.h_out = lstm_h[:1, :]
			lstm_out_flat = tf.reshape(lstm_out, [-1, self.hidden_dim])

			# policy and value function
			self._pi = layers.fully_connected(lstm_out_flat, num_outputs=self.action_dim, activation_fn=None)
			self._v = layers.fully_connected(lstm_out_flat, num_outputs=1, activation_fn=None)
			self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

		# copy state
		self.prev_h = self.h_init.copy()


    def gru_unit(self, input, hold, activation=tf.nn.relu):
		Wz = tf.get_variable(name='Wz', shape=[self.hidden_dim + 11, self.hidden_dim],
							 initializer=tf.contrib.layers.xavier_initializer())
		Wr = tf.get_variable(name='Wr', shape=[self.hidden_dim + 11, self.hidden_dim],
							 initializer=tf.contrib.layers.xavier_initializer())
		W = tf.get_variable(name='W', shape=[self.hidden_dim + 11, self.hidden_dim],
							initializer=tf.contrib.layers.xavier_initializer())

		xin = tf.unstack(input, axis=0)
		output = []

		for x in xin:
			z = activation(
				tf.matmul(tf.concat([hold, x], axis=1), Wz))  # [1, hidden+ input] x [hidden+ input, hidden] = [1, hidden]
			r = activation(
				tf.matmul(tf.concat([hold, x], axis=1), Wr))  # [1, hidden+ input] x [hidden+ input, hidden] = [1, hidden]

			hh = activation(tf.matmul(tf.concat([r * hold, x], axis=0), W))  # [hidden] ?
			h = (1 - z) * hold + z * hh  # [hidden]
			hold = tf.copy(h)

			output.append(h)

		output = tf.stack(output)

		return output, output

    @property
    def recurrent(self):
        return True

    @property
    def V(self):
        return self._v

    @property
    def pi(self):
        return self._pi

    @property
    def variables(self):
        return self._variables

    def reset(self):
        self.prev_h = self.h_init.copy()

    def save(self, session, filename):
        pass