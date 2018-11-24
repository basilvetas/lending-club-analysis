import numpy as np
import pandas as pd

import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, Categorical, Normal, Empirical, Multinomial

from utils.utils import reset_axes


def compute_mle(matrix):
	""" computes the maximum likelihood estimate for a stationary
	transition matrix given a matrix of realized transitions data
	"""
	transitions_mle = matrix.values.astype(float)

	for i in range(transitions_mle.shape[0]):
		n_i_all = sum(transitions_mle[i,:]) # count how many i => j for this i and any j
		if n_i_all != 0:
			transitions_mle[i,:] *= (1/n_i_all)

	return reset_axes(pd.DataFrame(np.round(transitions_mle, 2)))


def infer_matrix_no_priors(x_data, x, T, n_states, chain_len, **kwargs):
	""" runs variational inference given model inputs """

	# posteriors
	qx = [Categorical(
		probs=tf.nn.softmax(tf.Variable(tf.ones(n_states)))) for _ in range(chain_len)]

	# placeholders
	inferred_matrix = pd.DataFrame()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# print('Before inference')
		# print(sess.run(T))
		inference = ed.KLqp(dict(zip(x, qx)), dict(zip(x, x_data)))
		inference.run(**kwargs)
		inferred_matrix = pd.DataFrame(sess.run(T))
		# print('#'*40)
		# print('After inference')
		# print(sess.run(T))
		# print('#'*40)
		# print('qx:')
		# pprint(sess.run([foo.probs for foo in qx]))
		# print('#'*40)
		# print('x:')
		# pprint(sess.run([foo.probs for foo in x]))

	return reset_axes(inferred_matrix)
