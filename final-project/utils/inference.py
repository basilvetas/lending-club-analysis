import numpy as np
import pandas as pd

import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, Categorical, Normal, Empirical, Dirichlet, Multinomial

from utils.utils import pretty_matrix, get_cache_or_execute

def generator(df, batch_size):
  """ Generate batches of data from df based on batch_size """
  starts = 0 # pointer to where we are in iteration
  while True:
    start = starts
    stop = start + batch_size
    diff = stop - df.shape[0]
    if diff <= 0:
      batch = df.iloc[start:stop]
      starts += batch_size
    else:
      batch = pd.concat((df.iloc[start:], df.iloc[:diff]))
      starts = diff

    yield batch

def compute_mle(matrix):
	""" computes the maximum likelihood estimate for a stationary
	transition matrix given a matrix of realized transitions data
	"""
	transitions_mle = matrix.values.astype(float)

	for i in range(transitions_mle.shape[0]):
		n_i_all = sum(transitions_mle[i,:]) # count how many i => j for this i and any j
		if n_i_all != 0:
			transitions_mle[i,:] *= (1/n_i_all)

	return pretty_matrix(pd.DataFrame(transitions_mle))


def infer_mc_no_priors(x_data, x, T, n_states, chain_len):
	""" runs variational inference given mc model without priors """

	def function(x_data, x, T, n_states, chain_len):
		sess = tf.Session() # create our own session
		# posteriors
		qx = [Categorical(
			probs=tf.nn.softmax(tf.Variable(tf.ones(n_states)))) for _ in range(chain_len)]

		inference = ed.KLqp(dict(zip(x, qx)), dict(zip(x, x_data)))

		inferred_matrix = pd.DataFrame() # placeholder
		 # set sess as default but doesn't close it so we can re-use it later:
		with sess.as_default():
			# sess.run(tf.global_variables_initializer())

			# inference.run(n_iter=20000)
			inference.run(n_iter=5000)
			inferred_matrix = pd.DataFrame(sess.run(T))

		return pretty_matrix(inferred_matrix), sess, qx

	# args = [x_data, x, T, n_states, chain_len]
	# kwargs = { 'format': 'table' }
	# return get_cache_or_execute('experiment2', function, *args, **kwargs)
	return function(x_data, x, T, n_states, chain_len)


def infer_mc_with_priors(x_data, x, pi_0, pi_T, n_states, chain_len, batch_size):
	""" runs variational inference given mc model with priors """

	def function(x_data, x, pi_0, pi_T, n_states, chain_len, batch_size):
		sess = tf.Session()
		data = generator(x_data, batch_size)

		n_batch = int(x_data.shape[0] / batch_size)
		n_epoch = 10

		qpi_0 = Dirichlet(tf.nn.softplus(tf.Variable(tf.ones(n_states))))
		qpi_T = Dirichlet(tf.nn.softplus(tf.Variable(tf.ones([n_states, n_states]))))

		X = np.array([tf.placeholder(tf.int32, [batch_size]) for _ in range(chain_len)])

		inference = ed.KLqp({pi_0: qpi_0, pi_T: qpi_T}, data=dict(zip(x, X)))
		inference.initialize(n_iter=n_batch * n_epoch, n_samples=5, optimizer=tf.train.AdamOptimizer(0.005))

		inferred_matrix = pd.DataFrame() # placeholder
		with sess.as_default():
			sess.run(tf.global_variables_initializer())
			for _ in range(inference.n_iter):
				x_batch = next(data)
				info_dict = inference.update(dict(zip(X, x_batch.values.T)))
				inference.print_progress(info_dict)

			inferred_matrix_mean = pd.DataFrame(sess.run(pi_T.mean()))

		return pretty_matrix(inferred_matrix), sess, qpi_0, qpi_T

	# args = [x_data, x, pi_0, pi_T, n_states, chain_len, batch_size]
	# kwargs = { 'format': 'table' }
	# return get_cache_or_execute('experiment3', function, *args, **kwargs)
	return function(x_data, x, pi_0, pi_T, n_states, chain_len, batch_size)
