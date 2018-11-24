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

	return pretty_matrix(pd.DataFrame(np.round(transitions_mle, 2)))


def infer_mc_no_priors(x_data, x, T, n_states, chain_len):
	""" runs variational inference given mc model without priors """

	# posteriors
	qx = [Categorical(
		probs=tf.nn.softmax(tf.Variable(tf.ones(n_states)))) for _ in range(chain_len)]

	inference = ed.KLqp(dict(zip(x, qx)), dict(zip(x, x_data)))

	inferred_matrix = pd.DataFrame() # placeholder
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		inference.run(n_iter=20000)
		inferred_matrix = pd.DataFrame(sess.run(T))

	return pretty_matrix(inferred_matrix)


def infer_mc_with_priors(x_data, x, pi_0, pi_T, n_states, chain_len, batch_size):
	""" runs variational inference given mc model with priors """
	data = generator(x_data, batch_size)

	n_batch = int(x_data.shape[0] / batch_size)
	n_epoch = 100

	qpi_0 = Dirichlet(tf.nn.softplus(tf.Variable(tf.ones(n_states))))
	qpi_T = Dirichlet(tf.nn.softplus(tf.Variable(tf.ones([n_states, n_states]))))

	X = np.array([tf.placeholder(tf.int32, [batch_size]) for _ in range(chain_len)])

	inference = ed.KLqp({pi_0: qpi_0, pi_T: qpi_T}, data=dict(zip(x, X)))
	inference.initialize(n_iter=n_batch * n_epoch, n_samples=5, optimizer=tf.train.AdamOptimizer(0.005))

	inferred_matrix = pd.DataFrame() # placeholder
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for _ in range(inference.n_iter):
			x_batch = next(data)
			info_dict = inference.update(dict(zip(X, x_batch.values.T)))
			inference.print_progress(info_dict)

		inferred_matrix = pd.DataFrame(sess.run(pi_T))

	return pretty_matrix(inferred_matrix)

