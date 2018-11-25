from os.path import dirname, realpath, join
import numpy as np
import pandas as pd

import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, Categorical, Normal, Empirical, Dirichlet, Multinomial

from utils.utils import pretty_matrix, get_cache_or_execute

cache_path = join(dirname(realpath(__file__)), '../cache/')

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
	transition_matrix = matrix.values.astype(float)
	mle_matrix = np.empty_like(transition_matrix)

	for j in range(transition_matrix.shape[0]):
		# total transitions from state j to all other states
		j_sum = sum(transition_matrix[j,:])
		for k in range(transition_matrix.shape[1]):
			# total transitions from state j to state k
			j_to_k = transition_matrix[j,k]
			mle_matrix[j,k] = j_to_k/j_sum

	return pretty_matrix(pd.DataFrame(np.nan_to_num(mle_matrix)))


def infer_mc_no_priors(x_data, x, T, n_states, chain_len):
	""" runs variational inference given mc model without priors """
	def function(x_data, x, T, n_states, chain_len, **kwargs):
		sess = tf.Session() # create our own session
		# posteriors
		qx = kwargs['ed_model']['qx']

		inference = ed.KLqp(dict(zip(x, qx)), dict(zip(x, x_data)))

		saver = tf.train.Saver()
		inferred_matrix = pd.DataFrame() # placeholder

		# set sess as default but doesn't close it so we can re-use it later:
		with sess.as_default():
			inference.run(n_iter=5000)

			save_path = saver.save(sess, join(cache_path, 'experiment2.ckpt'))
			inferred_matrix = pd.DataFrame(sess.run(T))

		return pretty_matrix(inferred_matrix), sess, T, qx

	args = [x_data, x, T, n_states, chain_len]
	kwargs = {
	'format': 'table',
	'ed_model': {
			'T': T,
			'qx': [Categorical(
				probs=tf.nn.softmax(tf.Variable(tf.ones(n_states), name=f'qx_{i}'))) for i in range(chain_len)]
		}
	}
	return get_cache_or_execute('experiment2', function, *args, **kwargs)


def infer_mc_with_priors(x_data, x, pi_0, pi_T, n_states, chain_len, batch_size, n_samples=10, n_epoch=10, lr=0.005):
	""" runs variational inference given mc model with priors """
	def function(x_data, x, pi_0, pi_T, n_states, chain_len, batch_size, **kwargs):
		sess = tf.Session()
		data = generator(x_data, batch_size)

		n_batch = int(x_data.shape[0] / batch_size)

		qpi_0 = kwargs['ed_model']['qpi_0']
		qpi_T = kwargs['ed_model']['qpi_T']

		X = np.array([tf.placeholder(tf.int32, [batch_size]) for _ in range(chain_len)])

		inference = ed.KLqp({pi_0: qpi_0, pi_T: qpi_T}, data=dict(zip(x, X)))
		inference.initialize(n_iter=n_batch * n_epoch, n_samples=5, optimizer=tf.train.AdamOptimizer(0.005))

		saver = tf.train.Saver()
		inferred_matrix = pd.DataFrame() # placeholder

		# set sess as default but doesn't close it so we can re-use it later:
		with sess.as_default():
			sess.run(tf.global_variables_initializer())
			for _ in range(inference.n_iter):
				x_batch = next(data)
				info_dict = inference.update(dict(zip(X, x_batch.values.T)))
				inference.print_progress(info_dict)

			save_path = saver.save(sess, join(cache_path, 'experiment3.ckpt'))
			inferred_matrix = pd.DataFrame(sess.run(qpi_T.mean()))

		return pretty_matrix(inferred_matrix_mean), sess, qpi_0, qpi_T
	args = [x_data, x, pi_0, pi_T, n_states, chain_len, batch_size]
	kwargs = {
		'format': 'table',
		'ed_model': {
			'qpi_0': Dirichlet(tf.nn.softplus(tf.get_variable("qpi0/concentration", [n_states]))),
			'qpi_T': Dirichlet(tf.nn.softplus(tf.get_variable("qpiT/concentration", [n_states, n_states])))
		}
	}
	return get_cache_or_execute('experiment3', function, *args, **kwargs)


def infer_mc_with_priors_2(x_data, model, pi_0, pi_T, n_states, chain_len, batch_size, n_samples=10, n_epoch=1, lr=0.005):
    """ runs variational inference given mc model with priors """
    # TODO needs to be fixed to support caching
    sess = tf.Session()
    data = generator(x_data, batch_size)

    n_batch = int(x_data.shape[0] / batch_size)
    # n_epoch = 10
    # n_epoch = 1

    # qpi_0 = Dirichlet(tf.nn.softplus(tf.Variable(tf.ones(n_states))))
    # qpi_T = Dirichlet(tf.nn.softplus(tf.Variable(tf.ones([n_states, n_states]))))
    qpi_0 = Dirichlet(tf.nn.softplus(tf.get_variable("qpi0/concentration", [n_states])))
    qpi_T = Dirichlet(tf.nn.softplus(tf.get_variable("qpiT/concentration", [n_states, n_states])))

    X = tf.placeholder(tf.int32, [batch_size, chain_len])

    # inference = ed.KLqp({pi_0: qpi_0, pi_T: qpi_T}, data={model: X})
    inference = ed.KLqp({pi_0: qpi_0,
                         pi_T: qpi_T},
                        data={model: X})
    inference.initialize(n_iter=n_batch * n_epoch,
    					 n_samples=n_samples,
    					 optimizer=tf.train.AdamOptimizer(lr))
    					 # scale={model: x_data.shape[0]/batch_size}) # doesn't seem to help
    # inference.initialize(n_iter=n_batch * n_epoch, n_samples=100)
    inferred_matrix_mean = pd.DataFrame() # placeholder
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        for _ in range(inference.n_iter):
            x_batch = next(data)
            info_dict = inference.update({X: x_batch.values})
            inference.print_progress(info_dict)

        inferred_matrix_mean = pd.DataFrame(sess.run(qpi_T.mean()))

    return inferred_matrix_mean, sess, qpi_0, qpi_T	
