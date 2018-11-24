from os.path import dirname, realpath, join, exists
from timeit import default_timer as timer
import pandas as pd

import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, Categorical, Normal, Empirical, Dirichlet, Multinomial

from utils.utils import pretty_matrix, get_cache_or_execute


def build_mle_matrix(df):
	""" return transition matrix based on loan term and age of loan """
	def function(df):
		print('Building transition matrix...')

		df['previous_month'] = df.age_of_loan - 1
		transitions =  pd.merge(df, df, left_on=['id', 'age_of_loan'], right_on=['id', 'previous_month'])
		transition_matrix = pd.crosstab(transitions['loan_status_x'], transitions['loan_status_y'])

		# if there were no transitions for given state, it will be missing so fill it in
		for i in range(df.loan_status.unique().shape[0]):
			if i not in transition_matrix.index:
				# if no row, create it and set to 0:
				print(f'Filling in empty row {i}...')
				transition_matrix.loc[i] = 0
			if i not in transition_matrix.columns:
				# if no column, create it and set to 0:
				print(f'Filling in empty column {i}...')
				transition_matrix[i] = 0

		return pretty_matrix(transition_matrix)

	kwargs = { 'format': 'table' }
	return get_cache_or_execute('transitions', function, df, **kwargs)


def build_mc_no_priors(n_states, chain_len):
	"""
	models a stationary markov chain in edward without priors
	for more see: https://github.com/blei-lab/edward/issues/450
	"""
	tf.reset_default_graph()

	# create x_0, a default beginning state probability
	# vector with equal probabilties for each state
	p = tf.fill([n_states], 1.0 / n_states)
	x_0 = Categorical(probs=p)

	# create transition matrix for all other transitions after x_0
	T = tf.nn.softmax(tf.Variable(tf.random_uniform([n_states, n_states])), axis=0)

	# model the chain priors
	x = []
	for _ in range(chain_len):
		x_tm1 = x[-1] if x else x_0
		x_t = Categorical(probs=T[x_tm1, :])
		x.append(x_t)

	return x, T

def build_mc_with_priors(n_states, chain_len, batch_size):
	""" models a stationary markov chain in edward with Dirichlet priors """
	tf.reset_default_graph()

	# create default starting state probability vector
	pi_0 = Dirichlet(tf.ones(n_states))
	x_0 = Categorical(pi_0, sample_shape=batch_size)

	# transition matrix
	pi_T = Dirichlet(tf.ones([n_states, n_states]))

	x = []
	for _ in range(chain_len):
		x_tm1 = x[-1] if x else x_0
		x_t = Categorical(probs=tf.gather(pi_T, x_tm1))
		x.append(x_t)

	return x, pi_0, pi_T
