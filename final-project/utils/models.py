from os.path import dirname, realpath, join, exists
from timeit import default_timer as timer
import pandas as pd

import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, Categorical, Normal, Empirical, Multinomial

from utils.utils import reset_axes

cache_path = join(dirname(realpath(__file__)), '../cache/')


def build_mle_matrix(df):
	""" return transition matrix based on loan term and age of loan """
	transition_matrix_cache = join(cache_path, 'transition_matrix.hdf')

	start = timer()
	if not exists(transition_matrix_cache):
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

		transition_matrix = reset_axes(transition_matrix)

		print(f'Caching...')
		with pd.HDFStore(transition_matrix_cache, mode='w') as store:
			store.append('matrix', transition_matrix, format='table')

		print(f'Building transition matrix took {timer() - start:.2f} seconds')
	else:
		print(f'Loading transition matrix from hdf5 cache...')
		transition_matrix = pd.read_hdf(transition_matrix_cache, 'matrix')
		print(f'Fetching transition matrix took {timer() - start:.2f} seconds')

	return transition_matrix


def build_markov_chain_no_priors(n_states, chain_len):
	"""
	models a stationary markov chain in edward without priors
	for more see: https://github.com/blei-lab/edward/issues/450
	"""

	# create x_0, a default beginning state probability
	# vector with equal probabilties for each state
	p = tf.fill([n_states], 1.0 / n_states)
	x_0 = Categorical(probs=p)

	# create transition matrix for all other transitions after x_0
	T = tf.nn.softmax(tf.Variable(tf.random_uniform([n_states, n_states])), axis=0)

	# model the chain
	x = []
	for _ in range(chain_len):
		x_tm1 = x[-1] if x else x_0
		x_t = Categorical(probs=T[x_tm1, :])
		x.append(x_t)

	return x, T
