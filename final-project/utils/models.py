import pandas as pd
import numpy as np

import tensorflow as tf
from edward.models import Categorical, Dirichlet, Multinomial

from utils.utils import pretty_matrix, get_cache_or_execute
from utils.tf.tf_hidden_markov_model import HiddenMarkovModel


def build_mle_matrix(df):
    """ return transition matrix based on loan term and age of loan """
    def function(df):
        print('Building transition matrix...')

        df = df.T.melt().reset_index()
        df.rename({'index': 'age_of_loan', 'value': 'loan_status'},
                  axis=1, inplace=True)
        df['previous_month'] = df.age_of_loan - 1
        transitions = pd.merge(df, df,
                               left_on=['id', 'age_of_loan'],
                               right_on=['id', 'previous_month'])
        transition_matrix = pd.crosstab(transitions['loan_status_x'],
                                        transitions['loan_status_y'])

        # if there were no transitions for given state,
        # it will be missing so fill it in
        for i in range(df.loan_status.unique().shape[0]):
            if i not in transition_matrix.index:
                # if no row, create it and set to 0:
                print(f'Filling in empty row {i}...')
                transition_matrix.loc[i] = 0
            if i not in transition_matrix.columns:
                # if no column, create it and set to 0:
                print(f'Filling in empty column {i}...')
                transition_matrix[i] = 0

        return pretty_matrix(transition_matrix), None

    kwargs = {'format': 'table'}
    return get_cache_or_execute('transitions', function, df, **kwargs)[0]


def model_stationary_dirichlet_categorical_edward(n_states,
                                                  chain_len,
                                                  batch_size):
    """ Models a stationary Dirichlet-Categorical Markov Chain in Edward """
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


def model_stationary_dirichlet_categorical_tfp(n_states,
                                               chain_len,
                                               batch_size):
    """ Models a stationary Dirichlet-Categorical Markov Chain
    in TensorFlow Probability/Edward2 """
    tf.reset_default_graph()

    # create default starting state probability vector
    pi_0 = Dirichlet(tf.ones(n_states))
    x_0 = Categorical(pi_0, sample_shape=batch_size)

    # transition matrix
    pi_T = Dirichlet(tf.ones([n_states, n_states]))
    transition_distribution = Categorical(probs=pi_T)

    pi_E = np.eye(n_states, dtype=np.float32)  # identity matrix
    emission_distribution = Categorical(probs=pi_E)

    model = HiddenMarkovModel(
        initial_distribution=x_0,
        transition_distribution=transition_distribution,
        observation_distribution=emission_distribution,
        num_steps=chain_len,
        sample_shape=batch_size)

    return model, pi_0, pi_T


def model_non_stationary_dirichlet_categorical(n_states,
                                               chain_len,
                                               batch_size):
    """ Models a non-stationary Dirichlet-Categorical
    Markov Chain in Edward """
    tf.reset_default_graph()

    # create default starting state probability vector
    pi_0 = Dirichlet(tf.ones(n_states))
    x_0 = Categorical(pi_0, sample_shape=batch_size)

    pi_T, x = [], []
    for _ in range(chain_len):
        x_tm1 = x[-1] if x else x_0
        # transition matrix, one per position in the chain:
        # i.e. we now condition both on previous state and age of the loan
        pi_T_t = Dirichlet(tf.ones([n_states, n_states]))
        x_t = Categorical(probs=tf.gather(pi_T_t, x_tm1))
        pi_T.append(pi_T_t)
        x.append(x_t)

    return x, pi_0, pi_T


def model_stationary_dirichlet_multinomial(n_states, chain_len,
                                           total_counts_per_month):
    """ Models a stationary Dirichlet-Multinomial Markov Chain in Edward """
    tf.reset_default_graph()

    # create default starting state probability vector
    pi_list = [Dirichlet(tf.ones(n_states)) for _ in range(chain_len)]
    # now instead of sample_shape we use total_count which
    # is how many times we sample from each categorical
    # i.e. number of accounts
    counts = [Multinomial(probs=pi, total_count=float(total_counts_per_month))
              for pi in pi_list]

    return pi_list, counts
