from os.path import dirname, realpath, join
import numpy as np
import pandas as pd

import tensorflow as tf
import edward as ed
from edward.models import Dirichlet

from utils.utils import pretty_matrix, get_cache_or_execute

cache_path = join(dirname(realpath(__file__)), '../cache/')


def generator(df, batch_size):
    """ Generate batches of data from df based on batch_size """
    starts = 0  # pointer to where we are in iteration
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
        j_sum = sum(transition_matrix[j, :])
        for k in range(transition_matrix.shape[1]):
            # total transitions from state j to state k
            j_to_k = transition_matrix[j, k]
            mle_matrix[j, k] = j_to_k/j_sum

    return pretty_matrix(pd.DataFrame(np.nan_to_num(mle_matrix)))


def infer_stationary_dirichlet_categorical_edward(x_data, x, pi_0,
                                                  pi_T, n_states,
                                                  chain_len, batch_size,
                                                  n_samples=10,
                                                  n_epoch=5,
                                                  lr=0.005):
    """ runs variational inference given mc model with priors """
    def function(x_data, x, pi_0, pi_T, n_states,
                 chain_len, batch_size, **kwargs):
        sess = tf.Session()
        data = generator(x_data, batch_size)

        n_batch = int(x_data.shape[0] / batch_size)

        qpi_0 = kwargs['ed_model']['qpi_0']
        qpi_T = kwargs['ed_model']['qpi_T']

        X = np.array([tf.placeholder(tf.int32, [batch_size])
                     for _ in range(chain_len)])

        inference = ed.KLqp({pi_0: qpi_0, pi_T: qpi_T}, data=dict(zip(x, X)))
        inference.initialize(n_iter=n_batch * n_epoch, n_samples=5,
                             optimizer=tf.train.AdamOptimizer(lr))

        saver = tf.train.Saver()
        inferred_matrix = pd.DataFrame()  # placeholder

        # set sess as default but doesn't close it so we can re-use it later:
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            for _ in range(inference.n_iter):
                x_batch = next(data)
                info_dict = inference.update(dict(zip(X, x_batch.values.T)))
                inference.print_progress(info_dict)

            saver.save(sess, join(cache_path, 'experiment2.ckpt'))
            inferred_matrix = pd.DataFrame(sess.run(qpi_T.mean()))

        print()  # hack for printing new line
        return pretty_matrix(inferred_matrix), sess, qpi_0, qpi_T

    args = [x_data, x, pi_0, pi_T, n_states, chain_len, batch_size]
    kwargs = {
        'format': 'table',
        'ed_model': {
            'qpi_0': Dirichlet(tf.nn.softplus(
                tf.get_variable("qpi0/concentration", [n_states]))),
            'qpi_T': Dirichlet(tf.nn.softplus(
                tf.get_variable("qpiT/concentration", [n_states, n_states])))
        }
    }
    return get_cache_or_execute('experiment2', function, *args, **kwargs)


def infer_stationary_dirichlet_categorical_tfp(x_data, model, pi_0, pi_T,
                                               n_states, chain_len,
                                               batch_size, n_samples=10,
                                               n_epoch=1, lr=0.005):
    """ runs variational inference given mc model with priors """

    def function(x_data, x, pi_0, pi_T, n_states, chain_len,
                 batch_size, n_samples, n_epoch, lr, **kwargs):
        sess = tf.Session()
        data = generator(x_data, batch_size)

        n_batch = int(x_data.shape[0] / batch_size)

        qpi_0 = kwargs['ed_model']['qpi_0']
        qpi_T = kwargs['ed_model']['qpi_T']

        X = tf.placeholder(tf.int32, [batch_size, chain_len])

        inference = ed.KLqp({pi_0: qpi_0,
                            pi_T: qpi_T},
                            data={model: X})
        inference.initialize(n_iter=n_batch * n_epoch,
                             n_samples=n_samples,
                             optimizer=tf.train.AdamOptimizer(lr),
                             logdir='log/experiment3')

        saver = tf.train.Saver()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            for _ in range(inference.n_iter):
                x_batch = next(data)
                info_dict = inference.update({X: x_batch.values})
                inference.print_progress(info_dict)

            saver.save(sess, join(cache_path, 'experiment3.ckpt'))
            inferred_qpi_0, inferred_qpi_T = sess.run(
                [qpi_0.mean(), qpi_T.mean()])
            inferred_qpi_T = pretty_matrix(pd.DataFrame(inferred_qpi_T))
            inferred_qpi_0 = pd.DataFrame([inferred_qpi_0],
                                          index=["probs"],
                                          columns=inferred_qpi_T.columns)

        print()  # hack for printing new line
        return [inferred_qpi_0, inferred_qpi_T], sess, qpi_0, qpi_T

    args = [x_data, model, pi_0, pi_T, n_states, chain_len,
            batch_size, n_samples, n_epoch, lr]
    kwargs = {
        'format': 'table',
        'n_items': 2,
        'ed_model': {
            'qpi_0': Dirichlet(tf.nn.softplus(
                tf.get_variable("qpi0/concentration", [n_states]))),
            'qpi_T': Dirichlet(tf.nn.softplus(
                tf.get_variable("qpiT/concentration", [n_states, n_states])))
        }
    }
    return get_cache_or_execute('experiment3', function, *args, **kwargs)


def infer_non_stationary_dirichlet_categorical(x_data, x, pi_0, pi_T_list,
                                               n_states, chain_len,
                                               batch_size, n_samples=10,
                                               n_epoch=5, lr=0.005):
    """ runs variational inference given mc model with priors,
    conditioning on position in the chain """
    def function(x_data, x, pi_0, pi_T_list, n_states, chain_len,
                 batch_size, n_samples=10, n_epoch=5, lr=0.005, **kwargs):
        sess = tf.Session()
        data = generator(x_data, batch_size)

        n_batch = int(x_data.shape[0] / batch_size)

        qpi_0 = kwargs['ed_model']['qpi_0']
        qpi_T_list = kwargs['ed_model']['qpi_T_list']

        X = np.array([tf.placeholder(tf.int32, [batch_size])
                      for _ in range(chain_len)])

        latent_vars_map = {pi_0: qpi_0}
        latent_vars_map.update(dict(zip(pi_T_list, qpi_T_list)))
        inference = ed.KLqp(latent_vars_map, data=dict(zip(x, X)))
        inference.initialize(n_iter=n_batch * n_epoch,
                             n_samples=n_samples,
                             optimizer=tf.train.AdamOptimizer(lr))

        saver = tf.train.Saver()

        # set sess as default but doesn't close it so we can re-use it later:
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            for _ in range(inference.n_iter):
                x_batch = next(data)
                info_dict = inference.update(dict(zip(X, x_batch.values.T)))
                inference.print_progress(info_dict)

            saver.save(sess, join(cache_path, 'experiment4.ckpt'))
            inferred_matrices = [pd.DataFrame(sess.run(qpi_T.mean()))
                                 for qpi_T in qpi_T_list]

        print()  # hack for printing new line
        pretty_matrices = [pretty_matrix(inferred_matrix)
                           for inferred_matrix in inferred_matrices]
        return pretty_matrices, sess, qpi_0, qpi_T_list

    var_names = [f'qpiT_{i}/concentration_' for i in range(chain_len)]
    args = [x_data, x, pi_0, pi_T_list, n_states,
            chain_len, batch_size, n_samples, n_epoch, lr]
    kwargs = {
        'format': 'table',
        'n_items': chain_len,
        'ed_model': {
            'qpi_0': Dirichlet(tf.nn.softplus(
                tf.get_variable("qpi0/concentration", [n_states]))),
            'qpi_T_list': [Dirichlet(tf.nn.softplus(
                tf.get_variable(name, [n_states, n_states])))
                for name in var_names]
        }
    }
    return get_cache_or_execute('experiment4', function, *args, **kwargs)


def infer_stationary_dirichlet_multinomial(x_data, pi_list, counts,
                                           total_counts_per_month,
                                           n_states, chain_len, n_samples=10):
    """ runs variational inference given mc model with priors,
    conditioning on position in the chain """
    def function(x_data, pi_list, counts, total_counts_per_month,
                 n_states, chain_len, n_samples=10, **kwargs):
        sess = tf.Session()
        qpi_list = kwargs['ed_model']['qpi_list']
        saver = tf.train.Saver()

        # set sess as default but doesn't close it so we can re-use it later:
        with sess.as_default():
            inference = ed.KLqp(dict(zip(pi_list, qpi_list)),
                                data=dict(zip(counts, [x_data[i, :]
                                          for i in range(chain_len)])))

            inference.run(n_iter=3000)

            saver.save(sess, join(cache_path, 'experiment5.ckpt'))
            inferred_probs = [pd.DataFrame(sess.run(pi.mean()))
                              for pi in qpi_list]

        inferred_matrix = np.array([prob.values.reshape(-1)
                                   for prob in inferred_probs])
        inferred_matrix = pd.DataFrame(inferred_matrix)
        inferred_matrix = pretty_matrix(inferred_matrix)  # add column names
        # get rid of row names
        inferred_matrix = inferred_matrix.reset_index().drop('index', axis=1)

        print()  # hack for printing new line
        return inferred_matrix, sess, qpi_list

    var_names = [f'qpi_{i}/concentration_' for i in range(chain_len)]
    args = [x_data, pi_list, counts, total_counts_per_month,
            n_states, chain_len, n_samples]
    kwargs = {
        'format': 'table',
        'ed_model': {
            'qpi_list': [Dirichlet(1 + tf.nn.softplus(
                tf.get_variable(name, [n_states]))) for name in var_names]
        }

    }
    return get_cache_or_execute('experiment5', function, *args, **kwargs)
