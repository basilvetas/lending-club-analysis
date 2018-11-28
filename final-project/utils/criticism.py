import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import tensorflow as tf

from edward.models import Categorical
from utils.tf.tf_hidden_markov_model import HiddenMarkovModel

sns.set_style('white')


def sample_mle(mle_table, length=36, initial_state='Current', add_done=True):
    """
    Given a table of MLE estimates for the Markov Chain, samples
    states until length is reached or Done is reached, start from 'Current'
    """
    chain = list()
    current_state = initial_state
    for i in range(length):
        if current_state != "Done" or add_done:
            chain.append(current_state)
        if current_state != "Done":
            probs_next_state = mle_table.loc[current_state, :].values
            # fix in case rounding the MLE made the sum != 1 but close
            probs_next_state /= probs_next_state.sum()
            current_state = np.random.choice(
                list(mle_table.index), p=probs_next_state)
        else:
            break
    return pd.Series(chain)


def plot_length(true_counts, sampled_counts):
    plt.figure(figsize=(15, 8))
    plt.hist([true_counts, sampled_counts], bins=36,
             normed=True, label=['True loan lengths', 'Sampled loan lengths'])
    plt.legend()
    plt.xlabel('Length (months)')
    plt.ylabel('Number of loans (normalized)')
    plt.show()

    sampled_mean = np.mean(sampled_counts)
    true_mean = true_counts.mean()
    print(f'Average length of sampled loans: {sampled_mean:.2f} months')
    print(f'Average length of true loans: {true_mean:.2f} months')


def sample_and_plot_length(mle_table, true_data, n_samples=10000):
    """ plots sampled lengths """
    sampled_trajectories = [sample_mle(mle_table, add_done=False)
                            for _ in range(n_samples)]
    done_idx = [i for i, k in enumerate(mle_table.keys()) if k == "Done"][0]
    true_counts = (true_data != done_idx).sum(axis=1)
    sampled_counts = [t.shape[0] for t in sampled_trajectories]

    plot_length(true_counts, sampled_counts)


def graph_trajectory(trajectory):
    """ plots unique state changes for input trajectory """
    source = trajectory.shift()
    target = trajectory

    # add subscripts so transitions are unique
    for i, r in source.iteritems():
        if r is not np.nan:
            source[i] = f'{r}$_{{{i-1}}}$'

    for i, r in target.iteritems():
        target[i] = f'{r}$_{{{i}}}$'

    df = pd.concat([source, target], axis=1, ).reset_index(drop=True)
    df.columns = ['source', 'target']
    df = df.iloc[1:]  # drop first row with nan
    G = nx.from_pandas_edgelist(df, 'source', 'target',
                                create_using=nx.DiGraph())
    plt.figure(figsize=(20, 10), dpi=80, facecolor='w')
    nx.draw_spectral(G, with_labels=True, node_color='#d3d3d3')
    plt.show()


def plot_probs_from_state_j(matrices, state_j, states_k=None):
    """
    plots the probabilities of transitioning
    from state_j to all states_k across each time step
    """
    if states_k is None:
        states_k = ['Charged Off',
                    'Current',
                    'Default',
                    'Fully Paid',
                    'In Grace Period',
                    'Late (16-30 days)',
                    'Late (31-120 days)']

    data = []
    for month_i in range(len(matrices)):
        for state_k, prob_jk in matrices[month_i].T[state_j].iteritems():
            if (state_k in states_k):
                data.append({
                    'month': month_i,
                    'state': state_k,
                    'probability': prob_jk
                })

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 10), dpi=80, facecolor='w')
    ax = sns.lineplot(x='month', y='probability', hue='state', data=df)
    ax.set_title(f'Estimated Transition Probability From State: {state_j}')
    plt.show()

# EXPERIMENT 2

def copy_model_ed(qpi_0, qpi_T, chain_len, n_states, batch_size):
    """
    Used in place of ed.copy as it seems like ed.copy doesn't take
    into account all the necessary dependencies in our graph.
    """
    x_0_post = Categorical(probs=qpi_0, sample_shape=batch_size)

    x_post = []
    for _ in range(chain_len):
        x_tm1 = x_post[-1] if x_post else x_0_post
        x_t = Categorical(probs=tf.gather(qpi_T, x_tm1))
        x_post.append(x_t)

    return x_0_post, x_post


def sample_ed(x_0_post, x_post, sess, inferred_matrix, n_samples=1, batch_size=1000):
    """ creates the posterior predictive and samples from it """
    with sess.as_default():
        samples = []
        current_state = sess.run(x_0_post.sample(batch_size))
        for i in range(1, len(x_post)):
            samples.append(current_state)
            current_state = sess.run(x_post[i].sample(), {x_post[i-1]: current_state})
    samples = np.array(samples).T
    
    def pretty_sample(s):
        pretty_s = []
        for k in s:
            if inferred_matrix.keys()[k] != "Done":
                pretty_s.append(inferred_matrix.keys()[k])
            else:
                break
        return pretty_s

    if n_samples == 1:
        return pd.Series(pretty_sample(samples[0]))
    else:
        return [pretty_sample(s) for s in samples]


def sample_and_plot_length_ed(x_0_post, x_post, sess, true_data,
                               inferred_matrix, n_samples=1000):
    """ plots sampled lengths """
    sampled_trajectories = sample_ed(x_0_post, x_post, sess,
                                      inferred_matrix, n_samples)
    done_idx = [i for i, k in enumerate(
                inferred_matrix.keys()) if k == "Done"][0]
    true_counts = (true_data != done_idx).sum(axis=1)
    sampled_counts = [len(t) for t in sampled_trajectories]

    plot_length(true_counts, sampled_counts)

# EXPERIMENT 3

def copy_model_tfp(qpi_0, qpi_T, chain_len, n_states, sample_shape):
    """
    Used in place of ed.copy as it seems like ed.copy doesn't take
    into account all the necessary dependencies in our graph.
    """
    x_0 = Categorical(probs=qpi_0, sample_shape=sample_shape)

    # transition matrix
    transition_distribution = Categorical(probs=qpi_T)

    pi_E = np.eye(n_states, dtype=np.float32)  # identity matrix
    emission_distribution = Categorical(probs=pi_E)

    model_post = HiddenMarkovModel(
            initial_distribution=x_0,
            transition_distribution=transition_distribution,
            observation_distribution=emission_distribution,
            num_steps=chain_len,
            sample_shape=sample_shape)

    return model_post


def sample_tfp(model_post, sess, inferred_matrix, n_samples=1):
    """ creates the posterior predictive and samples from it """
    with sess.as_default():
        samples = sess.run(model_post.sample(n_samples))

    def pretty_sample(s):
        pretty_s = []
        for k in s:
            if inferred_matrix.keys()[k] != "Done":
                pretty_s.append(inferred_matrix.keys()[k])
            else:
                break
        return pretty_s

    if n_samples == 1:
        return pd.Series(pretty_sample(samples[0]))
    else:
        return [pretty_sample(s) for s in samples]


def sample_and_plot_length_tfp(model_post, sess, true_data,
                               inferred_matrix, n_samples=10000):
    """ plots sampled lengths """
    sampled_trajectories = sample_tfp(model_post, sess,
                                      inferred_matrix, n_samples)
    done_idx = [i for i, k in enumerate(
                inferred_matrix.keys()) if k == "Done"][0]
    true_counts = (true_data != done_idx).sum(axis=1)
    sampled_counts = [len(t) for t in sampled_trajectories]

    plot_length(true_counts, sampled_counts)


# EXPERIMENT 5

def plot_multinomial_probs(inferred_matrix, states='all'):
    if states == 'all':
        states = list(inferred_matrix.columns)
    plt.figure(figsize=(15, 8))
    for s in states:
        # plt.plot(list(range(inferred_matrix.shape[0])), inferred_matrix[states], label=states)
        plt.plot(inferred_matrix[s], label=s)
    plt.legend()
    plt.xlabel('Age of loan (months)')
    plt.ylabel('Probability of each state')
    plt.show()

if __name__ == '__main__':
    series = pd.Series(['Current', 'Late', 'Default'])
    graph_trajectory(series)
