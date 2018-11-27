import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

sns.set_style('white')

def sample_mle(mle_table, length=36, initial_state='Current'):
  """
  Given a table of MLE estimates for the Markov Chain, samples
  states until length is reached or Charged Off or Fully Paid is reached, start from 'Current'
  """
  chain = list()
  current_state = initial_state
  for i in range(36):
    chain.append(current_state)
    if current_state not in ['Charged Off', 'Fully Paid']:
      probs_next_state = mle_table.loc[current_state,:].values
      probs_next_state /= probs_next_state.sum() # fix in case rounding the MLE made the sum != 1 but close
      current_state = np.random.choice(list(mle_table.index), p=probs_next_state)
    else:
      break
  return pd.Series(chain)

def plot_sampled_lengths(sampled_trajectories, true_data):
  """ plots sampled lengths """
  plt.hist(true_data.groupby('id').size(), bins=36, normed=True, label='True loan lengths');
  plt.hist([t.shape[0] for t in sampled_trajectories], bins=36, normed=True, label='Sampled loan lengths');
  plt.legend()
  plt.xlabel('Length (months)')
  plt.ylabel('Number of loans (normalized)')
  plt.show()

  sampled_mean = np.mean([t.shape[0] for t in sampled_trajectories])
  true_mean = true_data.groupby('id').size().mean()
  print(f'Average length of sampled loans: {sampled_mean:.2f} months')
  print(f'Average length of true loans: {true_mean:.2f} months')

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
  df = df.iloc[1:] # drop first row with nan
  G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.DiGraph())
  plt.figure(figsize=(20, 10), dpi= 80, facecolor='w')
  nx.draw_spectral(G, with_labels=True, node_color='#d3d3d3')
  plt.show()

def plot_probs_from_state_j(matrices, state_j, states_k=None):
  """ plots the probabilities of transitioning from state_j to all states_k across each time step """

  if states_k is None:
    states_k = ['Charged Off', 'Current', 'Default', 'Fully Paid', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']

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
  plt.figure(figsize=(10, 10), dpi= 80, facecolor='w')
  ax = sns.lineplot(x='month', y='probability', hue='state', data=df)
  ax.set_title(f'Estimated Transition Probability From State: {state_j}')
  plt.show()

if __name__ == '__main__':

  series = pd.Series(['Current', 'Late', 'Default'])
  graph_trajectory(series)























