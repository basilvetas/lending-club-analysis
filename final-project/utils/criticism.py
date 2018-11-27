import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

sns.set_style('whitegrid')

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
      probs_next_state = mle_table.loc[current_state,:].values
      probs_next_state /= probs_next_state.sum() # fix in case rounding the MLE made the sum != 1 but close
      current_state = np.random.choice(list(mle_table.index), p=probs_next_state)
    else:
      break
  return pd.Series(chain)

def sample_and_plot_length(mle_table, true_data, n_samples=10000):
  """ plots sampled lengths """
  sampled_trajectories = [sample_mle(mle_table, add_done=False) for _ in range(n_samples)]
  done_idx = [i for i,k in enumerate(mle_table.keys()) if k == "Done"][0]
  true_counts = (true_data != done_idx).sum(axis=1)
  sampled_counts = [t.shape[0] for t in sampled_trajectories]
  
  plt.figure(figsize=(15,8))
  plt.hist([true_counts, sampled_counts], bins=36, normed=True,
            label=['True loan lengths', 'Sampled loan lengths'])
  plt.legend()
  plt.xlabel('Length (months)')
  plt.ylabel('Number of loans (normalized)')
  plt.show()

  sampled_mean = np.mean(sampled_counts)
  true_mean = true_counts.mean()
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
  fig=plt.figure(figsize=(20, 10), dpi= 80, facecolor='w')
  nx.draw_spectral(G, with_labels=True, node_color='#d3d3d3')
  plt.show()

if __name__ == '__main__':

  series = pd.Series(['Current', 'Late', 'Default'])
  graph_trajectory(series)


