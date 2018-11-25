import numpy as np
import matplotlib.pyplot as plt


def sample_mle(mle_table, length=36, initial_state='Issued'):
    """
    Given a table of MLE estimates for the Markov Chain,
    samples states until length is reached or Charged Off or Fully Paid is reached,
    starting from 'Issued'.
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
    return chain

def plot_sampled_lengths(sampled_trajectories, true_data):
    plt.hist(true_data.groupby('id').size(), bins=36, normed=True, label='True loan lengths');
    plt.hist([len(t) for t in sampled_trajectories], bins=36, normed=True, label='Sampled loan lengths');
    plt.legend()
    plt.xlabel('Length (months)')
    plt.ylabel('Number of loans (normalized)')
    plt.show()
    
    sampled_mean = np.mean([len(t) for t in sampled_trajectories])
    true_mean = true_data.groupby('id').size().mean()
    print("Average length of sampled loans: %.2f months" % (sampled_mean,))
    print("Average length of true loans: %.2f months" % (true_mean,))