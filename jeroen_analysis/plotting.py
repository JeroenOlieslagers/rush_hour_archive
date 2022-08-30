import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def custom_cmap(n, m):
    """
    Create custom colormap for subject x puzzle instance matrix plot
    """
    surrender = [0, 1, 0, 1]
    zero = [1, 1, 1, 1]
    block = mpl.cm.plasma(np.linspace(0, 1, m))
    upper = block[-1, :]*np.ones((n-m, 4))

    # combine parts of colormap
    cmap = np.vstack((surrender, zero, block, upper))

    # convert to matplotlib colormap
    cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
    return cmap


def puzzle_matshow(dfs, opt_len_dict):
    workers = dfs['worker'].unique()
    puzzles = dfs.sort_values('opt_len')['instance'].unique()

    M = np.zeros((len(workers), len(puzzles)))
    m = np.zeros((1, len(puzzles)))

    for n, worker in enumerate(workers):
        df = dfs[dfs['worker'] == worker]
        wins = df[df['event'] == 'win'][1::2]
        surrenders = df[df['event'] == 'surrender']

        for index, win in wins.iterrows():
            idx = np.where(win['instance'] == puzzles)[0][0]
            M[n, idx] = int(win['move_nu'])

        for index, surrender in surrenders.iterrows():
            idx = np.where(surrender['instance'] == puzzles)[0][0]
            M[n, idx] = -1
    
    for n, puzzle in enumerate(puzzles):
        m[0, n] = opt_len_dict[puzzle] - 1

    # Remove puzzles that are not solved
    non_zero_cols = np.logical_not(np.all((M == 0), axis=0))
    M = M[:, non_zero_cols]
    m = m[:, non_zero_cols]
    # Plotting
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [51, 1]})
    axs[0].matshow(M, cmap=custom_cmap(int(np.max(M))))
    axs[1].matshow(m, cmap='plasma')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[0].set_xlabel('Puzzle number')
    axs[0].set_ylabel('Subject number')
    axs[1].set_xlabel('Optimal length')
    axs[0].xaxis.set_label_position('top')
    plt.tight_layout()
    plt.show()


def separate_block_puzzle_matshow(dfs, opt_len_dict):
    ANOMALIES = 30

    workers = dfs['worker'].unique()
    puzzles = dfs.sort_values('opt_len')['instance'].unique()

    opt_lengths = np.sort(dfs['opt_len'].unique())
    opt_counts = np.zeros(max(opt_lengths)+1)
    for puzzle in puzzles:
        opt_counts[opt_len_dict[puzzle]] += 1
    opt_counts = opt_counts[opt_counts != 0].astype(int)

    opt_lengths = np.delete(opt_lengths, 1)
    opt_counts = np.delete(opt_counts, 1)

    Ms = [np.zeros((len(workers), count)) for count in opt_counts]
    m = np.zeros((1, len(puzzles)))

    for i, opt_len in enumerate(opt_lengths):
        # Setect entries of optimal length of each block
        df = dfs[dfs['opt_len'] == opt_len]
        puzzles_block = df['instance'].unique()
        for n, worker in enumerate(workers):
            dff = df[df['worker'] == worker]
            wins = dff[dff['event'] == 'win'][1::2]
            surrenders = dff[dff['event'] == 'surrender']

            for index, win in wins.iterrows():
                idx = np.where(win['instance'] == puzzles_block)[0][0]
                Ms[i][n, idx] = int(win['move_nu'])

            for index, surrender in surrenders.iterrows():
                idx = np.where(surrender['instance'] == puzzles_block)[0][0]
                Ms[i][n, idx] = -1

    Mss = []
    opt_counts_dummy = []
    for M in Ms:
        non_zero_cols = np.logical_not(np.all((M == 0), axis=0))
        if non_zero_cols.any(): 
            Mss.append(M[:, non_zero_cols])
            opt_counts_dummy.append(sum(non_zero_cols))
    opt_counts = opt_counts_dummy

    fig, axs = plt.subplots(1, len(opt_counts), gridspec_kw={'width_ratios': opt_counts}, figsize=(10, 6))
    for n, ax in enumerate(axs):
        minn = int(np.sort(Mss[n].flatten())[-ANOMALIES])
        im = ax.matshow(Mss[n], cmap=custom_cmap(int(np.amax(Mss[n])), minn))#np.ptp(Ms[n])
        if n > 0:
            ax.set_yticks([])
            ax.set_xticks([0, opt_counts[n]-1], labels=[np.sum(opt_counts[:n])+1, np.sum(opt_counts[:n+1])])
        else:
            ax.set_yticks([0, 10, 20, 30, 40, 50, 60], labels=[1, 11, 21, 31, 41, 51, 61])
            ax.set_xticks([0, opt_counts[0]-1], labels=[1, opt_counts[0]])
        #ax.xaxis.set_ticks_position("bottom")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03*sum(opt_counts)/opt_counts[n])
        cbar.ax.set_ylim(opt_lengths[n]-1, minn)
        cbar.ax.set_yticks([opt_lengths[n]-1, minn])
        ax.set_title('Optimal='+str(int(opt_lengths[n]-1)), y=-0.08)
        if n == 0:
            ax.set_ylabel('Subject number')
        ax.set_xlabel('Puzzle number')
        ax.xaxis.set_label_position('top') 
    plt.tight_layout()
    plt.show()