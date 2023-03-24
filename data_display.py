import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
# assume data is a numpy array of shape (1264, 5628, 24)
# with (n_sequences, seq_len, features)

n_plots = 8
features_per_plot = 3

dataset = np.load('data_shape(1208_3540_24).npy')
# dataset = torch.load('filename')
labels = np.load('labels_shape(1208_1).npy')

for seq_idx in range(dataset.shape[0]):

    # create subplots for the current sequence
    fig, axs = plt.subplots(n_plots, sharex=True)

    # plot the features in groups of 3 in each subplot
    for plot_idx in range(n_plots):
        start_feature_idx = plot_idx * features_per_plot
        end_feature_idx = start_feature_idx + features_per_plot
        features = dataset[seq_idx, :, start_feature_idx:end_feature_idx]
        axs[plot_idx].plot(features)
        axs[plot_idx].set_ylabel(f'Features {start_feature_idx}-{end_feature_idx-1}')

    fig.suptitle(f'{labels[seq_idx]}')
    # set x-axis label for the last subplot
    axs[-1].set_xlabel('seq_len')

    # show the plot
    plt.show()

    # close the plot to move on to the next sequence
    plt.close()
