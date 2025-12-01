# This script plots the PCA projections colored by loudness levels and words.
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
import os
from pathlib import Path
import pickle as pkl  

'''
Example cmd:
For t15,
    python pca.py --participant t15 --session word-loudness --nbins_before_onset 75 --nbins_after_onset 75 --savepath_data ../plotting_data/t15/word-loudness/PCA/ --savepath_fig ../plotting_figures/t15/word-loudness/PCA/
For t16,
    python pca.py --participant t16 --session word-loudness --nbins_before_onset 75 --nbins_after_onset 75 --savepath_data ../plotting_data/t16/word-loudness/PCA/ --savepath_fig ../plotting_figures/t16/word-loudness/PCA/
'''

amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going']
amplitude_color = [
    (167/255, 185/255, 207/255),
    (114/255, 159/255, 207/255),
    (53/255, 126/255, 221/255),
    (0, 79/255, 158/255),
]
word_color = [
    (0.9254902, 0.12156863, 0.14117647),
    (0.98431373, 0.72941176, 0.07058824),
    (0.384, 0.682, 0.2),
    (0.43137255, 0.79607843, 0.85490196),
    [0.45568627, 0.31764706, 0.63529412],
    (0.84705882, 0.2627451, 0.59215686)
]

fontsize = 40
scatter_size = 3000
    

def plot_pca_projections(x_ax, y_ax, z_ax, expl_var, amp_label, word_label):

    # legend based on loudness
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_ax, y_ax, z_ax, color=np.array(amplitude_color)[amp_label], s=scatter_size, alpha = 0.8)
    ax.set_xlabel(f'PC 1 ({expl_var[0]*100:.1f}%)', fontsize = fontsize)
    ax.set_ylabel(f'PC 2 ({expl_var[1]*100:.1f}%)', fontsize = fontsize)
    ax.set_zlabel(f'PC 3 ({expl_var[2]*100:.1f}%)', fontsize = fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if args.participant == 't15':
        ax.view_init(elev = -140, azim = 120, roll = 0)
    elif args.participant == 't16':
        ax.view_init(elev = 70, azim = -120, roll = 8)
    ax.grid(False)

    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{formatted_datetime}_pca_amplitude_spb_{args.nbins_before_onset}_{args.nbins_after_onset}.png', format='png')

    # legend based on words
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_ax, y_ax, z_ax, color=np.array(word_color)[word_label], s=scatter_size, alpha = 0.8)
    ax.set_xlabel('PC 1', fontsize = fontsize)
    ax.set_ylabel('PC 2', fontsize = fontsize)
    ax.set_zlabel('PC 3', fontsize = fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if args.participant == 't15':
        ax.view_init(elev = -140, azim = 120, roll = 0)
    elif args.participant == 't16':
        ax.view_init(elev = 70, azim = -120, roll = 8)
    ax.grid(False)

    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{formatted_datetime}_pca_word_spb_{args.nbins_before_onset}_{args.nbins_after_onset}.png', format='png')

    return

if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--nbins_before_onset', type=int, default = None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default = None, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)

    # load pca projections
    print('Loading PCA projections...')
    pca_data_path = f'{args.savepath_data}{args.participant}_{args.nbins_before_onset}_{args.nbins_after_onset}_pca_projections.pkl'
    with open(pca_data_path, 'rb') as f:
        pca_data = pkl.load(f)
    
    x_ax = pca_data['x_ax']
    y_ax = pca_data['y_ax']
    z_ax = pca_data['z_ax']
    expl_var = pca_data['expl_var']
    amp_label = pca_data['amp_label']
    word_label = pca_data['word_label']


    print('Plotting PCA projections...')
    plot_pca_projections(x_ax, y_ax, z_ax, expl_var, amp_label, word_label)

    print('DONE!')
