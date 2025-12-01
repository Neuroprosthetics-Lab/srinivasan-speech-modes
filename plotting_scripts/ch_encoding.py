# This script plots the channels that are significantly modulated by loudness levels and words.

import argparse
import numpy as np
from datetime import datetime
import os
import math
import matplotlib.pyplot as plt
import pickle as pkl

'''
Example cmd:
For t15,
    python ch_encoding.py --participant t15 --session word-loudness --savepath_data ../plotting_data/t15/word-loudness/ch_encoding/ --savepath_fig ../plotting_figs/t15/word-loudness/ch_encoding/
For t16,
    python ch_encoding.py --participant t16 --session word-loudness --savepath_data ../plotting_data/t16/word-loudness/ch_encoding/ --savepath_fig ../plotting_figs/t16/word-loudness/ch_encoding/

Data will be loaded from the specified savepath_data directory.
Figures will be saved in the specified savepath_fig directory.
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going']

plotting_order_1 = [
    63, 55, 47, 39, 31, 23, 15, 7,
    62, 54, 46, 38, 30, 22, 14, 6,
    61, 53, 45, 37, 29, 21, 13, 5,
    60, 52, 44, 36, 28, 20, 12, 4,
    59, 51, 43, 35, 27, 19, 11, 3,
    58, 50, 42, 34, 26, 18, 10, 2,
    57, 49, 41, 33, 25, 17, 9,  1,
    56, 48, 40, 32, 24, 16, 8,  0
]
plotting_order_2 = [
    0, 8,  16, 24, 32, 40, 48, 56,
    1, 9,  17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58,
    3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60,
    5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62,
    7, 15, 23, 31, 39, 47, 55, 63,
]

# in the order you want the arrays to appear in the fig
ch_sets = {
    't15': [list(range(192, 256)), list(range(128, 192)), list(range(0,64)), list(range(64, 128))],
    't16': [list(range(0,64)), list(range(64, 128))]#, list(range(128, 192)), list(range(192, 256))]
}
ch_set_names = {
    't15': ['55b', 'd6v', 'M1', 'v6v'], # using_correct_electrode_mapping = 0
    't16': ['55b/PEF', '6v']#,'HK1','HK2'] # only speech arrays needed
}
plotting_orders = {
    't15': [plotting_order_2, plotting_order_2, plotting_order_1, plotting_order_1],
    't16': [plotting_order_2, plotting_order_2]#, plotting_order_1, plotting_order_1]
}

fig_fontsize = 20
scatter_size_scales = {
    'loudness':30,
    'word': 20,
}

colors = {
    'loudness':(34/255, 54/255, 150/255), # loudness
    'word': (150/255, 54/255, 34/255), # word
}
#--------------------------------------------
# functions
#--------------------------------------------
def plot_significant_channels(ch_modulation_level, mark_channels, color, scatter_size_scale, legend_key):
    
    if args.participant == 't15':
        fig, ax = plt.subplots(len(ch_set_names[args.participant]), 1, figsize = (8, 11))
    elif args.participant == 't16':
        fig, ax = plt.subplots(len(ch_set_names[args.participant]), 1, figsize = (3, 6))

    # plot significant channels in an array 
    for n, (ch_set, current_plotting_order) in enumerate(zip(ch_sets[args.participant], plotting_orders[args.participant])):
        ax[n].set_xlim(-1, 8.5)
        ax[n].set_ylim(-1, 8.5)
        ax[n].set_aspect('equal')
        ax[n].axis('off')  # Turn off the axis

        for i, ch in enumerate(ch_set):
        # calculate row and col position
            row = current_plotting_order[i] // 8
            col = current_plotting_order[i] % 8
            if ch_modulation_level[ch][0] == 0:
                ax[n].scatter(col, 7 - row, s = 4 * scatter_size_scale, facecolors='none', edgecolors=color)
            else:
                ax[n].scatter(col, 7 - row, s = ch_modulation_level[ch][0] * scatter_size_scale, color = color)
            
            if ch in mark_channels:
                ax[n].scatter(col, 7 - row, s = 12 * scatter_size_scale, facecolors='none', edgecolors='darkorange', linewidth = 2)
        
        ax[n].text(3.5,8, ch_set_names[args.participant][n], fontsize = fig_fontsize, ha = 'center')   

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_ch_encoding_{legend_key}_{formatted_datetime}.png', format='png')

    return

def plot_legend(color, scatter_size_scale, legend_key):
    
    # legend
    if legend_key == 'loudness':
        n_pairs = math.comb(len(amplitudes), 2)
    elif legend_key == 'word':
        n_pairs = math.comb(len(words), 2)
    x = np.arange(0, n_pairs + 1, 1)
    y = 0
    fig, ax = plt.subplots(figsize = (7,2))
    for i in range(n_pairs + 1): # 1 added to manage no tuning (0, empty circle)
        if i == 0:
            ax.scatter(x[i], y, s = 4 * scatter_size_scale, facecolors='none', edgecolors = color)
        else:
            ax.scatter(x[i], y, s = i * scatter_size_scale, color = color)

    txt = f'Number of {legend_key} pairs with significantly different firing rates\n0 {n_pairs}'
    ax.annotate(txt, (x[0], y + 0.03), fontsize = 13)
    plt.axis('off')
    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_ch_encoding_{legend_key}_legend_{formatted_datetime}.png', format='png')

    return


if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_data):
        raise FileNotFoundError(f'Specified data path does not exist: {args.savepath_data}')
    
    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)

    print('Running ch_encoding.py')
    print(args)

    # same order as arrays above
    mark_channels = { # channels shown in fig 1 (supp fig 1)
        't15': [197, 158, 38, 120], # channel (0-indexed), ordered according to implanted arrays
        't16': [57, 89], # channel (0-indexed), ordered according to implanted arrays (only speech arrays considered)
    }

    # load ch encoding for loudness
    with open(f'{args.savepath_data}ch_encoding_loudness.pkl', 'rb') as f:
        n_significant_amp_encoding = pkl.load(f)

    # plot significant channels
    print('Plotting channels with number of loudness level tuning ...')
    plot_significant_channels(n_significant_amp_encoding, mark_channels[args.participant], colors['loudness'], scatter_size_scales['loudness'], legend_key = 'loudness')
    plot_legend(colors['loudness'], scatter_size_scales['loudness'], legend_key = 'loudness')

    # load ch encoding for word
    print('Plotting channels with number of word tuning ...')
    with open(f'{args.savepath_data}ch_encoding_word.pkl', 'rb') as f:
        n_significant_word_encoding = pkl.load(f)   

    plot_significant_channels(n_significant_word_encoding, mark_channels[args.participant], colors['word'], scatter_size_scales['word'], legend_key = 'word')
    plot_legend(colors['word'], scatter_size_scales['word'], legend_key = 'word')

    print('DONE!')
