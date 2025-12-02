# This script plots the decoding performance when randomly dropping X channels.

import argparse
import os
import numpy as np
import pickle as pkl
from datetime import datetime
import matplotlib.pyplot as plt

'''
Example cmd (when run from this directory; provide python script path appropriately if run from different directory):
For t15,
    python ch_dropping.py --participant t15 --session word-loudness --nbins_before_onset 60 --nbins_after_onset 60 --savepath_data ../plotting_data/t15/word-loudness/ch_dropping/ --savepath_fig ../plotting_figures/t15/word-loudness/ch_dropping/
For t16,
    python ch_dropping.py --participant t16 --session word-loudness --nbins_before_onset 60 --nbins_after_onset 60 --savepath_data ../plotting_data/t16/word-loudness/ch_dropping/ --savepath_fig ../plotting_figures/t16/word-loudness/ch_dropping/

Data will be loaded from the specified savepath_data directory.
Figures will be saved in the specified savepath_fig directory.
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
# plotting
fontsize = 17
my_color = 'navy'

#--------------------------------------------
# functions
#--------------------------------------------
def plot_channel_dropping_curve(accuracies):

    n_ch_kept = list(accuracies['all_mean'].keys())
    acc = [ch_acc for ch_acc in accuracies['all_mean'].values()]
    std = [ch_acc for ch_acc in accuracies['all_std'].values()]
    
    fig = plt.figure(figsize = (5,5))
    plt.plot(n_ch_kept, acc, 'o', markersize=6, markerfacecolor = my_color, markeredgecolor = my_color, markeredgewidth=2)
    plt.errorbar(n_ch_kept, acc, yerr = std, fmt = 'none', ecolor = my_color, elinewidth = 1, capsize = 2)

    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.2), np.arange(0, 101, 20), fontsize = fontsize)
    plt.ylabel('Accuracy (%)', fontsize = fontsize)
    plt.xticks(n_ch_kept[0::4], n_ch_kept[0::4], fontsize = fontsize, rotation = 90)

    plt.xlabel('Number of electrodes', fontsize = fontsize)
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 
    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_channel_dropping_acc_{formatted_datetime}.png', format='png')

    return



if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)
    
    print('Running channel_dropping_performance.py')
    print(args)

    print('Loading channel dropping results...')
    with open(f'{args.savepath_data}{args.participant}_ch_dropping_results.pkl', 'rb') as f:
        channel_dropped_accuracies = pkl.load(f)

    # plot results (accuracies)
    print('Plotting channel dropping results...')
    plot_channel_dropping_curve(channel_dropped_accuracies)

    print('DONE!')