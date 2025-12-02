# This script plots trial-averaged loudness and word decoding accuracy across time.

import argparse
import os
import numpy as np
import pickle as pkl
from datetime import datetime
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test

'''
Example cmd (when run from this directory; provide python script path appropriately if run from different directory):
For t15,
    python ol_striding_classification.py --participant t15 --session word-loudness --nbins_before_onset 150 --nbins_after_onset 150 --bins_before_trial_start 100 --bins_after_trial_start 100 --bins_before_trial_end 50 --bins_after_trial_end 200 --stream_window_len 40 --stream_window_stride 1 --savepath_data ../plotting_data/t15/word-loudness/ol_striding_classification/ --savepath_fig ../plotting_figures/t15/word-loudness/ol_striding_classification/
For t16,
    python ol_striding_classification.py --participant t16 --session word-loudness --nbins_before_onset 150 --nbins_after_onset 150 --bins_before_trial_start 100 --bins_after_trial_start 100 --bins_before_trial_end 50 --bins_after_trial_end 200 --stream_window_len 40 --stream_window_stride 1 --savepath_data ../plotting_data/t16/word-loudness/ol_striding_classification/ --savepath_fig ../plotting_figures/t16/word-loudness/ol_striding_classification/

Args:
nbins_before_onset = 150
nbins_after_onset = 150
bins_before_trial_start = 100
bins_after_trial_start = 100
bins_before_trial_end = 50
bins_after_trial_end = 200
stream_window_len = 40
stream_window_stride = 1

Data will be loaded from the specified savepath_data directory.
Figures will be saved in the specified savepath_fig directory.
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
bin_size_ms = 10
# plotting
fontsize = 16
scattersize = 70
linewidth = 4
my_color = 'green'
word_color = (150/255, 54/255, 34/255)
loudness_color = (34/255, 54/255, 150/255)
#--------------------------------------------
# functions
#--------------------------------------------
def plot_striding_performance(data):

    cue_onset_ind = (args.bins_before_trial_start - args.stream_window_len) / args.stream_window_stride 
    speech_onset_ind = (args.nbins_before_onset - args.stream_window_len) / args.stream_window_stride
    trial_end_onset_ind = (args.bins_before_trial_end - args.stream_window_len) / args.stream_window_stride
    print('Cue, speech and trial end onset in plot:', cue_onset_ind, speech_onset_ind, trial_end_onset_ind)
    
    fig, ax = plt.subplots(1,3,figsize=(15,4), gridspec_kw={'width_ratios': [len(data['mean_cue']), len(data['mean_speech']), len(data['mean_trial_end'])]})
    if args.participant == 't15':
        fontsize = 16
    elif args.participant == 't16':
        fontsize = 15

    # cue period
    ax[0].plot(data['mean_cue'], label = 'Loudness decoder', color = my_color, linewidth = linewidth)
    ax[0].fill_between(np.arange(len(data['mean_cue'])),
                                data['mean_cue'] - data['sem_cue'],
                                data['mean_cue'] + data['sem_cue'],
                                alpha = 0.5, label = '_hidden', color = my_color)

    ax[0].plot(data['mean_cue_chance'], label = 'Chance', color = 'black', linestyle = '--', linewidth = linewidth)
    ax[0].fill_between(np.arange(len(data['mean_cue_chance'])),
                                data['mean_cue_chance'] - data['sem_cue_chance'],
                                data['mean_cue_chance'] + data['sem_cue_chance'],
                                alpha = 0.5, label = 'hidden', color = 'black')

    ax[0].set_ylim([0,1])
    ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100], fontsize = fontsize)
    ax[0].set_ylabel('Accuracy (%)', fontsize = fontsize)
    ax[0].set_xticks([])

    for pos in ['right', 'top', 'bottom']: 
        ax[0].spines[pos].set_visible(False)
    if args.participant == 't16':
        ax[0].set_xticks(np.arange(0, len(data['mean_cue']), 50)+10, [k if k!=0 else '' for k in (np.arange(-(args.bins_before_trial_start-args.stream_window_len),args.bins_after_trial_start, 50)+10) * 10], fontsize = fontsize)
        ax[0].spines['bottom'].set_visible(True)

    ax[0].scatter(cue_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[0].text(cue_onset_ind, -0.08, "Cue", fontsize = fontsize, ha = "center")

    # time cluster permutation
    ax[0].scatter(data['cluster_permutation_bin'], 0.02, s = scattersize, color = my_color, marker = '*')

    # go period
    ax[1].plot(data['mean_speech'], label = 'Loudness decoder', color = my_color, linewidth = linewidth)
    ax[1].fill_between(np.arange(len(data['mean_speech'])),
                                data['mean_speech'] - data['sem_speech'],
                                data['mean_speech'] + data['sem_speech'],
                                alpha = 0.5, label = '_hidden', color = my_color)

    ax[1].plot(data['mean_speech_chance'], label = 'Chance', color = 'black', linestyle = '--', linewidth = linewidth)
    ax[1].fill_between(np.arange(len(data['mean_speech_chance'])),
                                data['mean_speech_chance'] - data['sem_speech_chance'],
                                data['mean_speech_chance'] + data['sem_speech_chance'],
                                alpha = 0.5, label = 'hidden', color = 'black')
    
    # add vertical line from where we can significantly decode loudness
    ax[1].vlines(data['peak_acc_ind'], 0, data['peak_acc'], color='black', linewidth=3, alpha = 0.4)
    ax[1].text(data['peak_acc_ind'], data['peak_acc'] + 0.05, f'{data["peak_acc"] * 100:.1f}%', fontsize = fontsize, ha = 'center')
    
    ax[1].vlines(data['prep_acc_ind'], 0, data['prep_acc'], color = 'black', linewidth = 3, alpha = 0.4)
    ax[1].text(data['prep_acc_ind'], data['prep_acc'] + 0.07, f'{data["prep_acc"] * 100:.1f}%', fontsize = fontsize, ha = 'center')
    
    
    ax[1].set_ylim([0,1])
    for pos in ['right', 'top', 'left', 'bottom']: 
        ax[1].spines[pos].set_visible(False) 
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    if args.participant == 't16':
        ax[1].set_xticks(np.arange(0, len(data['mean_speech']), 50) + 10, [k if k!= 0 else '' for k in (np.arange(-(args.nbins_before_onset-args.stream_window_len), args.nbins_after_onset, 50) + 10) * 10], fontsize = fontsize)
        ax[1].spines['bottom'].set_visible(True)
        ax[1].set_xlabel('Time (ms)', fontsize = fontsize)

    ax[1].scatter(speech_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[1].text(speech_onset_ind, -0.17, "Speech\nonset", fontsize = fontsize, ha = "center")
    
    
    # trial end period
    ax[2].plot(data['mean_trial_end'], label = 'Loudness decoder', color = my_color, linewidth = linewidth)
    ax[2].fill_between(np.arange(len(data['mean_trial_end'])),
                                data['mean_trial_end'] - data['sem_trial_end'],
                                data['mean_trial_end'] + data['sem_trial_end'],
                                alpha = 0.5, label = '_hidden', color = my_color)

    ax[2].plot(data['mean_trial_end_chance'], label = 'Chance', color = 'black', linestyle = '--', linewidth = linewidth)
    ax[2].fill_between(np.arange(len(data['mean_trial_end_chance'])),
                                data['mean_trial_end_chance'] - data['sem_trial_end_chance'],
                                data['mean_trial_end_chance'] + data['sem_trial_end_chance'],
                                alpha = 0.5, label = '_hidden', color = 'black')
    
    ax[2].set_ylim([0,1])
    for pos in ['right', 'top', 'left', 'bottom']: 
        ax[2].spines[pos].set_visible(False) 
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    if args.participant == 't16':
        ax[2].set_xticks(np.arange(0, len(data['mean_trial_end']), 50)[1:], np.arange(0, len(data['mean_trial_end']), 50)[1:] * 10, fontsize = fontsize)
        ax[2].spines['bottom'].set_visible(True)

    ax[2].scatter(trial_end_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[2].text(trial_end_onset_ind, -0.16, 'Speech\nOffset', fontsize = fontsize, ha = "center")
    if args.participant == 't15':
        ax[2].hlines(0.01, xmin = len(data['mean_trial_end']) - (50 / args.stream_window_stride), xmax = len(data['mean_trial_end']), linewidth = 3, color = 'black')
        ax[2].text(len(data['mean_trial_end']) - (25 / args.stream_window_stride), -0.08, "500 ms", ha = "center", fontsize = fontsize)
    
    plt.subplots_adjust(wspace=0.02) 
    fig.tight_layout()
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1), fontsize = fontsize)
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_loudness_striding_performance_{formatted_datetime}.png', format='png')

    return

def plot_striding_performance_word_loudness(data_loudness, data_word):

    cue_onset_ind = (args.bins_before_trial_start - args.stream_window_len) / args.stream_window_stride 
    speech_onset_ind = (args.nbins_before_onset - args.stream_window_len) / args.stream_window_stride
    trial_end_onset_ind = (args.bins_before_trial_end - args.stream_window_len) / args.stream_window_stride
    print('Cue, speech and trial end onset in plot:', cue_onset_ind, speech_onset_ind, trial_end_onset_ind)

    assert len(data_loudness['mean_cue']) == len(data_word['mean_cue'])
    assert len(data_loudness['mean_speech']) == len(data_word['mean_speech'])
    assert len(data_loudness['mean_trial_end']) == len(data_word['mean_trial_end'])

    fig, ax = plt.subplots(1,3,figsize=(15,4), gridspec_kw={'width_ratios': [len(data_loudness['mean_cue']), len(data_loudness['mean_speech']), len(data_loudness['mean_trial_end'])]})
    fontsize = 16

    # cue period
    ax[0].plot(data_loudness['mean_cue'], label = 'Loudness decoder', color = loudness_color, linewidth = linewidth)
    ax[0].fill_between(np.arange(len(data_loudness['mean_cue'])),
                                data_loudness['mean_cue'] - data_loudness['sem_cue'],
                                data_loudness['mean_cue'] + data_loudness['sem_cue'],
                                alpha = 0.5, label = '_hidden', color = loudness_color)

    ax[0].plot(data_loudness['mean_cue_chance'], label = 'Chance (loudness)', color = 'black', linestyle = '--', linewidth = linewidth, alpha = 1)
    ax[0].fill_between(np.arange(len(data_loudness['mean_cue_chance'])),
                                data_loudness['mean_cue_chance'] - data_loudness['sem_cue_chance'],
                                data_loudness['mean_cue_chance'] + data_loudness['sem_cue_chance'],
                                alpha = 0.8, label = 'hidden', color = 'black')
    
    ax[0].plot(data_word['mean_cue'], label = 'Word decoder', color = word_color, linewidth = linewidth)
    ax[0].fill_between(np.arange(len(data_word['mean_cue'])),
                                data_word['mean_cue'] - data_word['sem_cue'],
                                data_word['mean_cue'] + data_word['sem_cue'],
                                alpha = 0.5, label = '_hidden', color = word_color)

    ax[0].plot(data_word['mean_cue_chance'], label = 'Chance (word)', color = 'black', linestyle = '--', linewidth = linewidth, alpha = 0.6)
    ax[0].fill_between(np.arange(len(data_word['mean_cue_chance'])),
                                data_word['mean_cue_chance'] - data_word['sem_cue_chance'],
                                data_word['mean_cue_chance'] + data_word['sem_cue_chance'],
                                alpha = 0.4, label = 'hidden', color = 'black')

    ax[0].set_ylim([0,1])
    ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100], fontsize = fontsize)
    ax[0].set_ylabel('Accuracy (%)', fontsize = fontsize)
    ax[0].set_xticks([])
    for pos in ['right', 'top', 'bottom']: 
        ax[0].spines[pos].set_visible(False)
    ax[0].set_xticks(np.arange(0, len(data_word['mean_cue']), 50)+10, [k if k!=0 else '' for k in (np.arange(-(args.bins_before_trial_start-args.stream_window_len),args.bins_after_trial_start, 50)+10) * 10], fontsize = fontsize)
    ax[0].spines['bottom'].set_visible(True)

     
    ax[0].scatter(cue_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[0].text(cue_onset_ind, -0.08, "Cue", fontsize = fontsize, ha = "center")
    ax[0].scatter(data_loudness['cluster_permutation_bin'], 0.02, s = scattersize, color = loudness_color, marker = '*')
    ax[0].scatter(data_word['cluster_permutation_bin'], 0.02, s = scattersize, color = word_color, marker = '*')

    # go period
    ax[1].plot(data_loudness['mean_speech'], label = 'Loudness decoder', color = loudness_color, linewidth = linewidth)
    ax[1].fill_between(np.arange(len(data_loudness['mean_speech'])),
                                data_loudness['mean_speech'] - data_loudness['sem_speech'],
                                data_loudness['mean_speech'] + data_loudness['sem_speech'],
                                alpha = 0.5, label = '_hidden', color = loudness_color)

    ax[1].plot(data_loudness['mean_speech_chance'], label = 'Chance (loudness)', color = 'black', linestyle = '--', linewidth = linewidth, alpha = 1)
    ax[1].fill_between(np.arange(len(data_loudness['mean_speech_chance'])),
                                data_loudness['mean_speech_chance'] - data_loudness['sem_speech_chance'],
                                data_loudness['mean_speech_chance'] + data_loudness['sem_speech_chance'],
                                alpha = 0.8, label = 'hidden', color = 'black')
    
    ax[1].plot(data_word['mean_speech'], label = 'Word decoder', color = word_color, linewidth = linewidth)
    ax[1].fill_between(np.arange(len(data_word['mean_speech'])),
                                data_word['mean_speech'] - data_word['sem_speech'],
                                data_word['mean_speech'] + data_word['sem_speech'],
                                alpha = 0.5, label = '_hidden', color = word_color)

    ax[1].plot(data_word['mean_speech_chance'], label = 'Chance (word)', color = 'black', linestyle = '--', linewidth = linewidth, alpha = 0.6)
    ax[1].fill_between(np.arange(len(data_word['mean_speech_chance'])),
                                data_word['mean_speech_chance'] - data_word['sem_speech_chance'],
                                data_word['mean_speech_chance'] + data_word['sem_speech_chance'],
                                alpha = 0.4, label = 'hidden', color = 'black')
    
    # add vertical line from where we can decode loudness with max accuracy
    ax[1].vlines(data_loudness['peak_acc_ind'], 0, data_loudness['peak_acc'], linewidth=3, alpha = 0.4, color = loudness_color)
    ax[1].text(data_loudness['peak_acc_ind'], data_loudness['peak_acc'] + 0.05, f'{data_loudness["peak_acc"] * 100:.1f}%', fontsize = fontsize, ha = 'center', color = loudness_color)
    ax[1].vlines(data_word['peak_acc_ind'], 0, data_word['peak_acc'], linewidth=3, alpha = 0.4, color = word_color)
    ax[1].text(data_word['peak_acc_ind'], data_word['peak_acc'] + 0.05, f'{data_word["peak_acc"] * 100:.1f}%', fontsize = fontsize, ha = 'center', color = word_color)
    
    ax[1].set_ylim([0,1])
    for pos in ['right', 'top', 'left', 'bottom']: 
        ax[1].spines[pos].set_visible(False) 
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_xticks(np.arange(0, len(data_word['mean_speech']), 50) + 10, [k if k!=0 else '' for k in (np.arange(-(args.nbins_before_onset-args.stream_window_len), args.nbins_after_onset, 50) + 10) * 10], fontsize = fontsize)
    ax[1].spines['bottom'].set_visible(True)
    ax[1].set_xlabel('Time (ms)', fontsize = fontsize)

    ax[1].scatter(speech_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[1].text(speech_onset_ind, -0.17, "Speech\nonset", fontsize = fontsize, ha = "center")
    
    
    # trial end period
    ax[2].plot(data_loudness['mean_trial_end'], label = 'Loudness decoder', color = loudness_color, linewidth = linewidth)
    ax[2].fill_between(np.arange(len(data_loudness['mean_trial_end'])),
                                data_loudness['mean_trial_end'] - data_loudness['sem_trial_end'],
                                data_loudness['mean_trial_end'] + data_loudness['sem_trial_end'],
                                alpha = 0.5, label = '_hidden', color = loudness_color)

    ax[2].plot(data_loudness['mean_trial_end_chance'], label = 'Chance (loudness)', color = 'black', linestyle = '--', linewidth = linewidth, alpha = 1)
    ax[2].fill_between(np.arange(len(data_loudness['mean_trial_end_chance'])),
                                data_loudness['mean_trial_end_chance'] - data_loudness['sem_trial_end_chance'],
                                data_loudness['mean_trial_end_chance'] + data_loudness['sem_trial_end_chance'],
                                alpha = 0.8, label = '_hidden', color = 'black')
    
    ax[2].plot(data_word['mean_trial_end'], label = 'Word decoder', color = word_color, linewidth = linewidth)
    ax[2].fill_between(np.arange(len(data_word['mean_trial_end'])),
                                data_word['mean_trial_end'] - data_word['sem_trial_end'],
                                data_word['mean_trial_end'] + data_word['sem_trial_end'],
                                alpha = 0.5, label = '_hidden', color = word_color)

    ax[2].plot(data_word['mean_trial_end_chance'], label = 'Chance (word)', color = 'black', linestyle = '--', linewidth = linewidth, alpha = 0.6)
    ax[2].fill_between(np.arange(len(data_word['mean_trial_end_chance'])),
                                data_word['mean_trial_end_chance'] - data_word['sem_trial_end_chance'],
                                data_word['mean_trial_end_chance'] + data_word['sem_trial_end_chance'],
                                alpha = 0.4, label = '_hidden', color = 'black')
    
    ax[2].set_ylim([0,1])
    for pos in ['right', 'top', 'left', 'bottom']: 
        ax[2].spines[pos].set_visible(False) 
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_xticks(np.arange(0, len(data_word['mean_trial_end']), 50)[1:], np.arange(0, len(data_word['mean_trial_end']), 50)[1:] * 10, fontsize = fontsize)
    ax[2].spines['bottom'].set_visible(True)


    ax[2].scatter(trial_end_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[2].text(trial_end_onset_ind, -0.17, 'Speech\noffset', fontsize = fontsize, ha = "center")
    
    plt.subplots_adjust(wspace=0.02) 
    fig.tight_layout()
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1), fontsize = fontsize)
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_word_and_loudness_striding_performance_{formatted_datetime}.png', format='png')

    return


if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--stream_window_len', type=int, default=None, help = 'stream window length')
    parser.add_argument('--stream_window_stride', type=int, default=None, help = 'stream window stride')
    parser.add_argument('--bins_before_trial_start', type=int, default=100, help = 'number of bins to consider before start of the trial')
    parser.add_argument('--bins_after_trial_start', type=int, default = 100, help = 'number of bins to consider after the start of the trial')
    parser.add_argument('--bins_before_trial_end', type=int, default=100, help = 'number of bins to consider before end of trial')
    parser.add_argument('--bins_after_trial_end', type=int, default = 100, help = 'number of bins to consider after end of trial')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()
    
    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)
    
    print('Running ol_striding_classification.py')
    print(args)

    # load loudness decoding results
    print('Loading loudness decoding across time...')
    with open(f'{args.savepath_data}{args.participant}_ol_striding_loudness_results_for_github.pkl', 'rb') as f: # 40 win 1 stride
        data_loudness = pkl.load(f)

    # plot loudness results
    print('Plot loudness decoding across time...')
    plot_striding_performance(data_loudness)

    # plot word decoding results
    print('Loading word decoding across time...')
    with open(f'{args.savepath_data}{args.participant}_ol_striding_word_results_for_github.pkl', 'rb') as f: # 40 win 1 stride
        data_word = pkl.load(f)

    # plot loudness and word results together
    print('Plotting word and loudness decodign together...')
    plot_striding_performance_word_loudness(data_loudness, data_word)

    print('DONE!')
