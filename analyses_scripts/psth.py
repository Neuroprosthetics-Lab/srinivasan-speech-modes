# This script computes the psth given sample neural data.

import argparse
import os
import scipy
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from functions import get_audio_onset_offset
import pickle as pkl
from datetime import datetime
import sys
import matplotlib.transforms as transforms

'''
Example cmd:
For t15,
    python psth.py --participant t15 --session word-loudness --nbins_before_onset 100 --nbins_after_onset 50 --nbins_before_offset 25 --nbins_after_offset 25 --plot_speech_offset --savepath_data ../analyses_figures_data/t15/word-loudness/psth/ --savepath_fig ../analyses_figures/t15/word-loudness/psth/
For t16,
    python psth.py --participant t16 --session word-loudness --nbins_before_onset 100 --nbins_after_onset 50 --nbins_before_offset 25 --nbins_after_offset 25 --plot_speech_offset --savepath_data ../analyses_figures_data/t16/word-loudness/psth/ --savepath_fig ../analyses_figures/t16/word-loudness/psth/

Sample neural data will be loaded from ../sample_neural_data/{participant}/{session}/.
Results obtained from this script will be saved in savepath_data.
Figures generated using those results will be saved in savepath_fig.
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------

bin_size_ms = 10
fs = 30000
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
n_channels = 256
bins_before_trial_end = 10
delay_onset_raw_threshcross_from_pre_cue = 100 # rdbmat has 1s of raw thx before trial start included "raw_threshcross_from_pre_cue"
required_binned_delay_duration = 100 # required delay duration bins for plotting

# gassian smoothing params
sigma = {
    't15': 4,
    't16': 4,
}
order = 0

# plotting
my_color = [
    (167/255, 185/255, 207/255),
    (114/255, 159/255, 207/255),
    (53/255, 126/255, 221/255),
    (0, 79/255, 158/255),
]
fontsize = 12 # general for all psth plots; not for select channels
scatter_size = 30
delay_duration = 100
linewidth = 2

arrays = {
    't15': ['M1', 'v6v','d6v','55b'], # correct_electrode_mapping = 0
    't16': ['55b/PEF', '6v', 'HK1', 'HK2'],
}

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

firingrate_scatter_size = 100
firingrate_fontsize = 14

#--------------------------------------------
# functions
#--------------------------------------------
def load_rdbmat(participant, session, required_keys):
    # load data, remove incorrect trials
    data_path = f'../sample_neural_data/{participant}/{session}/' # t15.2023.11.04 has using_correct_electrode_mapping = 0
    files = os.listdir(data_path)

    data = {}

    for file in files:
        name, extension = os.path.splitext(file)
        if extension == '.mat':
            fullPath = str(Path(data_path, file).resolve())
            print(f'Loading {fullPath} ...')
            data_temp_required = scipy.io.loadmat(fullPath)

            # append data to master dict
            if data == {}:
                for key in required_keys:
                    data[key] = data_temp_required[key]
            else:
                for key in required_keys:
                    data[key] = np.append(data[key], data_temp_required[key], axis = -1)

    print('Data loaded ...')
    for key in data:
        print(key, data[key].shape)

    return data


def compute_psth(data):

    avg_go_thx = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels)) # n_amp x time_bins x n_channels
    sem_go_thx = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels)) # n_amp x time_bins x n_channels

    avg_delay_thx = np.empty((0, bins_before_trial_end + required_binned_delay_duration, n_channels)) # n_amp x time_bins x n_channels
    sem_delay_thx = np.empty((0, bins_before_trial_end + required_binned_delay_duration, n_channels)) # n_amp x time_bins x n_channels

    avg_end_thx = np.empty((0, args.nbins_before_offset + args.nbins_after_offset, n_channels)) # n_amp x time_bins x n_channels
    sem_end_thx = np.empty((0, args.nbins_before_offset + args.nbins_after_offset, n_channels)) # n_amp x time_bins x n_channels

    skipped_trial_ids = []

    for amp in amplitudes:

        go_thx = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels)) # speech onset
        delay_thx = np.empty((0, bins_before_trial_end + required_binned_delay_duration, n_channels))
        end_thx = np.empty((0, args.nbins_before_offset + args.nbins_after_offset, n_channels)) # speech offset


        if args.participant == 't15':
            inds = [i for i in range(len(data['cue'])) if amp in data['cue'][i] and 
                    max(np.squeeze(data['predaudio16k'])[i]) != 0]
        elif args.participant in ['t16', 't19']:
            inds = [i for i in range(len(data['cue'])) if amp in data['cue'][i]]
        
        for ind in inds:
            # get current data
            prev_threshcross = np.squeeze(data['raw_threshcross_from_pre_cue'])[ind]
            threshcross = np.squeeze(data['raw_threshcross'])[ind]
            # next_threshcross = np.squeeze(data['raw_threshcross_with_post_end'])[ind]
            delay_duration_ms = np.squeeze(data['delay_duration_ms'])[ind]
            binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)
            
            # speech onset
            if args.participant == 't15':
                predaudio16k = np.squeeze(data['predaudio16k'])[ind]
                start_ind, end_ind = get_audio_onset_offset(predaudio16k, display_audio = False, 
                                                                    mic_audio = None, cue = None, 
                                                                    intersegment_duration = 3000, 
                                                                    amplitude_percentage = 0.1) # returns ind at 30k
            elif args.participant == 't16':
                start_ind = np.squeeze(data['speech_onsets'])[ind]
                end_ind = np.squeeze(data['speech_offsets'])[ind]

            
            # binned start and end ind
            start_ind = math.floor((start_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index
            end_ind = math.ceil((end_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index

            # gaussian smoothing
            prev_threshcross_smth = gaussian_filter1d(prev_threshcross.astype(float), sigma = sigma[args.participant], order = order, axis = 0) # sigma controls smoothing, higher is more smoothed
            threshcross_smth = gaussian_filter1d(threshcross.astype(float), sigma = sigma[args.participant], order = order, axis = 0) # sigma controls smoothing, higher is more smoothed
            # next_threshcross_smth = gaussian_filter1d(next_threshcross.astype(float), sigma = sigma[args.participant], order = order, axis = 0)

            # speech onset and offset threshcross
            speech_onset_thx = np.expand_dims(threshcross[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :], 0)
            speech_offset_thx = np.expand_dims(threshcross[binned_delay_duration + (end_ind - args.nbins_before_offset): binned_delay_duration + (end_ind + args.nbins_after_offset), :], 0)
            if speech_onset_thx.shape[1] == args.nbins_before_onset + args.nbins_after_onset and speech_offset_thx.shape[1] == args.nbins_before_offset + args.nbins_after_offset:

                # print(binned_delay_duration + start_ind, binned_delay_duration + end_ind, threshcross_smth.shape,)

                # go period threshcross
                temp_thx = threshcross_smth[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :] * 100 # multiply by 100 for firing rate, shape (time_bins x 256)
                go_thx = np.append(go_thx, np.expand_dims(temp_thx, 0), axis = 0)
            
                # delay period threshcross, with some threshold crossings before cue onset
                temp_thx = prev_threshcross_smth[(delay_onset_raw_threshcross_from_pre_cue - bins_before_trial_end): (delay_onset_raw_threshcross_from_pre_cue + required_binned_delay_duration), :] * 100 # multiply by 100 for firing rate, shape (time_bins x 256); first 200 bins after trial start
                delay_thx = np.append(delay_thx, np.expand_dims(temp_thx, axis = 0), axis = 0)

                # speech offset threshcross
                temp_thx = threshcross_smth[binned_delay_duration + (end_ind - args.nbins_before_offset): binned_delay_duration + (end_ind + args.nbins_after_offset), :] * 100
                end_thx = np.append(end_thx, np.expand_dims(temp_thx, 0), axis = 0)

            else:
                skipped_trial_ids.append(ind)
                # print(f'Skipping because {args.nbins_after_onset} bins after speech onset is after the end of the trial',binned_delay_duration + start_ind, binned_delay_duration + end_ind, threshcross_smth.shape,)
                
        avg_go_thx = np.append(avg_go_thx, np.expand_dims(np.mean(go_thx, axis = 0), axis = 0), axis = 0)
        sem_go_thx = np.append(sem_go_thx, np.expand_dims(np.std(go_thx, axis = 0) / np.sqrt(go_thx.shape[0]), axis = 0), 
                                        axis = 0)
        avg_delay_thx = np.append(avg_delay_thx, np.expand_dims(np.mean(delay_thx, axis = 0), axis = 0), axis = 0)
        sem_delay_thx = np.append(sem_delay_thx, np.expand_dims(np.std(delay_thx, axis = 0) / np.sqrt(delay_thx.shape[0]), axis = 0),
                                    axis = 0)
        
        avg_end_thx = np.append(avg_end_thx, np.expand_dims(np.mean(end_thx, axis = 0), axis = 0), axis = 0)
        sem_end_thx = np.append(sem_end_thx, np.expand_dims(np.std(end_thx, axis = 0) / np.sqrt(end_thx.shape[0]), axis = 0), 
                                        axis = 0)
        
    print('Skipped trials: ', skipped_trial_ids)
    print('Number of skipped trials:', len(skipped_trial_ids))
        
    print('AVG and SEM thx shapes (go, delay, end)', avg_go_thx.shape, sem_go_thx.shape, avg_delay_thx.shape, sem_delay_thx.shape, avg_end_thx.shape, sem_end_thx.shape)
    return avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, avg_end_thx, sem_end_thx



def plot_psth_per_channel(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx):

    for i in range(n_channels):
        fig, ax = plt.subplots(1, 2, figsize = (11, 6), 
                               gridspec_kw={'width_ratios': [bins_before_trial_end + required_binned_delay_duration, args.nbins_before_onset + args.nbins_after_onset]}) # 'width_ratios': [n_time_bins_plotted in left subplot, n_time_bins plotted in right subplot]
        fig.tight_layout()

        # delay period
        for a in range(len(amplitudes)):
            ax[0].plot(avg_delay_thx[a, :, i].T, color = my_color[a])
            ax[0].fill_between(np.arange(bins_before_trial_end + required_binned_delay_duration),
                                avg_delay_thx[a, :, i].T - sem_delay_thx[a, :, i].T,
                                avg_delay_thx[a, :, i].T + sem_delay_thx[a, :, i].T,
                                alpha = 0.5, label='_hidden', color = my_color[a])
        ax[0].set_ylim([0, 90])
        ax[0].set_yticks([0, 90], [0, 90], fontsize = fontsize)
        ax[0].set_ylabel('Firing rate (Hz)', fontsize = fontsize)
        ax[0].set_xticks([])

        for pos in ['right', 'top', 'bottom']: 
            ax[0].spines[pos].set_visible(False) 

        ax[0].scatter(bins_before_trial_end, 2, color='black', s=scatter_size)
        ax[0].text(bins_before_trial_end, -4, 'Cue', color='black', ha = 'center', fontsize = fontsize)

        # go period
        for a in range(len(amplitudes)):
            ax[1].plot(avg_go_thx[a, :, i].T, label = amplitudes[a], color = my_color[a])
            ax[1].fill_between(np.arange(args.nbins_before_onset + args.nbins_after_onset),
                                avg_go_thx[a, :, i].T - sem_go_thx[a, :, i].T,
                                avg_go_thx[a, :, i].T + sem_go_thx[a, :, i].T,
                                alpha = 0.5, label = '_hidden', color = my_color[a])

        ax[1].set_ylim([0, 90])
        for pos in ['right', 'top', 'left', 'bottom']: 
            ax[1].spines[pos].set_visible(False) 
        ax[1].set_yticks([])
        ax[1].set_xticks([])

        ax[1].scatter(args.nbins_before_onset, 2, color='black', s=scatter_size)
        ax[1].text(args.nbins_before_onset, -6, 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
        ax[1].hlines(0, xmin = len(avg_go_thx[a, :, i]) - 50, xmax = len(avg_go_thx[a, :, i]), linewidth = 5, color = 'black')
        ax[1].text(len(avg_go_thx[a, :, i]) - 25, -4, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)

        # legend
        for a in range(4):
            plt.text(220, 80 - a * 5, amplitudes[a], ha = "center", fontsize = fontsize, color = my_color[a])

        plt.text(0, 60, f'Channel {i}\nElectrode {(i+1)%64} in {arrays[args.participant][math.floor(i/64)]}', color = 'red', fontsize = fontsize)
        
        # save figure
        # plt.show()
        plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_channel_{i}_psth.png')
        
    return



def plot_psth_given_channel(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, avg_end_thx, sem_end_thx, plt_channel):

    num_subplots = 2
    width_ratios = [bins_before_trial_end + required_binned_delay_duration, args.nbins_before_onset + args.nbins_after_onset]
    if args.plot_speech_offset:
        num_subplots = 3
        width_ratios.extend([args.nbins_before_offset + args.nbins_after_offset])

    # plot one example electrode per array
    if args.participant == 't15':
        fontsize = 14
        if args.plot_speech_offset:
            fontsize = 13
        fig, ax = plt.subplots(len(plt_channel), num_subplots, figsize = (6, 8), 
                            gridspec_kw={'width_ratios': width_ratios}) # 'width_ratios': [n_time_bins_plotted in left subplot, n_time_bins plotted in right subplot]
    
    elif args.participant == 't16':
        fontsize = 14
        if args.plot_speech_offset:
            fontsize = 13
        fig, ax = plt.subplots(len(plt_channel), num_subplots, figsize = (8, 4), # (6, 4) originally
                            gridspec_kw={'width_ratios': width_ratios}) # 'width_ratios': [n_time_bins_plotted in left subplot, n_time_bins plotted in right subplot]
    
    for subplot in range(len(plt_channel)):
        # delay period
        for a in range(len(amplitudes)):
            ax[subplot, 0].plot(avg_delay_thx[a, :, plt_channel[subplot]].T, color = my_color[a], linewidth = linewidth)
            ax[subplot, 0].fill_between(np.arange(bins_before_trial_end + required_binned_delay_duration),
                                avg_delay_thx[a, :, plt_channel[subplot]].T - sem_delay_thx[a, :, plt_channel[subplot]].T,
                                avg_delay_thx[a, :, plt_channel[subplot]].T + sem_delay_thx[a, :, plt_channel[subplot]].T,
                                alpha = 0.5, label='_hidden', color = my_color[a], linewidth = linewidth)
        
        ax[subplot, 0].set_ylim([0, 110])
        if subplot == 0:
            ax[subplot, 0].set_yticks([0, 110], [0, 110], fontsize = fontsize)
            ax[subplot, 0].set_ylabel('Firing rate (Hz)', fontsize = fontsize)
        else:
            ax[subplot, 0].set_yticks([])
        if args.participant != 't16':
            ax[subplot, 0].set_xticks([])
        elif args.participant == 't16':
            ax[subplot, 0].set_xticks(np.arange(10, required_binned_delay_duration + 11, 50), [k*10 if k != 0 else '' for k in (np.arange(10, required_binned_delay_duration + 11, 50) - 10)], fontsize = fontsize, rotation = 45)
            if subplot != len(plt_channel) - 1:
                ax[subplot, 0].tick_params(labelbottom = False)
        if args.participant != 't16':
            for pos in ['right', 'top', 'bottom']: 
                ax[subplot, 0].spines[pos].set_visible(False) 
        elif args.participant == 't16':
            for pos in ['right', 'top']: 
                ax[subplot, 0].spines[pos].set_visible(False) 

        ax[subplot, 0].scatter(bins_before_trial_end, 3.5, color='black', s=scatter_size)
        if subplot == len(plt_channel) - 1:
            if args.participant == 't15':
                ax[subplot, 0].text(bins_before_trial_end, - 12, 'Cue', color='black', ha = 'center', fontsize = fontsize)
            elif args.participant == 't16':
                ax[subplot, 0].text(bins_before_trial_end, - 16, 'Cue', color='black', ha = 'center', fontsize = fontsize)

        # go period
        for a in range(len(amplitudes)):
            ax[subplot, 1].plot(avg_go_thx[a, :, plt_channel[subplot]].T, label = amplitudes[a], color = my_color[a], linewidth = linewidth)
            ax[subplot, 1].fill_between(np.arange(args.nbins_before_onset + args.nbins_after_onset),
                                avg_go_thx[a, :, plt_channel[subplot]].T - sem_go_thx[a, :, plt_channel[subplot]].T,
                                avg_go_thx[a, :, plt_channel[subplot]].T + sem_go_thx[a, :, plt_channel[subplot]].T,
                                alpha = 0.5, label = '_hidden', color = my_color[a], linewidth = linewidth)

        ax[subplot, 1].set_ylim([0, 110])
        ax[subplot, 1].set_yticks([])
        if args.participant != 't16':
            for pos in ['right', 'top', 'left', 'bottom']: 
                ax[subplot, 1].spines[pos].set_visible(False) 
        elif args.participant == 't16':
            for pos in ['right', 'top', 'left']: 
                ax[subplot, 1].spines[pos].set_visible(False)
        if args.participant != 't16':
            ax[subplot, 1].set_xticks([])
        elif args.participant == 't16':
            ax[subplot, 1].set_xticks(np.arange(0, avg_go_thx.shape[1] + 1, 50), [k*10 if k != 0 else '' for k in (np.arange(0, avg_go_thx.shape[1] + 1, 50) - args.nbins_before_onset)], fontsize = fontsize, rotation = 45)
            if subplot != len(plt_channel) - 1:
                ax[subplot, 1].tick_params(labelbottom = False)
                

        ax[subplot, 1].scatter(args.nbins_before_onset, 3.5, color='black', s=scatter_size)
        if subplot == len(plt_channel) - 1:
            if args.participant == 't15':
                ax[subplot, 1].text(args.nbins_before_onset, -25, 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
            elif args.participant == 't16':
                ax[subplot, 1].text(args.nbins_before_onset, -29, 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
                ax[subplot, 1].set_xlabel('Time (ms)', fontsize = fontsize)
            
        if not args.plot_speech_offset:
            ax[subplot, 1].hlines(0, xmin = len(avg_go_thx[a, :, plt_channel[subplot]]) - 50, xmax = len(avg_go_thx[a, :, plt_channel[subplot]]), linewidth = 4, color = 'black')
            if subplot == len(plt_channel) - 1:
                if args.participant == 't15':
                    ax[subplot, 1].text(len(avg_go_thx[a, :, plt_channel[subplot]]) - 25, -14, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
                elif args.participant == 't16':
                    ax[subplot, 1].text(len(avg_go_thx[a, :, plt_channel[subplot]]) - 25, -14, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
        
        elif args.plot_speech_offset:
            # plot the 3rd column of psth w.r.t. speech offset, add speech offset and 500 ms marker
            # psth
            for a in range(len(amplitudes)):
                ax[subplot, 2].plot(avg_end_thx[a, :, plt_channel[subplot]].T, label = amplitudes[a], color = my_color[a], linewidth = linewidth)
                ax[subplot, 2].fill_between(np.arange(args.nbins_before_offset + args.nbins_after_offset),
                                    avg_end_thx[a, :, plt_channel[subplot]].T - sem_end_thx[a, :, plt_channel[subplot]].T,
                                    avg_end_thx[a, :, plt_channel[subplot]].T + sem_end_thx[a, :, plt_channel[subplot]].T,
                                    alpha = 0.5, label = '_hidden', color = my_color[a], linewidth = linewidth)

            ax[subplot, 2].set_ylim([0, 110])
            if args.participant != 't16':
                for pos in ['right', 'top', 'left', 'bottom']: 
                    ax[subplot, 2].spines[pos].set_visible(False) 
            elif args.participant == 't16':
                for pos in ['right', 'top', 'left']: 
                    ax[subplot, 2].spines[pos].set_visible(False) 
            ax[subplot, 2].set_yticks([])
            if args.participant != 't16':
                ax[subplot, 2].set_xticks([])
            elif args.participant == 't16':
                ax[subplot, 2].set_xticks(np.arange(0, avg_end_thx.shape[1], 25), [k*10 if k != 0 else '' for k in (np.arange(0, avg_end_thx.shape[1], 25) - args.nbins_before_offset)], fontsize = fontsize, rotation = 45) # ms labels
                if subplot != len(plt_channel) - 1:
                    ax[subplot, 2].tick_params(labelbottom = False)

            # speech offset marker
            ax[subplot, 2].scatter(args.nbins_before_offset, 3.5, color='black', s=scatter_size)
            if subplot == len(plt_channel) - 1:
                if args.participant == 't15':
                    ax[subplot, 2].text(args.nbins_before_offset, -25, 'Speech\noffset', color='black', ha='center', fontsize = fontsize)
                elif args.participant == 't16':
                    ax[subplot, 2].text(args.nbins_before_offset, -29, 'Speech\noffset', color='black', ha='center', fontsize = fontsize)

            # 500ms, plot it in the cue subplot
            if args.participant != 't16': # do not add 500ms marker to t16 supplementary plot in paper
                ax[subplot, 0].hlines(0, xmin = len(avg_delay_thx[a, :, plt_channel[subplot]]) - 50, xmax = len(avg_delay_thx[a, :, plt_channel[subplot]]), linewidth = 4, color = 'black')
            if subplot == len(plt_channel) - 1:
                if args.participant == 't15':
                    ax[subplot, 0].text(len(avg_delay_thx[a, :, plt_channel[subplot]]) - 25, -14, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)

        # electrode number
        ax[subplot, 1].text(-130, 70, f'Electrode {plt_channel[subplot] + 1} (array {arrays[args.participant][math.floor(plt_channel[subplot]/64)]})', color = 'darkorange', fontsize = fontsize)

    # legend
        legend_x_loc = args.nbins_before_onset + args.nbins_after_onset
        if args.plot_speech_offset:
            legend_x_loc = args.nbins_before_offset + args.nbins_after_offset
        if args.participant == 't15':
            for a in range(4):
                plt.text(legend_x_loc, 90 - a * 15, amplitudes[a], ha = "right", fontsize = fontsize, color = my_color[a])
        elif args.participant == 't16':
            for a in range(4):
                plt.text(legend_x_loc, 230 - a * 15, amplitudes[a], ha = "right", fontsize = fontsize, color = my_color[a])
        elif args.participant == 't19':
            for a in range(4):
                plt.text(legend_x_loc, 100 - a * 15, amplitudes[a], ha = "right", fontsize = fontsize, color = my_color[a])

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05) 
    plt.subplots_adjust(hspace=0.1)
    # plt.show()
    
    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_psth_{formatted_datetime}.png', format='png')

    return


def plot_psth_per_array(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, avg_end_thx, sem_end_thx, plt_channels):

    ch_sets_name = ch_set_names[args.participant]
    print(ch_sets_name)

    # stitch together delay, speech onset and speech offset periods for plotting
    # introduce X bins of NaNs between each period for visual separation
    n_nan_bins = 30
    avg_thx = np.concatenate((avg_delay_thx,
                                np.full((avg_go_thx.shape[0], n_nan_bins, n_channels), np.nan),
                                avg_go_thx,
                                np.full((avg_go_thx.shape[0], n_nan_bins, n_channels), np.nan),
                                avg_end_thx), axis = 1)
    sem_thx = np.concatenate((sem_delay_thx,
                                np.full((sem_go_thx.shape[0], n_nan_bins, n_channels), np.nan),
                                sem_go_thx,
                                np.full((sem_go_thx.shape[0], n_nan_bins, n_channels), np.nan),
                                sem_end_thx), axis = 1)

    # plotting positions of key events
    cue_onset_bin = bins_before_trial_end
    speech_onset_bin = cue_onset_bin + required_binned_delay_duration + n_nan_bins + args.nbins_before_onset    
    speech_offset_bin = speech_onset_bin + args.nbins_after_onset + n_nan_bins + args.nbins_before_offset

    print('bins cue, onset, offset', cue_onset_bin, speech_onset_bin, speech_offset_bin)

    for n, (ch_set, current_plotting_order) in enumerate(zip(ch_sets[args.participant], plotting_orders[args.participant])):

        fig, ax = plt.subplots(8, 8, figsize = (8, 8))

        for i, ch in enumerate(ch_set):
        # calculate row and col position
            row = current_plotting_order[i] // 8
            col = current_plotting_order[i] % 8
            
            # plot psth
            for a in range(len(amplitudes)):
                ax[row][col].plot(avg_thx[a, :, ch].T, color = my_color[a])


            ylim_max = int(np.nanmax(avg_thx[:, :, ch])) + 2
            ylim_annot = 10 # draw vertical bar up to 10 Hz
            if ylim_max < ylim_annot:
                ylim_max = ylim_annot
            ax[row][col].set_ylim([-2, ylim_max])
            ax[row][col].vlines(x = - n_nan_bins, ymin = 0, ymax = ylim_annot, color = 'black', linewidth = 1.5, alpha = 0.8)
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])

            for spine in ax[row][col].spines.values():
                spine.set_visible(False)
                if row == 7 and col == 0:
                    if n == 0:
                        # ax[row][col].spines['left'].set_visible(True)
                        ax[row][col].set_ylabel('10 Hz', fontsize = fontsize + 2)
                        ax[row][col].set_xticks([cue_onset_bin, speech_onset_bin, speech_offset_bin],['C', 'On', 'Off'], fontsize = fontsize + 4, rotation = 45)
                    else:
                        ax[row][col].set_ylabel('Firing rate (Hz)', fontsize = fontsize + 2) # set label color to white
                        ax[row][col].yaxis.label.set_color('white') 
                        ax[row][col].set_xticks([cue_onset_bin, speech_onset_bin, speech_offset_bin],['C', 'On', 'Off'], fontsize = fontsize + 4, rotation = 45)
                        ax[row][col].tick_params(axis='x', colors='white')

            # mark orange asterisk for channel in main figure
            if ch in plt_channels:
                ax[row][col].text(x = avg_thx.shape[1], y = ylim_max, s='*', fontsize=fontsize + 8, color = 'darkorange', fontweight = 'bold')

            trans1 = transforms.blended_transform_factory(ax[row][col].transData, ax[row][col].transAxes)

            # key event scatters
            ax[row][col].scatter(cue_onset_bin, 0.08, color='gray', s=scatter_size, transform=trans1, zorder=5) #int(scatter_size/2)*0+1
            ax[row][col].scatter(speech_onset_bin, 0.08, color='gray', s=scatter_size, transform=trans1, zorder=5)
            ax[row][col].scatter(speech_offset_bin, 0.08, color='gray', s=scatter_size, transform=trans1, zorder=5)
        
        fig.suptitle(f'{ch_sets_name[n]}', fontsize = fontsize + 5)
        fig.tight_layout()
        # plt.show()

        plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_{ch_sets_name[n][:3]}_psth_{formatted_datetime}.png', format='png')

    return
        

if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms', 'predaudio16k', 'raw_threshcross', 'raw_threshcross_from_pre_cue'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--nbins_before_offset', type=int, default=None, help='number of bins before speech offset')
    parser.add_argument('--nbins_after_offset', type=int, default=None, help='number of bins after speech offset')
    parser.add_argument('--plot_speech_offset', action='store_true', help='whether to plot psth around speech offset; default psth around cue and speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()
    
    if args.participant == 't16':
        args.required_keys.extend(['speech_onsets', 'speech_offsets'])
        args.required_keys.remove('predaudio16k')

    if not os.path.exists(args.savepath_data):
        os.makedirs(args.savepath_data, exist_ok = True)
    
    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok = True)

    print('Running psth.py')
    print(args)

    # load data
    data = load_rdbmat(args.participant, args.session, args.required_keys)

    # some data statistics
    n_trials = len(data['cue'])
    print('Total number of trials:', n_trials)
    
    if args.participant == 't15':
        print('Total number of valid word-amplitude trials:', len([i for i in range(n_trials) 
                                                                if 'DO NOTHING' not in data['cue'][i] and 
                                                                max(np.squeeze(data['predaudio16k'])[i]) != 0])) # trials that are not DO NOTHING or do not have b2v predictions (which is used to determine speech onset)
    elif args.participant in ['t16', 't19']:
        print('Total number of valid word-amplitude trials:', len([i for i in range(n_trials) 
                                                                if 'DO NOTHING' not in data['cue'][i]]))

    print('Total number of DO NOTHING trials:', len([i for i in range(n_trials) if 'DO NOTHING' in data['cue'][i]]))

    for amp in amplitudes:
        if args.participant == 't15':
            print(f'Number of valid {amp} trials:', len([i for i in range(n_trials) if amp in data['cue'][i]
                                                and max(np.squeeze(data['predaudio16k'])[i]) != 0]))
        elif args.participant in ['t16', 't19']:
            print(f'Number of valid {amp} trials:', len([i for i in range(n_trials) if amp in data['cue'][i]]))
        
    # compute psth
    print('Computing firing rates ...')
    avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, avg_end_thx, sem_end_thx = compute_psth(data)
    # save psth
    with open(f'{args.savepath_data}{args.participant}_{args.session}_psth_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump({
            'avg_go_thx': avg_go_thx,
            'sem_go_thx': sem_go_thx,
            'avg_delay_thx': avg_delay_thx,
            'sem_delay_thx': sem_delay_thx,
            'avg_end_thx': avg_end_thx,
            'sem_end_thx': sem_end_thx,
        }, f)

    # plot psth
    print('Plotting all channels ...')
    plot_psth_per_channel(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx)

    # plot particular channel psth
    print('Plotting particular channels ...')
    plt_channel = {
        't15': [197, 158, 38, 120], # channel (0-indexed), ordered according to implanted arrays
        't16': [57, 89], # channel (0-indexed), ordered according to implanted arrays (only speech arrays considered)
        't19': [23, 71, 146, 223] # channel (0-indexed)
    }
    plot_psth_given_channel(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, avg_end_thx, sem_end_thx, plt_channel[args.participant]) # 0-indexed channel

    # plot psth per array
    plot_psth_per_array(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, avg_end_thx, sem_end_thx, plt_channel[args.participant])

    # save scipt args
    print('Saving scripts args and script ...')
    script_name = sys.argv[0]
    with open(script_name, 'r') as f:
        script_content = f.read()

    with open(f'{args.savepath_data}{args.participant}_{args.session}_psth_{formatted_datetime}.log', 'w') as f:
        f.write(str(args))
        f.write('\n\n----------\n\n')
        f.write(script_content)

    print('DONE!')