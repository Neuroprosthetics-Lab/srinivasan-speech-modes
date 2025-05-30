import argparse
import os
import scipy
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from session_metadata import incorrect_trials
from functions import get_audio_onset_offset
import pickle as pkl
from datetime import datetime
import sys

'''
Example run command:
python psth.py --participant <> --session t15.2023.11.04 --nbins_before_onset <> --nbins_after_onset <> --savepath_fig <>
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------

bin_size_ms = 10
fs = 30000
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
n_channels = 256
bins_before_trial_end = 10
delay_onset_raw_threshcross_from_pre_cue = 100 # rdbmat has 1s of raw thx before trial start 
required_binned_delay_duration = 100 # required delay duration bins for plotting

# gassian smoothing params
sigma = 4 # 1 == 10ms
order = 0

# plotting
my_color = [
    (167/255, 185/255, 207/255),
    (114/255, 159/255, 207/255),
    (53/255, 126/255, 221/255),
    (0, 79/255, 158/255),
]
fontsize = 12
scatter_size = 30
delay_duration = 100
linewidth = 2

arrays = {
    't15': ['M1', 'v6v','d6v','55b'], # correct_electrode_mapping = 0
    't16': ['55b/PEF', '6v', 'HK1', 'HK2'],
}

#--------------------------------------------
# functions
#--------------------------------------------

def load_rdbmat(participant, session, required_keys):
    # load data, remove incorrect trials
    data_path = f'../data/{participant}/{session}/RedisMat_b2s_amplitude/' # t15.2023.11.04 has using_correct_electrode_mapping = 0
    files = os.listdir(data_path)
    print(data_path, files)

    data = {}

    for file in files:
        name, extension = os.path.splitext(file)
        if extension == '.mat':
            fullPath = str(Path(data_path, file).resolve())
            print(f'Loading {fullPath} ...')
            data_temp = scipy.io.loadmat(fullPath)

            print(data_temp.keys())

            # remove incorrect trials
            curr_block = int(np.squeeze(data_temp['block_number']))
            remove_trials = incorrect_trials[participant][session][curr_block] # trial ids, 1-indexed
            print(f'Removing trials {remove_trials}')
            remove_trial_inds = [i-1 for i in remove_trials] # trial indices, 0-indexed

            data_temp_required = {}
            for key in required_keys:
                data_temp_required[key] = np.delete(data_temp[key], remove_trial_inds, axis = -1)

            # append data to master dict
            if data == {}:
                data = data_temp_required
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

    for amp in amplitudes:

        go_thx = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))
        delay_thx = np.empty((0, bins_before_trial_end + required_binned_delay_duration, n_channels))

        if args.participant == 't15':
            inds = [i for i in range(len(data['cue'])) if amp in data['cue'][i] and 
                    max(np.squeeze(data['predaudio16k'])[i]) != 0]
        elif args.participant == 't16':
            inds = [i for i in range(len(data['cue'])) if amp in data['cue'][i]]
        
        for ind in inds:
            # get current data
            prev_threshcross = np.squeeze(data['raw_threshcross_from_pre_cue'])[ind]
            threshcross = np.squeeze(data['raw_threshcross'])[ind]
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
            prev_threshcross_smth = gaussian_filter1d(prev_threshcross.astype(float), sigma = sigma, order = order, axis = 0) # sigma controls smoothing, higher is more smoothed
            threshcross_smth = gaussian_filter1d(threshcross.astype(float), sigma = sigma, order = order, axis = 0) # sigma controls smoothing, higher is more smoothed

            if np.expand_dims(threshcross[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :], 0).shape[1] == args.nbins_before_onset + args.nbins_after_onset:
                # go period threshcross
                temp_thx = threshcross_smth[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :] * 100 # multiply by 100 for firing rate, shape (time_bins x 256)
                go_thx = np.append(go_thx, np.expand_dims(temp_thx, 0), axis = 0)
            
                # delay period threshcross, with some threshold crossings before cue onset
                temp_thx = prev_threshcross_smth[(delay_onset_raw_threshcross_from_pre_cue - bins_before_trial_end): (delay_onset_raw_threshcross_from_pre_cue + required_binned_delay_duration), :] * 100 # multiply by 100 for firing rate, shape (time_bins x 256); first 200 bins after trial start
                delay_thx = np.append(delay_thx, np.expand_dims(temp_thx, axis = 0), axis = 0)

        print(f'{amp} go and delay thx shapes:', go_thx.shape, delay_thx.shape)
                
        avg_go_thx = np.append(avg_go_thx, np.expand_dims(np.mean(go_thx, axis = 0), axis = 0), axis = 0)
        sem_go_thx = np.append(sem_go_thx, np.expand_dims(np.std(go_thx, axis = 0) / np.sqrt(go_thx.shape[0]), axis = 0), 
                                        axis = 0)
        avg_delay_thx = np.append(avg_delay_thx, np.expand_dims(np.mean(delay_thx, axis = 0), axis = 0), axis = 0)
        sem_delay_thx = np.append(sem_delay_thx, np.expand_dims(np.std(delay_thx, axis = 0) / np.sqrt(delay_thx.shape[0]), axis = 0),
                                    axis = 0)
        
    print('AVG and SEM thx shapes', avg_go_thx.shape, sem_go_thx.shape, avg_delay_thx.shape, sem_delay_thx.shape)
    return avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx



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
        plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_channel_{i}_psth.png')
        
    return



def plot_psth_given_channel(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, plt_channel):

    # plot one example electrode per array
    if args.participant == 't15':
        fontsize = 14
        fig, ax = plt.subplots(len(plt_channel), 2, figsize = (6, 8), 
                            gridspec_kw={'width_ratios': [bins_before_trial_end + required_binned_delay_duration, args.nbins_before_onset + args.nbins_after_onset]}) # 'width_ratios': [n_time_bins_plotted in left subplot, n_time_bins plotted in right subplot]
    elif args.participant == 't16':
        fontsize = 14
        fig, ax = plt.subplots(len(plt_channel), 2, figsize = (6, 4), 
                            gridspec_kw={'width_ratios': [bins_before_trial_end + required_binned_delay_duration, args.nbins_before_onset + args.nbins_after_onset]}) # 'width_ratios': [n_time_bins_plotted in left subplot, n_time_bins plotted in right subplot]
    
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
        ax[subplot, 0].set_xticks([])

        for pos in ['right', 'top', 'bottom']: 
            ax[subplot, 0].spines[pos].set_visible(False) 

        ax[subplot, 0].scatter(bins_before_trial_end, 3.5, color='black', s=scatter_size)
        if subplot == len(plt_channel) - 1:
            ax[subplot, 0].text(bins_before_trial_end, - 12, 'Cue', color='black', ha = 'center', fontsize = fontsize)

        # go period
        for a in range(len(amplitudes)):
            ax[subplot, 1].plot(avg_go_thx[a, :, plt_channel[subplot]].T, label = amplitudes[a], color = my_color[a], linewidth = linewidth)
            ax[subplot, 1].fill_between(np.arange(args.nbins_before_onset + args.nbins_after_onset),
                                avg_go_thx[a, :, plt_channel[subplot]].T - sem_go_thx[a, :, plt_channel[subplot]].T,
                                avg_go_thx[a, :, plt_channel[subplot]].T + sem_go_thx[a, :, plt_channel[subplot]].T,
                                alpha = 0.5, label = '_hidden', color = my_color[a], linewidth = linewidth)

        ax[subplot, 1].set_ylim([0, 110])
        for pos in ['right', 'top', 'left', 'bottom']: 
            ax[subplot, 1].spines[pos].set_visible(False) 
        ax[subplot, 1].set_yticks([])
        ax[subplot, 1].set_xticks([])

        ax[subplot, 1].scatter(args.nbins_before_onset, 3.5, color='black', s=scatter_size)
        if subplot == len(plt_channel) - 1:
            if args.participant == 't15':
                ax[subplot, 1].text(args.nbins_before_onset, -25, 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
            elif args.participant == 't16':
                ax[subplot, 1].text(args.nbins_before_onset, -25, 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
        ax[subplot, 1].hlines(0, xmin = len(avg_go_thx[a, :, plt_channel[subplot]]) - 50, xmax = len(avg_go_thx[a, :, plt_channel[subplot]]), linewidth = 4, color = 'black')
        if subplot == len(plt_channel) - 1:
            if args.participant == 't15':
                ax[subplot, 1].text(len(avg_go_thx[a, :, plt_channel[subplot]]) - 25, -14, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
            elif args.participant == 't16':
                ax[subplot, 1].text(len(avg_go_thx[a, :, plt_channel[subplot]]) - 25, -14, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)

        
        ax[subplot, 1].text(-130, 70, f'Electrode {plt_channel[subplot] + 1} (array {arrays[args.participant][math.floor(plt_channel[subplot]/64)]})', color = 'darkorange', fontsize = fontsize)

    # legend
        if args.participant == 't15':
            for a in range(4):
                plt.text(250, 90 - a * 15, amplitudes[a], ha = "right", fontsize = fontsize, color = my_color[a])
        elif args.participant == 't16':
            for a in range(4):
                plt.text(250, 230 - a * 15, amplitudes[a], ha = "right", fontsize = fontsize, color = my_color[a])

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05) 
    plt.subplots_adjust(hspace=0.1)
    # plt.show()
    
    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_psth_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_psth_{formatted_datetime}.png', format='png')

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
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()
    
    if args.participant == 't16':
        args.required_keys.extend(['speech_onsets', 'speech_offsets'])
        args.required_keys.remove('predaudio16k')
        fontsize = 15

    if not os.path.exists(args.savepath_data):
        os.mkdir(args.savepath_data)
    
    if not os.path.exists(args.savepath_fig):
        os.mkdir(args.savepath_fig)

    print('Running significant_loudness_channels.py')
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
    elif args.participant == 't16':
        print('Total number of valid word-amplitude trials:', len([i for i in range(n_trials) 
                                                                if 'DO NOTHING' not in data['cue'][i]]))

    print('Total number of DO NOTHING trials:', len([i for i in range(n_trials) if 'DO NOTHING' in data['cue'][i]]))

    for amp in amplitudes:
        if args.participant == 't15':
            print(f'Number of valid {amp} trials:', len([i for i in range(n_trials) if amp in data['cue'][i]
                                                and max(np.squeeze(data['predaudio16k'])[i]) != 0]))
        elif args.participant =='t16':
            print(f'Number of valid {amp} trials:', len([i for i in range(n_trials) if amp in data['cue'][i]]))
        
    # compute psth
    print('Computing firing rates ...')
    avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx = compute_psth(data)
    # save psth
    with open(f'{args.savepath_data}{args.participant}_{args.session}_psth_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump({
            'avg_go_thx': avg_go_thx,
            'sem_go_thx': sem_go_thx,
            'avg_delay_thx': avg_delay_thx,
            'sem_delay_thx': sem_delay_thx
        }, f)

    # plot psth
    print('Plotting all channels ...')
    plot_psth_per_channel(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx)

    # plot particular channel psth
    print('Plotting particular channels ...')
    plt_channel = {
        't15': [197, 158, 38, 120], # channel (0-indexed), ordered according to implanted arrays
        't16': [57, 89], # channel (0-indexed), ordered according to implanted arrays (only speech arrays considered)
    }
    plot_psth_given_channel(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, plt_channel[args.participant]) # 0-indexed channel
