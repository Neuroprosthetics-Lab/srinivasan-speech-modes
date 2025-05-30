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
python instructed_breath_speech_psth.py --participant <> --session <> --nbins_before_onset <> --nbins_after_onset <> --savepath_fig <> --savepath_data <>
'''

#--------------------------------------------
# global variables
#--------------------------------------------
breath_types = ['NORMALLY', 'DEEPLY']
speech_types = ['NORMAL', 'LOUD']
bin_size_ms = 10
fs = 30000
n_channels = 256

# gassian smoothing params
sigma = 4 # 1 == 10ms
order = 0

# plotting
scatter_size = 30
fontsize = 13
my_color_breath = [(230/255, 143/255, 172/255), (153/255, 15/255, 75/255)]
my_color_speech = [(86/255, 180/255, 233/255), (31/255, 120/255, 180/255)]

arrays = {
    't15': ['v6v', 'M1', '55b', 'd6v'], # correct_electrode_mapping = 1
    't16': ['55b/PEF', '6v', 'HK1', 'HK2'],
}

#--------------------------------------------
# functions
#--------------------------------------------

def load_rdbmat(participant, session, required_keys):
    # load data, remove incorrect trials
    data_path = f'../data/{participant}/{session}/RedisMat_b2s_amplitude/' # t15.2023.11.04 has using_correct_electrode_mapping = 0
    files = os.listdir(data_path)

    data = {}

    for file in files:
        name, extension = os.path.splitext(file)
        omit_rdbmat = False
        if extension == '.mat':
            fullPath = str(Path(data_path, file).resolve())
            print(f'Loading {fullPath} ...')
            data_temp = scipy.io.loadmat(fullPath)
            # print(data_temp.keys())

            for key in required_keys:
                if key not in data_temp.keys():
                    print('Skipping this rdbmat ...')
                    omit_rdbmat = True
                    break
            
            if omit_rdbmat:
                continue

            # remove incorrect trials
            curr_block = int(np.squeeze(data_temp['block_number']))
            if session in incorrect_trials[participant]:
                remove_trials = incorrect_trials[participant][session][curr_block] # trial ids, 1-indexed
            else:
                remove_trials = []
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


def compute_breath_psth(data):

    # collect breath w.r.t. start of exhalation 
    # as we want to compare it with the activity during speech onset which happens during exhalation

    n_trials = len(data['cue'])

    avg_thx = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))
    sem_thx = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))

    breath_threshcross = {}
    for breath_type in breath_types:
        breath_threshcross[breath_type] = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))

    for breath_type in breath_types:
        for tr in range(n_trials):
            if breath_type in data['cue'][tr]:

                # get current data
                threshcross = np.squeeze(data['raw_threshcross'])[tr]
                delay_duration_ms = np.squeeze(data['delay_duration_ms'])[tr]
                binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)
                binned_breath_troughs = np.squeeze(data['binned_breath_min_loc'])[tr].squeeze()
                binned_breath_peaks = np.squeeze(data['binned_breath_max_loc'])[tr].squeeze(axis = 0) 

                # smooth threshcross
                threshcross_smth = gaussian_filter1d(threshcross.astype(float), sigma = sigma, order = order, axis = 0) # sigma controls smoothing, higher is more smoothed

                if args.participant == 't15':
                    for i in range(len(binned_breath_troughs) - 1): # exhalation start is trough
                        if binned_delay_duration + (binned_breath_troughs[i] - args.nbins_before_onset) >= 0 and binned_delay_duration + (binned_breath_troughs[i] + args.nbins_after_onset) <= threshcross.shape[0]:
                            temp_thx = threshcross_smth[binned_delay_duration + (binned_breath_troughs[i] - args.nbins_before_onset): binned_delay_duration + (binned_breath_troughs[i] + args.nbins_after_onset), :] * 100 # multiply by 100 for firing rate, shape (time_bins x 256)
                            breath_threshcross[breath_type] = np.append(breath_threshcross[breath_type], np.expand_dims(temp_thx, 0), axis = 0)
                elif args.participant == 't16':
                    for i in range(len(binned_breath_peaks)): # exhalation start is peak
                        if binned_delay_duration + (binned_breath_peaks[i] - args.nbins_before_onset) >= 0 and binned_delay_duration + (binned_breath_peaks[i] + args.nbins_after_onset) <= threshcross.shape[0]:
                            temp_thx = threshcross_smth[binned_delay_duration + (binned_breath_peaks[i] - args.nbins_before_onset): binned_delay_duration + (binned_breath_peaks[i] + args.nbins_after_onset), :] * 100 # multiply by 100 for firing rate, shape (time_bins x 256)
                            breath_threshcross[breath_type] = np.append(breath_threshcross[breath_type], np.expand_dims(temp_thx, 0), axis = 0)
                

        print(f'{breath_type} thx shapes:', breath_threshcross[breath_type].shape)
        
        avg_thx = np.append(avg_thx, np.expand_dims(np.mean(breath_threshcross[breath_type], axis = 0), axis = 0), axis = 0)
        sem_thx = np.append(sem_thx, np.expand_dims(np.std(breath_threshcross[breath_type], axis = 0) / np.sqrt(breath_threshcross[breath_type].shape[0]), axis = 0), 
                                        axis = 0)
    
    print('AVG and SEM thx shapes', avg_thx.shape, sem_thx.shape)
    
    return avg_thx, sem_thx



def compute_speech_psth(data):

    n_trials = len(data['cue'])

    avg_thx = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))
    sem_thx = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))

    speech_threshcross = {}
    for speech_type in speech_types:
        speech_threshcross[speech_type] = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))

    
    for tr in range(n_trials):
        # get current data
        threshcross = np.squeeze(data['raw_threshcross'])[tr]
        delay_duration_ms = np.squeeze(data['delay_duration_ms'])[tr]
        binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)
        cue = data['cue'][tr]
        words = cue.strip().split(' ')
        
        # speech onset
        start_ind = np.squeeze(data['speech_onsets'])[tr].squeeze()
        end_ind = np.squeeze(data['speech_offsets'])[tr].squeeze()

        # binned start and end ind
        start_ind = [math.floor((s/fs) * (1000/bin_size_ms)) for s in start_ind]
        end_ind = [math.ceil((e/fs) * (1000/bin_size_ms)) for e in end_ind]
        
        # smooth threshcross
        threshcross_smth = gaussian_filter1d(threshcross.astype(float), sigma = sigma, order = order, axis = 0) # sigma controls smoothing, higher is more smoothed

        assert len(words) == len(start_ind) == len(end_ind)

        for i in range(len(start_ind)):
            if start_ind[i] != -1 and end_ind[i] != 0: # there is a valid annotation for this word
                if binned_delay_duration + start_ind[i] - args.nbins_before_onset >= 0 and binned_delay_duration + start_ind[i] + args.nbins_after_onset < threshcross_smth.shape[0]:
                    temp_thx = threshcross_smth[binned_delay_duration + start_ind[i] - args.nbins_before_onset: binned_delay_duration + start_ind[i] + args.nbins_after_onset, :] * 100
                    if words[i].isupper():
                        speech_threshcross['LOUD'] = np.append(speech_threshcross['LOUD'], np.expand_dims(temp_thx, 0), axis = 0)
                    elif words[i].islower():
                        speech_threshcross['NORMAL'] = np.append(speech_threshcross['NORMAL'], np.expand_dims(temp_thx, 0), axis = 0)

    for speech_type in speech_types:
        print(f'{speech_type} thx shapes:', speech_threshcross[speech_type].shape)
        avg_thx = np.append(avg_thx, np.expand_dims(np.mean(speech_threshcross[speech_type], axis = 0), axis = 0), axis = 0)
        sem_thx = np.append(sem_thx, np.expand_dims(np.std(speech_threshcross[speech_type], axis = 0) / np.sqrt(speech_threshcross[speech_type].shape[0]), axis = 0), 
                                        axis = 0)
    
    return avg_thx, sem_thx



def plot_psth_per_channel(avg_breath_thx, sem_breath_thx, avg_speech_thx, sem_speech_thx):

    for i in range(n_channels):
        fig = plt.figure(figsize = (5,5))

        # breath psth
        for a in range(len(breath_types)):
            plt.plot(avg_breath_thx[a, :, i].T, color = my_color_breath[a], label = f'{breath_types[a]} breath')
            plt.fill_between(np.arange(len(avg_breath_thx[a, :, i].T)),
                                avg_breath_thx[a, :, i].T - sem_breath_thx[a, :, i].T,
                                avg_breath_thx[a, :, i].T + sem_breath_thx[a, :, i].T,
                                alpha = 0.5, label='_hidden', color = my_color_breath[a])
        
        # speech psth
        for a in range(len(speech_types)):
            plt.plot(avg_speech_thx[a, :, i].T, color = my_color_speech[a], label = f'{speech_types[a]} speech')
            plt.fill_between(np.arange(len(avg_speech_thx[a, :, i].T)),
                                avg_speech_thx[a, :, i].T - sem_speech_thx[a, :, i].T,
                                avg_speech_thx[a, :, i].T + sem_speech_thx[a, :, i].T,
                                alpha = 0.5, label='_hidden', color = my_color_speech[a])
        plt.ylim([0, 90])
        plt.yticks([0, 90], [0, 90], fontsize = fontsize)
        plt.ylabel('Firing rate (Hz)', fontsize = fontsize)
        plt.xticks([])

        for pos in ['right', 'top', 'bottom']: 
            plt.gca().spines[pos].set_visible(False) 

        plt.scatter(args.nbins_before_onset, 2, color='black', s=scatter_size)
        plt.text(args.nbins_before_onset, -8, 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
        plt.hlines(0, xmin = len(avg_breath_thx[a, :, i]) - 50, xmax = len(avg_breath_thx[a, :, i]), linewidth = 5, color = 'black')
        plt.text(len(avg_breath_thx[a, :, i]) - 25, -4, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)

        # legend
        plt.legend()
        plt.text(0, 60, f'Channel {i}\nElectrode {i%64 + 1} in {arrays[args.participant][math.floor(i/64)]}', color = 'red', fontsize = fontsize)
        # plt.show()

        fig.tight_layout()

        # save figure
        plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_channel_{i}_psth.png')
        
    return



def plot_psth_given_channel(avg_breath_thx, sem_breath_thx, avg_speech_thx, sem_speech_thx, plt_channel):

    # plot one example electrode per array
    fig, ax = plt.subplots(len(plt_channel), 1, figsize = (6, 6))

    for subplot in range(len(plt_channel)):

        # breath psth
        for a in range(len(breath_types)):
            ax[subplot].plot(avg_breath_thx[a, :, plt_channel[subplot]].T, color = my_color_breath[a], label = f'{breath_types[a]} breath')
            ax[subplot].fill_between(np.arange(len(avg_breath_thx[a, :, plt_channel[subplot]].T)),
                                avg_breath_thx[a, :, plt_channel[subplot]].T - sem_breath_thx[a, :, plt_channel[subplot]].T,
                                avg_breath_thx[a, :, plt_channel[subplot]].T + sem_breath_thx[a, :, plt_channel[subplot]].T,
                                alpha = 0.5, label='_hidden', color = my_color_breath[a])
        
        # speech psth
        for a in range(len(speech_types)):
            ax[subplot].plot(avg_speech_thx[a, :, plt_channel[subplot]].T, color = my_color_speech[a], label = f'{speech_types[a]} speech')
            ax[subplot].fill_between(np.arange(len(avg_speech_thx[a, :, plt_channel[subplot]].T)),
                                avg_speech_thx[a, :, plt_channel[subplot]].T - sem_speech_thx[a, :, plt_channel[subplot]].T,
                                avg_speech_thx[a, :, plt_channel[subplot]].T + sem_speech_thx[a, :, plt_channel[subplot]].T,
                                alpha = 0.5, label='_hidden', color = my_color_speech[a])
        
        ax[subplot].set_ylim([0, 100])
        if subplot == 0:
            ax[subplot].set_yticks([0, 100], [0, 100], fontsize = fontsize)
            ax[subplot].set_ylabel('Firing rate (Hz)', fontsize = fontsize)
        else:
            ax[subplot].set_yticks([])
        ax[subplot].set_xticks([])

        for pos in ['right', 'top', 'bottom']: 
            ax[subplot].spines[pos].set_visible(False)
        

        ax[subplot].scatter(args.nbins_before_onset, 3, color='black', s=scatter_size)
        ax[subplot].text(0, 85, f'Electrode {plt_channel[subplot] + 1} (array {arrays[args.participant][math.floor(plt_channel[subplot]/64)]})', color = 'black', fontsize = fontsize)

        if subplot == 0:
            ax[subplot].text((args.nbins_before_onset + args.nbins_after_onset)/2, 105, f'{args.participant.upper()}', color='black', ha='center', fontsize = fontsize + 2)

    
    ax[subplot].text(args.nbins_before_onset, -25, 'Speech onset\n(Exhalation onset)', color='black', ha='center', fontsize = fontsize)
    ax[subplot].hlines(0, xmin = len(avg_breath_thx[a, :, plt_channel[subplot]]) - 50, xmax = len(avg_breath_thx[a, :, plt_channel[subplot]]), linewidth = 5, color = 'black')
    ax[subplot].text(len(avg_breath_thx[a, :, plt_channel[subplot]]) - 25, -10, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
    
    # legend
    # breath_types[0] = 'REGULAR' # for legend purpose
    legend_label = [f'{b} breathe' for b in breath_types ] + [f'{s} loudness' for s in speech_types]
    legend_color = my_color_breath + my_color_speech

    # plot legend only for t15
    # if args.participant == 't15':
    #     for a in range(4):
    #         ax[2].text(250, 90 - a * 10, legend_label[a], ha = "right", fontsize = fontsize, color = legend_color[a])
    
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05) 
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
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms', 'raw_threshcross'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_data):
        os.mkdir(args.savepath_data)
    
    if not os.path.exists(args.savepath_fig):
        os.mkdir(args.savepath_fig)

    print('Running significant_loudness_channels.py')
    print(args)
    
    # Instructed breath trials
    # ----------------------------------------------------------
    # load only instructed breathing data
    print('Psth for speech ...')
    required_keys = args.required_keys + ['binned_breath_min_loc', 'binned_breath_max_loc']
    print('Loading data ...')
    data = load_rdbmat(args.participant, args.session, required_keys)

    # get psth
    avg_breath_thx, sem_breath_thx = compute_breath_psth(data)

    # Speech trials
    # ----------------------------------------------------------
    # load only speech data
    print('Breath belt analysis for instructed breathing ...')
    required_keys = args.required_keys + ['speech_onsets', 'speech_offsets']
    print('Loading data ...')
    data = load_rdbmat(args.participant, args.session, required_keys)

    # get psth
    avg_speech_thx, sem_speech_thx = compute_speech_psth(data)

    # plot psth
    print('Plotting all channels ...')
    plot_psth_per_channel(avg_breath_thx, sem_breath_thx, avg_speech_thx, sem_speech_thx)

    # # plot particular channel psth
    print('Plotting particular channels ...')
    plt_channel = {
        't15': [141, 97, 34], # channel (0-indexed), ordered according to implanted arrays
        't16': [5, 67, 36], # channel (0-indexed), ordered according to implanted arrays (only speech arrays considered)
    }
    plot_psth_given_channel(avg_breath_thx, sem_breath_thx, avg_speech_thx, sem_speech_thx, plt_channel[args.participant]) # 0-indexed channel