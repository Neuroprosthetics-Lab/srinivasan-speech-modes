import argparse
import scipy
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import math
import itertools
from session_metadata import incorrect_trials
from functions import get_audio_onset_offset
from scipy.stats import f_oneway, tukey_hsd
import matplotlib.pyplot as plt
import pickle as pkl
import sys

'''
Example run command:
python significant_loudness_channels.py --participant <> --session <> --nbins_before_onset <> --nbins_after_onset <>
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------

bin_size_ms = 10
fs = 30000
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
n_channels = 256

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
        if extension == '.mat':
            fullPath = str(Path(data_path, file).resolve())
            print(f'Loading {fullPath} ...')
            data_temp = scipy.io.loadmat(fullPath)

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


def compute_firing_rate_per_channel(data):
    # compute average firing rate around speech onset, say -0.5s to 0.5s
    firing_rates = {}  # {channel: {amplitude: [average firing rate per trial]}

    speech_onset_ind_after_go_cue = [] # to determine average speech onset time from go cue, 
    # use this as a proxy speech onset time for DO NOTHING trials

    for ch in range(n_channels):
        firing_rates[ch] = {}
        for amp in amplitudes:
                firing_rates[ch][amp] = []
        firing_rates[ch]['DONOTHING'] = []

    if args.participant == 't15':
        speaking_inds = [i for i in range(len(data['cue'])) 
                        if 'DO NOTHING' not in data['cue'][i] and 
                        max(np.squeeze(data['predaudio16k'])[i]) != 0] # trials that are not DO NOTHING or do not have b2v predictions (which is used to determine speech onset)
    elif args.participant == 't16':
        speaking_inds = [i for i in range(len(data['cue'])) if 'DO NOTHING' not in data['cue'][i]]

    # firing rates for speech trials
    for ind in speaking_inds:

        # get current data
        threshcross = np.squeeze(data['raw_threshcross'])[ind]
        curr_amp = data['cue'][ind].split(':')[0]
        delay_duration_ms = np.squeeze(data['delay_duration_ms'])[ind]
        binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)

        if args.participant == 't15':
            predaudio16k = np.squeeze(data['predaudio16k'])[ind]
            start_ind, end_ind = get_audio_onset_offset(predaudio16k, display_audio = False, 
                                                                    mic_audio = None, cue = None, 
                                                                    intersegment_duration = 3000, 
                                                                    amplitude_percentage = 0.1) # returns ind at 30k
        elif args.participant == 't16':
            start_ind = np.squeeze(data['speech_onsets'])[ind]
            end_ind = np.squeeze(data['speech_offsets'])[ind]

        start_ind = math.floor((start_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index
        end_ind = math.ceil((end_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index

        if np.expand_dims(threshcross[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :], 0).shape[1] == args.nbins_before_onset + args.nbins_after_onset:
            temp_thx = threshcross[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :] * 100 # multiply by 100 for firing rate, shape (time_bins x 256)
            mean_temp_thx = np.mean(temp_thx, 0) # shape (256, )
            for ch in range(n_channels):
                firing_rates[ch][curr_amp].append(mean_temp_thx[ch])
            
            speech_onset_ind_after_go_cue.append(start_ind)


    # firing rates for non-speech DO NOTHING trials
    # determine the average speech onset time from go cue
    # use this average speech onset time as a proxy speech onset time for DO NOTHING trials
    # extract the firing rate around this proxy speech onset
    avg_speech_onset_ind_from_go = int(np.mean(speech_onset_ind_after_go_cue))
    silent_inds = [i for i in range(len(data['cue'])) if 'DO NOTHING' in data['cue'][i]]
    for ind in silent_inds:
        # get threshcross
        threshcross = np.squeeze(data['raw_threshcross'])[ind]
        start_ind = avg_speech_onset_ind_from_go

        if np.expand_dims(threshcross[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :], 0).shape[1] == args.nbins_before_onset + args.nbins_after_onset:
            temp_thx = threshcross[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :] * 100 # shape (time_bins x 256)
            mean_temp_thx = np.mean(temp_thx, 0) # shape (256, )
            for ch in range(n_channels):
                firing_rates[ch]['DONOTHING'].append(mean_temp_thx[ch])

    for amp in amplitudes:
        print('Number of {amp} trials considered:', len(firing_rates[0][amp]))
    print('Number of DO NOTHING trials considered:', len(firing_rates[0]['DONOTHING']))
    return firing_rates


def get_significant_loudness_tuning_per_channel(firing_rates):

    alpha = 0.05  # significance level

    # compute anova for each channel
    anova_results = {}  # {electrode: {amplitude: p_value}}
    for ch, rates in firing_rates.items():
        f_value, p_value = f_oneway(rates['DONOTHING'], rates['MIME'], rates['WHISPER'], 
                                               rates['NORMAL'], rates['LOUD']) # one-way anova
        # print(ch, f_value)
        anova_results[ch] = [f_value, p_value]
    # one-way anova: tests the null hypothesis that two or more groups have the same population mean.
    # one-way anova: do all the conditions have same firing rate? 


    # for the significantly tuned channels get post hoc pairwise comparison
    n_significant_amp_encoding = {} # {ch: value between 0 and combination(n_conditions, 2)}
    for ch, rates in firing_rates.items():
        # tuckey_obj = tukey_hsd(rates['DONOTHING'], rates['MIME'], rates['WHISPER'],
        #                        rates['NORMAL'], rates['LOUD']) # tuckey honestly significant difference
    # tuckey hsd: post hoc test used to compare the mean of each sample to the mean of each other sample.
    # tuckey hsd: pairwise comparison of all the conditions
        tuckey_obj = tukey_hsd(rates['MIME'], rates['WHISPER'],
                               rates['NORMAL'], rates['LOUD'])

        # also get the loudness-level pairs that had different firing rates
        count = 0; loudness_pairs = []
        for i in range(len(tuckey_obj.pvalue)):
            for j in range(len(tuckey_obj.pvalue[i])):
                if i != j and tuckey_obj.pvalue[i][j] < alpha:
                    count += 1 # (2,3) and (3,2) are counted separately
                    if [j, i] not in loudness_pairs: # pairs appear twice
                        loudness_pairs.append([i,j])
        
        # print(ch, count/2, loudness_pairs)
        n_significant_amp_encoding[ch] = [count/2, loudness_pairs] # key is channel, value is a tuple with count and list of loudness pairs, # each pair is counted twice, so divide count by 2

    return n_significant_amp_encoding, anova_results


def plot_significant_channels(ch_modulation_level, mark_channels):
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
            # print(ch, row, col)
            # ax[n].scatter(col, 7 - row, s = ch_modulation_level[ch]*30, 
            #               color = ['black' if ch_modulation_level[ch]!= len(list(itertools.combinations(amplitudes + ['DONOTHING'] , 2)))
            #                        else 'red']) # red color marks channels tuned to all pairs of loudness levels and DO NOTHING
            if ch_modulation_level[ch][0] == 0:
                ax[n].scatter(col, 7 - row, s = 4 * 30, facecolors='none', edgecolors='black')
            else:
                ax[n].scatter(col, 7 - row, s = ch_modulation_level[ch][0]*30, color = 'black')
                # ax[n].scatter(col, 7 - row, s = ch_modulation_level[ch]*30, 
                #           color = ['black' if ch_modulation_level[ch]!= len(list(itertools.combinations(amplitudes, 2)))
                #                    else 'green']) # red color marks channels tuned to all pairs of loudness levels
            
            if ch in mark_channels:
                ax[n].scatter(col, 7 - row, s = 12 * 30, facecolors='none', edgecolors='darkorange', linewidth = 2)
        
        ax[n].text(3.5,8, ch_set_names[args.participant][n], fontsize = fig_fontsize, ha = 'center')   

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    # plt.show()

    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_significant_channels_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_significant_channels_{formatted_datetime}.png', format='png')

    # legend
    x = [0.1, 0.1]
    y = [0.6, 0.3]
    n = ['No loudness tuning', 'Depth of loudness tuning']
    fig, ax = plt.subplots(figsize = (7,2))
    ax.scatter(x[0], y[0], s = 4 * 30, facecolors='none', edgecolors='black')
    ax.scatter(x[1], y[1], s = 4 * 30, color = 'black')
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i] + 0.05, y[i] - 0.05), fontsize = fig_fontsize)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.axis('off')
    fig.tight_layout()
    plt.show()

    # plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_significant_channels_legend_{formatted_datetime}.svg', format='svg', dpi=1200)
    # plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_significant_channels_legend_{formatted_datetime}.png', format='png')

    return



if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms', 'predaudio16k', 'raw_threshcross'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if args.participant == 't16':
        args.required_keys.extend(['speech_onsets', 'speech_offsets'])
        args.required_keys.remove('predaudio16k')

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

    # determine firing rate for each channel around speech onset
    print('Computing firing rates ...')
    firing_rates = compute_firing_rate_per_channel(data) # {channel: {amplitude: [average firing rate per trial]}

    # number of significant loundness level tuning per channel
    print('Computing number of significant loundness level tuning per channel ...')
    n_significant_amp_encoding, anova_results = get_significant_loudness_tuning_per_channel(firing_rates) 

    print("Anova results:")
    for ch, (f_value, p_value) in anova_results.items():
        print(f'Channel {ch}: F-value = {f_value:.2f}, p-value = {p_value:.4f}')

    # number of channels tuned to multiple loudness levels
    # print('Number of channels tuned to multiple loudness levels')
    n_ch_tuned_to_multiple_levels = 0
    for ch, (count, loudness_pairs) in n_significant_amp_encoding.items():
        loudness_levels = []
        for pair in loudness_pairs:
            loudness_levels.append(amplitudes[pair[0]])
            loudness_levels.append(amplitudes[pair[1]])
        # get unique loudness levels and number of times it appears in pairwise comparison
        level, count_loudness = np.unique(loudness_levels, return_counts = True)
        # at least two levels should appear two comparisons for an electrode to be considered tuned to multiple levels
        num_loudness_level_tuned = sum([1 for k in count_loudness if k >= 2])
        # print(ch, loudness_pairs, count_loudness, num_loudness_level_tuned)
        if num_loudness_level_tuned > 1:
            n_ch_tuned_to_multiple_levels += 1
        else:
            print(ch, 'not tuned to multiple levels', loudness_pairs, count_loudness, num_loudness_level_tuned)
    print("Number of channels tuned to multiple loudness levels:", n_ch_tuned_to_multiple_levels) # not absolutely correct given firing rate could modulate for multiple levels but their firing rates are not significantly different, see psth to confirm

    # plot significant channels
    print('Plotting channels with number of loudness level tuning ...')
    # same order as arrays above
    mark_channels = {
        't15': [197, 158, 38, 120], # channel (0-indexed), ordered according to implanted arrays
        't16': [57, 89], # channel (0-indexed), ordered according to implanted arrays (only speech arrays considered)
    }
    plot_significant_channels(n_significant_amp_encoding, mark_channels[args.participant])
