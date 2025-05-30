import argparse
from datetime import datetime
import os
import scipy
import numpy as np
from pathlib import Path 
from session_metadata import incorrect_trials
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle as pkl
import sys
import math
import pandas as pd
import seaborn as sns
from scipy.stats import ranksums
from itertools import combinations

'''
example run cmd:
python instructed_breath_speech_breath_belt_analysis.py --participant <> --session <> --nbins_before_onset <> --nbins_after_onset <> --savepath_fig <> --savepath_data <>
'''

#--------------------------------------------
# global variables
#--------------------------------------------
breath_types = ['NORMALLY', 'DEEPLY']
speech_types = ['NORMAL', 'LOUD']
bin_size_ms = 10
fs = 30000
n_trials_confidence = 0.90 # number of trials needed at a particular time bin to confidently plot the breath at that time point

# gassian smoothing params
sigma = 4 # 1 == 10ms
order = 0

# plotting
scatter_size = 30
fontsize = 15
linewidth = 3
my_color_breath = [(230/255, 143/255, 172/255), (153/255, 15/255, 75/255)]
my_color_speech = [(86/255, 180/255, 233/255), (31/255, 120/255, 180/255)]

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


def get_aligned_breath(data):

    n_trials = len(data['cue'])
    breath_cycles_trough = {}
    # breath_cycles_peak_loc = {}
    breath_cycles_duration = {}
    breath_cycles_expansion = {}

    n_trials_breath_trials = 0
    for breath_type in breath_types:
        breath_cycles_trough[breath_type] = []
        # breath_cycles_peak_loc[breath_type] = []
        breath_cycles_duration[breath_type] = []
        breath_cycles_expansion[breath_type] = []   
        
        for tr in range(n_trials):
            if breath_type in data['cue'][tr]:
                print(data['cue'][tr])
                n_trials_breath_trials += 1

                # print(data['binned_breath'].shape)
                binned_breath = np.squeeze(data['binned_breath'])[tr].squeeze()
                binned_breath_troughs = np.squeeze(data['binned_breath_min_loc'])[tr].squeeze()
                binned_breath_peaks = np.squeeze(data['binned_breath_max_loc'])[tr].squeeze()

                # smooth the binned breath data
                smth_binned_breath = gaussian_filter1d(binned_breath, sigma = sigma, order = order)
                # print(binned_breath.shape, smth_binned_breath.shape)

                if args.participant == 't16':
                    for i in range(len(binned_breath_troughs) - 1):
                        breath_cycle = smth_binned_breath[binned_breath_troughs[i]: binned_breath_troughs[i+1]]
                        # print(f'Trial{tr+1} cycle {i+1}:', breath_cycle.shape, max(breath_cycle), min(breath_cycle))
                        breath_cycles_trough[breath_type].append(breath_cycle)
                        # breath_cycles_peak_loc[breath_type].append(binned_breath_peaks[i] - binned_breath_troughs[i])
                        breath_cycles_duration[breath_type].append(len(breath_cycle)) #10ms binned value
                        breath_cycles_expansion[breath_type].append(max(breath_cycle) - min(breath_cycle))
                elif args.participant == 't15': # belt was flipped
                    for i in range(len(binned_breath_peaks) - 1):
                        breath_cycle = smth_binned_breath[binned_breath_peaks[i]: binned_breath_peaks[i+1]]
                        # print(f'Trial{tr+1} cycle {i+1}:', breath_cycle.shape, max(breath_cycle), min(breath_cycle))
                        breath_cycles_trough[breath_type].append(breath_cycle)
                        # breath_cycles_peak_loc[breath_type].append(binned_breath_troughs[i] - binned_breath_peaks[i]) # peak relative to inhalation
                        breath_cycles_duration[breath_type].append(len(breath_cycle)) #10ms binned value
                        breath_cycles_expansion[breath_type].append(max(breath_cycle) - min(breath_cycle))

    # pad each binned breath with nan so that all breath trials are of the same length
    breath_cycles_trough_padded = {}
    for breath_type, breath_list in breath_cycles_trough.items():
        max_len = max(map(len, breath_list))
        padded_breath_list = np.array([row.tolist() + [np.nan] * (max_len - len(row)) for row in breath_list])

        # truncate all trials to time bin where there is an entry for only one trial and the rest is all nan
        n_nan = np.isnan(padded_breath_list).sum(axis = 0)
        ind = np.argwhere(n_nan > int(n_trials_confidence * len(padded_breath_list))).squeeze()[0]
        breath_cycles_trough_padded[breath_type] = padded_breath_list[:, :ind]

    print('Number of breath trials:', n_trials_breath_trials)
    print('INSTRUCTED BREATH')
    for breath in breath_types:
        # print(f'{breath} expansions:', breath_cycles_expansion[breath])
        print(f'{breath} min, max, mean, mode:', np.min(breath_cycles_expansion[breath]), np.max(breath_cycles_expansion[breath]), 
              np.mean(breath_cycles_expansion[breath]), np.median(breath_cycles_expansion[breath]))
        print(f'{breath} number of breaths:', len(breath_cycles_expansion[breath]))

    return breath_cycles_trough_padded, breath_cycles_duration, breath_cycles_expansion   


def get_speech_aligned_breath(data):
    n_trials = len(data['cue'])
    breath_wrt_speech_onset = {}
    breath_duration_wrt_speech_onset = {}
    breath_expansion_wrt_speech_onset = {}
    for speech_type in speech_types:
        breath_wrt_speech_onset[speech_type] = []
        breath_duration_wrt_speech_onset[speech_type] = []
        breath_expansion_wrt_speech_onset[speech_type] = []

    n_trials_sentence = 0

    for tr in range(n_trials):
        # if 'inhale and exhale' not in data['cue'][tr].lower():
            binned_breath = np.squeeze(data['binned_breath'])[tr].squeeze()
            cue = data['cue'][tr]
            if not('inhale and exhale' in cue.lower() or 'DO NOTHING' in cue):
                n_trials_sentence += 1
            
            words = cue.strip().split(' ')

            # smooth the binned breath data
            smth_binned_breath = gaussian_filter1d(binned_breath, sigma = sigma, order = order)

            # speech onset
            start_ind = np.squeeze(data['speech_onsets'])[tr].squeeze()
            end_ind = np.squeeze(data['speech_offsets'])[tr].squeeze()

            # binned start and end ind
            start_ind = [math.floor((s/fs) * (1000/bin_size_ms)) for s in start_ind]
            end_ind = [math.ceil((e/fs) * (1000/bin_size_ms)) for e in end_ind]
            # print(start_ind, end_ind)

            if args.participant == 't15': # peaks are the points of inhalation
                inhale_before_speech_onset = np.squeeze(data['breath_max_loc_before_speech_onset'])[tr].squeeze()
                inhale_after_speech_onset = np.squeeze(data['breath_max_loc_after_speech_onset'])[tr].squeeze()
            elif args.participant == 't16':
                inhale_before_speech_onset = np.squeeze(data['breath_min_loc_before_speech_onset'])[tr].squeeze()
                inhale_after_speech_onset = np.squeeze(data['breath_min_loc_after_speech_onset'])[tr].squeeze()
            # print(inhale_before_speech_onset, inhale_after_speech_onset)

            assert len(words) == len(start_ind) == len(end_ind)
            assert len(words) == len(inhale_before_speech_onset) == len(inhale_after_speech_onset)

            for i in range(len(start_ind)):
                if start_ind[i] != -1 and end_ind[i] != 0: # there is a valid annotation for this word
                    if start_ind[i] - args.nbins_before_onset >= 0 and start_ind[i] + args.nbins_after_onset < len(binned_breath):
                        
                        binned_breath_word = smth_binned_breath[start_ind[i] - args.nbins_before_onset:start_ind[i] + args.nbins_after_onset]
                        
                        # breath duration
                        breath_duration_word = -1
                        if inhale_before_speech_onset[i] != -1 and inhale_after_speech_onset[i] != -1:
                            breath_duration_word = inhale_after_speech_onset[i] - inhale_before_speech_onset[i]
                            binned_breath_word_cycle = smth_binned_breath[inhale_before_speech_onset[i]:inhale_after_speech_onset[i]]

                        
                        if words[i].isupper():
                            breath_wrt_speech_onset['LOUD'].append(binned_breath_word)
                            if breath_duration_word != -1:
                                breath_duration_wrt_speech_onset['LOUD'].append(breath_duration_word)
                                breath_expansion_wrt_speech_onset['LOUD'].append(max(binned_breath_word_cycle) - min(binned_breath_word_cycle))
                                # print('trial', tr, words[i], breath_duration_word, max(binned_breath_word_cycle) - min(binned_breath_word_cycle))
                            
                        elif words[i].islower():
                            breath_wrt_speech_onset['NORMAL'].append(binned_breath_word)
                            if breath_duration_word != -1:
                                breath_duration_wrt_speech_onset['NORMAL'].append(breath_duration_word)
                                breath_expansion_wrt_speech_onset['NORMAL'].append(max(binned_breath_word_cycle) - min(binned_breath_word_cycle))
                                # print('trial', tr, words[i], breath_duration_word, max(binned_breath_word_cycle) - min(binned_breath_word_cycle))
                            

    print('Number of sentence trials:', n_trials_sentence)
    print('SPEECH BREATH')
    for speech in speech_types:
        # print(f'{speech} expansions:', breath_expansion_wrt_speech_onset[speech])
        print(f'{speech} min, max, mean:', np.min(breath_expansion_wrt_speech_onset[speech]), np.max(breath_expansion_wrt_speech_onset[speech]), 
              np.mean(breath_expansion_wrt_speech_onset[speech]), np.median(breath_expansion_wrt_speech_onset[speech]))
        print(f'{speech} number of breaths:', len(breath_expansion_wrt_speech_onset[speech]))

    return breath_wrt_speech_onset, breath_duration_wrt_speech_onset, breath_expansion_wrt_speech_onset


def plot_joint_breath_expansion_statistics(breath_breath, breath_color, speech_breath, speech_color):
    
    data = {}
    for key in breath_breath.keys():
            data[key] = breath_breath[key]
    data.update(speech_breath)

    df = pd.DataFrame([(k, v) for k, vals in data.items() for v in vals], columns=["Condition", "Expansion"])
    # Create a box plot
    fig = plt.figure(figsize=(5, 6))
    positions = [0,1,2,3]
    colors = breath_color + speech_color

    ax = sns.boxplot(x="Condition", y="Expansion", data=df, fill = False, palette=colors, linewidth=2)

    # Perform pairwise t-tests and annotate significant comparisons
    group_names = list(data.keys())

    # Function to get significance level
    def get_p_label(p):
        if p < 0.05:
            return '*'
        else:
            return 'ns'  # Not significant

    # Add annotations for pairwise comparisons
    y_max = df["Expansion"].max() + 0.1  # Position for the highest annotation
    if args.participant == 't15':
        y_offset = 0.5  # Spacing between annotations
    elif args.participant == 't16':
        y_offset = 4
    for (g1, g2), y in zip([('NORMALLY', 'DEEPLY'), ('NORMAL', 'LOUD'), ('DEEPLY', 'NORMAL'), ('NORMALLY', 'NORMAL'), ('DEEPLY', 'LOUD'), ('NORMALLY', 'LOUD')], 
                           np.arange(y_max, y_max + y_offset * 6, y_offset)):
        print(g1, g2, y)
        # Perform t-test
        _ , p_value = ranksums(data[g1], data[g2])
        significance = get_p_label(p_value)

        # Get x-axis positions of groups
        x1, x2 = positions[group_names.index(g1)], positions[group_names.index(g2)]
        
        # Plot a line connecting the groups
        if args.participant == 't15':
            ax.plot([x1, x1, x2, x2], [y, y + 0.1, y + 0.1, y], lw=1.5, color="black")
        elif args.participant == 't16':
            ax.plot([x1, x1, x2, x2], [y, y + 0.5, y + 0.5, y], lw=1.5, color="black")
        
        # Add the significance text
        if args.participant == 't15':
            ax.text((x1 + x2) / 2, y + 0.1, significance, ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
        elif args.participant == 't16':
            ax.text((x1 + x2) / 2, y + 0.2, significance, ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


    for pos in ['right', 'top', 'bottom']: 
        plt.gca().spines[pos].set_visible(False) 
    plt.xticks(positions, list(data.keys()), fontsize = fontsize)
    plt.yticks([], [])
    plt.ylabel('Breath belt expansion (a.u.)', fontsize = fontsize)
    plt.xlabel('')

    # Add additional x-axis labels for grouped categories
    if args.participant == 't15':
        plt.text(0.5, 0.5, "Instructed breath", ha='center', fontsize=fontsize)
        plt.text(2.5, 0.5, "Attempted loudness", ha='center', fontsize=fontsize)
    elif args.participant == 't16':
        plt.text(0.5, -6, "Instructed breath", ha='center', fontsize=fontsize)
        plt.text(2.5, -6, "Attempted loudness", ha='center', fontsize=fontsize)

    plt.suptitle(f'{args.participant.upper()}', fontsize = fontsize)
    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_breath_expansion_plot_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_breath_expansion_plot_{formatted_datetime}.png', format='png')
    return


def plot_joint_breath_belt(breath_cycles, breath_colors, speech_cycles, speech_colors):

    fontsize = 17

    fig, ax = plt.subplots(1, 2, figsize=(8,5))
    max_avg_len = 0
    min_avg_value = 100000
    max_avg_value = -10000
    
    # breath subplot
    breath_peak_loc = 0
    for key in breath_cycles.keys():
        avg = np.nanmean(breath_cycles[key], axis = 0)
        sem = np.nanstd(breath_cycles[key], axis = 0) / np.sqrt(len(breath_cycles[key]))

        # align avg to zero, i.e. the minimum value of breath belt is 0
        avg -= np.min(avg)

        if len(avg) > max_avg_len: 
            max_avg_len = len(avg)
        if min(avg - sem) < min_avg_value:
            min_avg_value = min(avg - sem)
        if max(avg + sem) > max_avg_value:
            max_avg_value = max(avg + sem)

        key_label = key # legend

        ax[0].plot(avg, color = breath_colors[list(breath_cycles.keys()).index(key)], label = key_label, linewidth = linewidth)
        ax[0].fill_between(np.arange(len(avg)), avg - sem, avg + sem, alpha = 0.5, color = breath_colors[list(breath_cycles.keys()).index(key)], label = '_hidden') 

        if args.participant == 't16':
            breath_peak_loc = np.argmax(avg)
        elif args.participant == 't15':
            breath_peak_loc = np.argmin(avg)

    # speech subplot
    max_avg_len_speech = -1
    for key in speech_cycles.keys():
        avg = np.nanmean(speech_cycles[key], axis = 0)
        sem = np.nanstd(speech_cycles[key], axis = 0) / np.sqrt(len(speech_cycles[key]))

        # align avg to zero, i..e minimum breath belt value is 0
        avg -= np.min(avg)

        if len(avg) > max_avg_len: 
            max_avg_len = len(avg)
        if min(avg - sem) < min_avg_value:
            min_avg_value = min(avg - sem)
        if max(avg + sem) > max_avg_value:
            max_avg_value = max(avg + sem)
        if len(avg) > max_avg_len_speech:
            max_avg_len_speech = len(avg)

        key_label = key
        ax[1].plot(avg, color = speech_colors[list(speech_cycles.keys()).index(key)], label = key_label, linewidth = linewidth)
        ax[1].fill_between(np.arange(len(avg)), avg - sem, avg + sem, alpha = 0.5, color = speech_colors[list(speech_cycles.keys()).index(key)], label = '_hidden')


    for pos in ['right', 'top', 'bottom']: 
            ax[0].spines[pos].set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']: 
            ax[1].spines[pos].set_visible(False)
    
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[0].set_title('Instructed breath', fontsize = fontsize)
    ax[1].set_title('Attempted loudness', fontsize = fontsize)
    ax[0].set_ylim([min_avg_value - 0.1 * (max_avg_value - min_avg_value), max_avg_value])
    ax[1].set_ylim([min_avg_value - 0.1 * (max_avg_value - min_avg_value), max_avg_value])  
    if args.participant == 't15':
        ax[0].set_ylim([max_avg_value + 0.1 * (max_avg_value - min_avg_value), min_avg_value - 0.1 * (max_avg_value - min_avg_value)])
        ax[1].set_ylim([max_avg_value + 0.1 * (max_avg_value - min_avg_value), min_avg_value - 0.1 * (max_avg_value - min_avg_value)])
    
    ax[0].set_ylabel('Breath belt (a.u.)', fontsize = fontsize)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    if args.participant == 't16':
        ax[1].hlines(min_avg_value - 0.05 * (max_avg_value - min_avg_value), xmin = max_avg_len_speech - 50, xmax = max_avg_len_speech, linewidth = 2, color = 'black')
        ax[1].text(max_avg_len_speech - 25, min_avg_value - 0.15 * (max_avg_value - min_avg_value), '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
    elif args.participant == 't15':
        ax[1].hlines(max_avg_value + 0.05 * (max_avg_value - min_avg_value), xmin = max_avg_len_speech - 50, xmax = max_avg_len_speech, linewidth = 2, color = 'black')
        ax[1].text(max_avg_len_speech - 25, max_avg_value + 0.17 * (max_avg_value - min_avg_value), '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
    
    # mark speech onset
    if args.participant == 't16':
        ax[1].scatter(args.nbins_before_onset, min_avg_value - 0.05 * (max_avg_value - min_avg_value), color='black', s=scatter_size)
        ax[1].text(args.nbins_before_onset, min_avg_value - 0.21 * (max_avg_value - min_avg_value), 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
    elif args.participant == 't15':
        ax[1].scatter(args.nbins_before_onset, max_avg_value + 0.05 * (max_avg_value - min_avg_value), color='black', s=scatter_size)
        ax[1].text(args.nbins_before_onset, max_avg_value + 0.22 * (max_avg_value - min_avg_value), 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
        
        # mark exhalation onset
    if args.participant == 't16':
        ax[0].scatter(breath_peak_loc, min_avg_value - 0.05 * (max_avg_value - min_avg_value), color='black', s=scatter_size)
        ax[0].text(breath_peak_loc, min_avg_value - 0.21 * (max_avg_value - min_avg_value), 'Exhalation\nonset', color='black', ha='center', fontsize = fontsize)
    elif args.participant == 't15':
        ax[0].scatter(breath_peak_loc, max_avg_value + 0.05 * (max_avg_value - min_avg_value), color='black', s=scatter_size)
        ax[0].text(breath_peak_loc, max_avg_value + 0.22 * (max_avg_value - min_avg_value), 'Exhalation\nonset', color='black', ha='center', fontsize = fontsize)
       
    
    # if args.participant == 't15':
    #     ax[0].legend(fontsize = fontsize, loc = "lower left")
    #     ax[1].legend(fontsize = fontsize, loc = "lower left")

    plt.suptitle(f'{args.participant.upper()}', fontsize = fontsize)
    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_belt_plot_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_belt_plot_{formatted_datetime}.png', format='png')
    return

def plot_legend_only():
    # Create a figure and an axes
    fig, ax = plt.subplots()

    # Plot data with labels
    for i in range(len(my_color_breath)):
        ax.plot([1, 2, 3], [4, 5, 6], color = my_color_breath[i], label = 'breathe ' + breath_types[i], linewidth=linewidth)
    for i in range(len(my_color_speech)):
        ax.plot([1, 2, 3], [4, 5, 6], color = my_color_speech[i], label = speech_types[i] + ' loudness level', linewidth=linewidth)

    # Generate the legend
    handles, labels = ax.get_legend_handles_labels()

    # Clear the axes, but keep the figure
    ax.remove()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    # Add legend to the figure instead of the axes
    fig.legend(handles, labels, loc='center', fontsize=fontsize, ncol=1, frameon=True)

    # Display the figure
    # plt.show()
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_legend_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_legend_{formatted_datetime}.png', format='png')

    return

if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_data):
        os.mkdir(args.savepath_data)
    
    if not os.path.exists(args.savepath_fig):
        os.mkdir(args.savepath_fig)
    
    print('Running instructed_breath_speech_breath_belt_analysis.py')
    print(args)

    # Instructed breath trials
    # ----------------------------------------------------------
    # load only instructed breathing data
    print('Breath belt analysis for instructed breathing ...')
    required_keys = args.required_keys + ['binned_breath', 'binned_breath_min_loc', 'binned_breath_max_loc']
    print('Loading data ...')
    data = load_rdbmat(args.participant, args.session, required_keys)

    # get aligned breath data
    aligned_breath_trough, breath_cycles_duration, breath_cycles_expansion = get_aligned_breath(data)

    # plot breath belt a.u. for different breath types
    # plot_breath_belt(aligned_breath_trough, my_color_breath)

    # Speech sentence trials
    # -----------------------------------------------------------
    # load sentence speech breath data
    print('Breath belt analysis for speech ...')
    required_keys = args.required_keys + ['speech_onsets', 'speech_offsets', 'binned_breath']
    if args.participant == 't15':
        required_keys.extend(['breath_max_loc_before_speech_onset', 'breath_max_loc_after_speech_onset'])
    elif args.participant == 't16':
        required_keys.extend(['breath_min_loc_before_speech_onset', 'breath_min_loc_after_speech_onset'])
    print('Loading data ...')
    data = load_rdbmat(args.participant, args.session, required_keys)

    # get breath data aligned to speech
    aligned_breath_to_speech, breath_duration, breath_expansion = get_speech_aligned_breath(data)

    # load breath duration of speech and breath together and plot
    plot_joint_breath_belt(aligned_breath_trough, my_color_breath, aligned_breath_to_speech, my_color_speech)
    plot_joint_breath_expansion_statistics(breath_cycles_expansion, my_color_breath, breath_expansion, my_color_speech)
    plot_legend_only()