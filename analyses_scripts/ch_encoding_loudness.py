# This script determines the channels tuned to different loudness levels.
import argparse
import scipy
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import math
from scipy.stats import f_oneway, tukey_hsd
import matplotlib.pyplot as plt
import pickle as pkl
import sys

'''
Example cmd (when run from this directory; provide python script path appropriately if run from different directory):
For t15,
    python ch_encoding_loudness.py --participant t15 --session word-loudness --nbins_before_onset 50 --nbins_after_onset 50 --savepath_data ../analyses_figures_data/t15/word-loudness/ch_encoding_loudness/ --savepath_fig ../analyses_figures/t15/word-loudness/ch_encoding_loudness/
For t16,
    python ch_encoding_loudness.py --participant t16 --session word-loudness --nbins_before_onset 50 --nbins_after_onset 50 --savepath_data ../analyses_figures_data/t16/word-loudness/ch_encoding_loudness/ --savepath_fig ../analyses_figures/t16/word-loudness/ch_encoding_loudness/

Neural data will be loaded from ../analyses_data/{participant}_{session}/.
Results obtained from this script will be saved in savepath_data.
Figures generated from this script will be saved in savepath_fig.
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------

bin_size_ms = 10
fs = 30000
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
n_channels = 256
alpha = 0.05  # significance level
pre_delay_nbins_in_raw_threshcross = 100

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
scatter_size_scale = 30
color = (34/255, 54/255, 150/255)

#--------------------------------------------
# functions
#--------------------------------------------
def load_rdbmat(participant, session, required_keys):
    # load data
    data_path = f'../analyses_data/{participant}_{session}/'
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

    data['cue'] = np.squeeze(data['cue']) # squeeze cue numpy array shape

    print('Data loaded ...')
    for key in data:
        print(key, data[key].shape)

    return data



def compute_firing_rate_per_channel(data):
    # compute average firing rate around speech onset, say -0.5s to 0.5s
    firing_rates = {}  # {channel: {amplitude: [average firing rate per trial]}

    for ch in range(n_channels):
        firing_rates[ch] = {}
        for amp in amplitudes:
                firing_rates[ch][amp] = []

    speaking_inds = [i for i in range(len(data['cue'])) if 'DO NOTHING' not in data['cue'][i]]

    # firing rates for speech trials
    for ind in speaking_inds:

        # get current data
        threshcross = np.squeeze(data['raw_threshcross'])[ind]
        curr_amp = data['cue'][ind].split(':')[0]
        delay_duration_ms = np.squeeze(data['delay_duration_ms'])[ind]
        binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)

        threshcross = threshcross[pre_delay_nbins_in_raw_threshcross:, :] # extract from delay period

        start_ind = np.squeeze(data['speech_onsets'])[ind].squeeze() # w.r.t. go onset
        end_ind = np.squeeze(data['speech_offsets'])[ind].squeeze() # w.r.t. go onset
        start_ind = math.floor((start_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index
        end_ind = math.ceil((end_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index

        if threshcross[binned_delay_duration + start_ind - args.nbins_before_onset: binned_delay_duration + start_ind + args.nbins_after_onset, :].shape[0] == args.nbins_before_onset + args.nbins_after_onset:
            temp_thx = threshcross[binned_delay_duration + start_ind - args.nbins_before_onset: binned_delay_duration + start_ind + args.nbins_after_onset, :] * 100 # multiply by 100 for firing rate, shape (time_bins x 256)
            mean_temp_thx = np.mean(temp_thx, 0) # shape (256, )
            for ch in range(n_channels):
                firing_rates[ch][curr_amp].append(mean_temp_thx[ch])
            
    for amp in amplitudes:
        print(f'Number of {amp} trials:', len(firing_rates[0][amp]))

    return firing_rates


def get_significant_loudness_tuning_per_channel(firing_rates):

    # compute anova for each channel
    anova_results = {}  # {electrode: {amplitude: p_value}}
    for ch, rates in firing_rates.items():
        f_value, p_value = f_oneway(rates['MIME'], rates['WHISPER'], rates['NORMAL'], rates['LOUD']) # one-way anova
        anova_results[ch] = [f_value, p_value]

    # for the significantly tuned channels get post hoc pairwise comparison
    n_significant_amp_encoding = {} # {ch: value between 0 and combination(n_conditions, 2)}
    for ch, rates in firing_rates.items():
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

    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_significant_channels_{formatted_datetime}.png', format='png')

    return

def plot_legend():
    # legend
    n_pairs = math.comb(len(amplitudes), 2)
    x = np.arange(0, n_pairs + 1, 1)
    y = 0
    fig, ax = plt.subplots(figsize = (7,2))
    for i in range(n_pairs + 1): # 1 added to manage no tuning (0, empty circle)
        if i == 0:
            ax.scatter(x[i], y, s = 4 * scatter_size_scale, facecolors='none', edgecolors = color)
        else:
            ax.scatter(x[i], y, s = i * scatter_size_scale, color = color)

    txt = f'Number of loudness pairs with significantly different firing rates\n0 {n_pairs}'
    ax.annotate(txt, (x[0], y + 0.03), fontsize = 13)
    plt.axis('off')
    fig.tight_layout()
    # plt.show()
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_significant_channels_legend_{formatted_datetime}.png', format='png')

    return



if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms', 'speech_onsets', 'speech_offsets', 'raw_threshcross'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=50, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=50, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

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

    # determine firing rate for each channel around speech onset
    print('Computing firing rates ...')
    firing_rates = compute_firing_rate_per_channel(data) # {channel: {amplitude: [average firing rate per trial]}

    # number of significant loundness level tuning per channel
    print('Computing number of significant loundness level tuning per channel ...')
    n_significant_amp_encoding, anova_results = get_significant_loudness_tuning_per_channel(firing_rates) 

    # save significant channels
    with open(f'{args.savepath_data}{args.participant}_{args.session}_significant_channels_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump(n_significant_amp_encoding, f)

    # number of channels per array tuned to loudness (# of filled circles)
    tot_n_ch_tuned_to_loudness = 0
    tot_n_ch = 0
    for i in range(len(ch_sets[args.participant])):
        ch_set = ch_sets[args.participant][i]
        tot_n_ch += len(ch_set)
        n_ch_tuned_to_loudness = 0
        for ch in ch_set:
            count_pairs = n_significant_amp_encoding[ch][0]
            if count_pairs > 0:
                n_ch_tuned_to_loudness += 1

        tot_n_ch_tuned_to_loudness += n_ch_tuned_to_loudness
        
        print(f'% of loudness tuned channels in {ch_set_names[args.participant][i]}: {n_ch_tuned_to_loudness * 100/len(ch_set)}%')

    print(f'total % of loudness tuned channels: {tot_n_ch_tuned_to_loudness * 100/tot_n_ch}%')


    # plot significant channels
    print('Plotting channels with number of loudness level tuning ...')
    # same order as arrays above
    mark_channels = {
        't15': [197, 158, 38, 120], # channel (0-indexed), ordered according to implanted arrays
        't16': [57, 89], # channel (0-indexed), ordered according to implanted arrays (only speech arrays considered)
    }
    plot_significant_channels(n_significant_amp_encoding, mark_channels[args.participant])
    plot_legend()

    # save script args
    print('Saving scripts args and script ...')
    script_name = sys.argv[0]
    with open(script_name, 'r') as f:
        script_content = f.read()

    with open(f'{args.savepath_data}{args.participant}_{args.session}_significant_channels_{formatted_datetime}.log', 'w') as f:
        f.write(str(args))
        f.write('\n\n----------\n\n')
        f.write(script_content)

    print('DONE!')