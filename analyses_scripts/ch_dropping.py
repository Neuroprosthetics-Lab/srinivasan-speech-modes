import argparse
import os
import scipy
import numpy as np
import pickle as pkl
from pathlib import Path 
import math
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sys
import pandas as pd

'''
Example cmd (when run from this directory; provide python script path appropriately if run from different directory):
For t15,
    python ch_dropping.py --participant t15 --session word-loudness --nbins_before_onset 60 --nbins_after_onset 60 --savepath_fig ../analyses_figures/t15/word-loudness/ch_dropping/ --savepath_data ../analyses_figures_data/t15/word-loudness/ch_dropping/ --n_repeats 10
For t16,
    python ch_dropping.py --participant t16 --session word-loudness --nbins_before_onset 60 --nbins_after_onset 60 --savepath_fig ../analyses_figures/t16/word-loudness/ch_dropping/ --savepath_data ../analyses_figures_data/t16/word-loudness/ch_dropping/ --n_repeats 10

Neural data will be loaded from ../analyses_data/{participant}_{session}/.
Results obtained from this script will be saved in savepath_data.
Figures generated from this script will be saved in savepath_fig.
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
n_channels = {
    't15': 256,
    't16': 128
}
n_electrodes_per_array = 64
bin_size_ms = 10
fs = 30000
pre_delay_nbins_in_neural_feat = 100
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going']

arrays = {
    't15': ['M1', 'v6v','d6v','55b'], # correct_electrode_mapping = 0, which is 2023 sessions
    't16': ['55b', '6v']#, 'HK1', 'HK2'],
}

# plotting
fontsize = 17
my_color = 'navy'

#--------------------------------------------
# functions
#--------------------------------------------
def load_rdbmat(participant, session, required_keys):
    # load data
    data_path = f'../analyses_data/{participant}_{session}/' # t15.2023.11.04 has using_correct_electrode_mapping = 0
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


def get_training_data(data):

    # get data around speech onset
    x_spikepow = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels[args.participant])) # n_trials x time bins x channels
    x_threshcross = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels[args.participant])) # n_trials x time bins x channels
    y_word = np.empty((0, 1)) # n_trials x word label 
    y_amp = np.empty((0, 1)) # n_trials x amplitude label
    
    n_trials = len(data['cue'])
    valid_trial_inds = [i for i in range(n_trials) if 'DO NOTHING' not in data['cue'][i]]
    print('Total number of trials:', len(valid_trial_inds))

    for ind in valid_trial_inds:
        
        # current data
        cue = data['cue'][ind].strip()
        amp_label = [amplitudes.index(cue.split(':')[0])]
        word_label = [words.index(cue.split(':')[-1].strip())]
        spikepow = np.squeeze(data['spikepow'])[ind]
        threshcross = np.squeeze(data['threshcross'])[ind]
        delay_duration_ms = np.squeeze(data['delay_duration_ms'])[ind]
        binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)

        # get neural features from delay
        spikepow = spikepow[pre_delay_nbins_in_neural_feat:, :n_channels[args.participant]] # shape (time_bins x 256) or (time_bins x 128)
        threshcross = threshcross[pre_delay_nbins_in_neural_feat:, :n_channels[args.participant]] # shape (time_bins x 256) or (time_bins x 128)

        start_ind = np.squeeze(data['speech_onsets'])[ind].squeeze()
        end_ind = np.squeeze(data['speech_offsets'])[ind].squeeze()
        start_ind = math.floor((start_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index
        end_ind = math.ceil((end_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index

        if spikepow[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :].shape[0] == args.nbins_before_onset + args.nbins_after_onset:
            
            # add spikepow and threshcross around speech onset
            temp_spikepow = spikepow[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :] # shape (time_bins x 256)
            x_spikepow = np.append(x_spikepow, np.expand_dims(temp_spikepow, 0), axis = 0)

            temp_threshcross = threshcross[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :] # shape (time_bins x 256)
            x_threshcross = np.append(x_threshcross, np.expand_dims(temp_threshcross, 0), axis = 0)

            # add labels
            y_word = np.append(y_word, np.expand_dims(word_label, 0), axis = 0)
            y_amp = np.append(y_amp, np.expand_dims(amp_label, 0), axis = 0)

    print('Spikepow, threshcross and label shapes for model:', x_spikepow.shape, x_threshcross.shape, y_word.shape, y_amp.shape)

    return x_spikepow, x_threshcross, y_word, y_amp



def train_logistic_regression(x_spikepow, x_threshcross, y_word, y_amp):

    n_channels_to_keep = np.arange(10, x_spikepow.shape[-1], 10)
    n_channels_to_keep = np.append(n_channels_to_keep, x_spikepow.shape[-1]) # all channels

    print('Channels to keep:', n_channels_to_keep)

    results = {} # key: channel_dropped, value: list of performance with random number of channels selected for channel_dropped
    # first n_fold entries in the dict values belong to first repetition (same channels dropped across n_folds)

    # cross validation (each fold has data from a word, test on unseen word)
    for channel_kept in n_channels_to_keep:
        results[channel_kept] = []

        for repeat in range(args.n_repeats):
            print(f'Channel kept {channel_kept}, Repetition {repeat}')
            # drop required number of channels randomly
            if x_spikepow.shape[-1] == n_electrodes_per_array: # for a particular array, no sampling strategy
                keep_channels = np.random.choice(x_spikepow.shape[-1], channel_kept, replace=False)
            elif x_spikepow.shape[-1] == n_channels[args.participant]: # uniformly sample across all arrays, i.e. lose equal number of electrodes per array
                keep_channels = []
                for arr in range(len(arrays[args.participant])):
                    arr_start_idx = arr * n_electrodes_per_array
                    keep_channels.extend(np.random.choice(np.arange(arr_start_idx, arr_start_idx + n_electrodes_per_array), channel_kept // len(arrays[args.participant]), replace=False))
 
            random_seed = np.random.randint(0, 10000)

            for fold in range(len(words)):
                print('Fold ', fold, words[fold])

                # inds in this fold
                test_inds = np.argwhere(y_word.squeeze() == fold).squeeze()
                train_inds = np.argwhere(y_word.squeeze() != fold).squeeze()

                # check non-overlap between train and test inds
                for ind in test_inds:
                    assert ind not in train_inds
                for ind in train_inds:
                    assert ind not in test_inds

                # test data
                x_test_spikepow = x_spikepow[test_inds, :, :]
                x_test_threshcross = x_threshcross[test_inds, :, :]
                x_test = np.concatenate([x_test_spikepow[:, :, keep_channels], x_test_threshcross[:, :, keep_channels]], axis = -1)
                y_test = y_amp[test_inds, :]

                # train data
                x_train_spikepow = x_spikepow[train_inds, :, :]
                x_train_threshcross = x_threshcross[train_inds, :, :]
                x_train = np.concatenate([x_train_spikepow[:, :, keep_channels], x_train_threshcross[:, :, keep_channels]], axis = -1)
                y_train = y_amp[train_inds, :].squeeze()

                # build and fit a classifier with a the seed for this repeat cycle
                clf = LogisticRegression(max_iter = 1000, random_state = random_seed) # random seed doesn't chance performance!!! as it is the same data
                clf.fit(np.reshape(x_train, (x_train.shape[0],-1)), y_train)
                y_pred = clf.predict(np.reshape(x_test, (x_test.shape[0], -1)))

                acc_binary = y_test.squeeze() == y_pred # binary values for correct and incorrect predictions
                acc = np.sum(acc_binary) / len(acc_binary)
                print(f'Channel kept: {channel_kept}, Repetition: {repeat}, Fold: {fold}, Word: {words[fold]}, Accuracy: {acc}')
                results[channel_kept].append(acc)

            if channel_kept == 256:
                continue # you do not have to repeat when no channel is dropped

    print('Accuracy across channel dropped, folds and repetitions:', results)
    print('Mean Accuracy across folds and repetitions:')
    for ch_kept, acc in results.items():
        print(ch_kept, np.mean(acc))

    return results


def plot_channel_dropping_curve(accuracies):

    for array, ch_accuracy in accuracies.items():

        n_ch_kept = list(ch_accuracy.keys())
        acc = [np.mean(ch_acc) for ch_acc in ch_accuracy.values()]
        std = [np.std(ch_acc) for ch_acc in ch_accuracy.values()]
        
        fig = plt.figure(figsize = (5,5))
        plt.plot(n_ch_kept, acc, 'o', markersize=6, markerfacecolor = my_color, markeredgecolor = my_color, markeredgewidth=2)
        plt.errorbar(n_ch_kept, acc, yerr = std, fmt = 'none', ecolor = my_color, elinewidth = 1, capsize = 2)

        plt.ylim([0, 1])
        plt.yticks(np.arange(0, 1.1, 0.2), np.arange(0, 101, 20), fontsize = fontsize)
        plt.ylabel('Accuracy (%)', fontsize = fontsize)
        if array == 'all':
            plt.xticks(n_ch_kept[0::4], n_ch_kept[0::4], fontsize = fontsize, rotation = 90)
        else:
            plt.xticks(n_ch_kept, n_ch_kept, fontsize = fontsize, rotation = 90)
        plt.xlabel('Number of electrodes', fontsize = fontsize)
        for pos in ['right', 'top']: 
            plt.gca().spines[pos].set_visible(False) 
        fig.tight_layout()
        # plt.show()

        # save figure
        plt.savefig(f'{args.savepath_fig}ch_dropping_{formatted_datetime}.png', format='png')

    return


if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms', 'spikepow', 'threshcross', 'speech_onsets', 'speech_offsets'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=60, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=60, help = 'number of bins after speech onset')
    parser.add_argument('--n_repeats', type=int, default=10, help = 'number of times each fold is modeled with a different random seed, or chance is computed per fold')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_data):
        os.mkdir(args.savepath_data)
    
    if not os.path.exists(args.savepath_fig):
        os.mkdir(args.savepath_fig)
    
    print('Running channel_dropping_performance.py')
    print(args)

    # load data
    print('Loading data ...')
    data = load_rdbmat(args.participant, args.session, args.required_keys)

    # get training data
    print('Loading training data ...')
    x_spikepow, x_threshcross, y_word, y_amp = get_training_data(data)
    # save training data
    with open(f'{args.savepath_data}{args.participant}_{args.session}_channel_dropping_processed_data_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump({
            'x_spikepow': x_spikepow,
            'x_threshcross': x_threshcross,
            'y_word': y_word,
            'y_amp': y_amp
        }, f)


    channel_dropped_accuracies = {}    
    print('Number of repeats per fold (or number of chance computations per fold):', args.n_repeats)

    # # logistic regression classification - all arrays
    #--------------------------------------------------
    print('All arrays: Training logistic regression')
    acc = train_logistic_regression(x_spikepow, x_threshcross, y_word, y_amp) # n_acc_values == n_folds 
    channel_dropped_accuracies['all'] = acc

    # save results
    with open(f'{args.savepath_data}{args.participant}_{args.session}_channel_dropping_allarrays_acc_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump({
            'all': acc,
        }, f)

    # save all results
    with open(f'{args.savepath_data}{args.participant}_{args.session}_channel_dropping_acc_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump(channel_dropped_accuracies, f)

    # plot results (accuracies)
    plot_channel_dropping_curve(channel_dropped_accuracies)

    # save scipt args
    print('Saving scripts args and script ...')
    script_name = sys.argv[0]
    with open(script_name, 'r') as f:
        script_content = f.read()

    with open(f'{args.savepath_data}{args.participant}_{args.session}_channel_dropping_performance_{formatted_datetime}.log', 'w') as f:
        f.write(str(args))
        f.write('\n\n----------\n\n')
        f.write(script_content)
    
    print('DONE!')