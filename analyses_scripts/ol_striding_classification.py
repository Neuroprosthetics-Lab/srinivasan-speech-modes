# This script trains a decoder taking in neural data from past X ms with a stride of Y ms.
# The results provide the trial-averaged decoding accuracy across time (each time point has a separate decoder).

import argparse
import os
import numpy as np
import scipy
from pathlib import Path
import math
from functions import get_audio_onset_offset
import pickle as pkl
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sys
from mne.stats import permutation_cluster_test

'''
Example cmd (when run from this directory; provide python script path appropriately if run from different directory):
For t15,
    python ol_striding_classification.py --participant t15 --session word-loudness --nbins_before_onset 150 --nbins_after_onset 150 --bins_before_trial_start 100 --bins_after_trial_start 100 --bins_before_trial_end 50 --bins_after_trial_end 200 --stream_window_len 40 --stream_window_stride 1 --savepath_data ../analyses_figures_data/t15/word-loudness/striding/ --savepath_fig ../analyses_figures/t15/word-loudness/striding/
For t16,
    python ol_striding_classification.py --participant t16 --session word-loudness --nbins_before_onset 150 --nbins_after_onset 150 --bins_before_trial_start 100 --bins_after_trial_start 100 --bins_before_trial_end 50 --bins_after_trial_end 200 --stream_window_len 40 --stream_window_stride 1 --savepath_data ../analyses_figures_data/t16/word-loudness/striding/ --savepath_fig ../analyses_figures/t16/word-loudness/striding/

Neural data will be loaded from ../analyses_data/{participant}_{session}/.
Results obtained from this script will be saved in savepath_data.
Figures generated from this script will be saved in savepath_fig.
To perform word classification (instead of loudness classification), add "classify_word" to the argument parser. 
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
n_channels = 256
bin_size_ms = 10
fs = 30000
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going']

pre_delay_nbins_in_neural_feat = 100 # 1s of neural features before trial start included in data
post_go_nbins_in_neural_feat = -300 # 3s of neural features after trial end included in data

# plotting
fontsize = 16
scattersize = 70
linewidth = 4
my_color = 'green'
trial_end_flag = -1
speech_offset_flag = -1

word_color = (150/255, 54/255, 34/255)
loudness_color = (34/255, 54/255, 150/255)

my_loudness_color = [
    (85/255, 150/255, 110/255), #(0/255, 170/255, 95/255),
    'green',
]
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


def get_training_data(data):

    # get data around speech onset and around cue onset
    x_spikepow_speech = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels)) # n_trials x time bins x channels
    x_spikepow_cue = np.empty((0, args.bins_before_trial_start + args.bins_after_trial_start, n_channels)) # n_trials x time bins x channels
    x_spikepow_trial_end = np.empty((0, args.bins_before_trial_end + args.bins_after_trial_end, n_channels)) # n_trials x time bins x channels
    x_threshcross_speech = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels)) # n_trials x time bins x channels
    x_threshcross_cue = np.empty((0, args.bins_before_trial_start + args.bins_after_trial_start, n_channels)) # n_trials x time bins x channels
    x_threshcross_trial_end = np.empty((0, args.bins_before_trial_end + args.bins_after_trial_end, n_channels)) # n_trials x time bins x channels
    y_word_speech = np.empty((0, 1)) # n_trials x word label 
    y_amp_speech = np.empty((0, 1)) # n_trials x amplitude label
    
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

        start_ind = np.squeeze(data['speech_onsets'])[ind].squeeze()
        end_ind = np.squeeze(data['speech_offsets'])[ind].squeeze()
        start_ind = math.floor((start_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index
        end_ind = math.ceil((end_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index

        spikepow_around_speech_onset = spikepow[pre_delay_nbins_in_neural_feat + binned_delay_duration + (start_ind - args.nbins_before_onset): pre_delay_nbins_in_neural_feat + binned_delay_duration + (start_ind + args.nbins_after_onset), :]
        threshcross_around_speech_onset = threshcross[pre_delay_nbins_in_neural_feat + binned_delay_duration + (start_ind - args.nbins_before_onset): pre_delay_nbins_in_neural_feat + binned_delay_duration + (start_ind + args.nbins_after_onset), :]
        spikepow_around_speech_offset = spikepow[pre_delay_nbins_in_neural_feat + binned_delay_duration + end_ind - args.bins_before_trial_end: pre_delay_nbins_in_neural_feat + binned_delay_duration + end_ind + args.bins_after_trial_end, :]
        threshcross_around_speech_offset = threshcross[pre_delay_nbins_in_neural_feat + binned_delay_duration + end_ind - args.bins_before_trial_end: pre_delay_nbins_in_neural_feat + binned_delay_duration + end_ind + args.bins_after_trial_end, :]

        if spikepow_around_speech_onset.shape[0] == args.nbins_before_onset + args.nbins_after_onset and spikepow_around_speech_offset.shape[0] == args.bins_before_trial_end + args.bins_after_trial_end:
            
            # go period spikepow
            temp_spikepow = spikepow_around_speech_onset
            x_spikepow_speech = np.append(x_spikepow_speech, np.expand_dims(temp_spikepow, 0), axis = 0)
            temp_threshcross = threshcross_around_speech_onset
            x_threshcross_speech = np.append(x_threshcross_speech, np.expand_dims(temp_threshcross, 0), axis = 0)
        
            # delay period spikepow, with some threshold crossings before cue onset
            temp_spikepow = spikepow[(pre_delay_nbins_in_neural_feat - args.bins_before_trial_start): (pre_delay_nbins_in_neural_feat + args.bins_after_trial_start), :] 
            x_spikepow_cue = np.append(x_spikepow_cue, np.expand_dims(temp_spikepow, axis = 0), axis = 0)
            temp_threshcross = threshcross[(pre_delay_nbins_in_neural_feat - args.bins_before_trial_start): (pre_delay_nbins_in_neural_feat + args.bins_after_trial_start), :] 
            x_threshcross_cue = np.append(x_threshcross_cue, np.expand_dims(temp_threshcross, axis = 0), axis = 0)

            # add neural features around speech offset
            temp_spikepow = spikepow_around_speech_offset
            x_spikepow_trial_end = np.append(x_spikepow_trial_end, np.expand_dims(temp_spikepow, axis = 0), axis = 0)
            temp_threshcross = threshcross_around_speech_offset
            x_threshcross_trial_end = np.append(x_threshcross_trial_end, np.expand_dims(temp_threshcross, axis = 0), axis = 0)

            # add labels
            y_word_speech = np.append(y_word_speech, np.expand_dims(word_label, 0), axis = 0)
            y_amp_speech = np.append(y_amp_speech, np.expand_dims(amp_label, 0), axis = 0)

    print('Spikepow, threshcross (speech, cue, trial_end) and label shapes for model:', x_spikepow_speech.shape, x_spikepow_cue.shape, x_spikepow_trial_end.shape,
          x_threshcross_speech.shape, x_threshcross_cue.shape, x_threshcross_trial_end.shape, y_word_speech.shape, y_amp_speech.shape)
    
    return x_spikepow_speech, x_spikepow_cue, x_spikepow_trial_end, x_threshcross_speech, x_threshcross_cue, x_threshcross_trial_end, y_word_speech, y_amp_speech


def train_logistic_regression(x_spikepow, y_word, y_amp):

    pred_stream_len = (int((x_spikepow.shape[1] - args.stream_window_len) / args.stream_window_stride)) + 1
    results_matrix = np.empty((0, pred_stream_len)) # (n_folds * n_repeats_per_fold) x pred_stream_len; (first n_repeat entries belong to the first fold)
    results_matrix_chance = np.empty((0, pred_stream_len))
    
    # cross validation (each fold has data from a word, test on unseen word)
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
        x_test = x_spikepow[test_inds, :, :]
        y_test = y_amp[test_inds, :]

        # train data
        x_train = x_spikepow[train_inds, :, :]
        y_train = y_amp[train_inds, :].squeeze()

        test_label = np.repeat(y_test, pred_stream_len, axis = 1) # n_trials x pred_stream_len # same gt test label for all repetitions
        for repeat in range(args.n_repeats_per_fold):
 
            pred_label = np.empty((x_test.shape[0], 0)) # n_trials x pred_stream_len
            pred_label_chance = np.empty((x_test.shape[0], 0))
            for win in range(args.stream_window_len, x_train.shape[1] + 1, args.stream_window_stride):
                x_train_win = x_train[:, win - args.stream_window_len: win, :]
                x_test_win = x_test[:, win - args.stream_window_len: win, :]

                # for this window, set a random seed, train and evaluate performance
                if repeat == 0:
                    random_seed = np.random.randint(0,10000)
                    clf = LogisticRegression(max_iter = 1000, random_state = random_seed) # same performance across repeats as it is the same data!
                    clf.fit(np.reshape(x_train_win, (x_train_win.shape[0],-1)), y_train)

                    y_pred = clf.predict(np.reshape(x_test_win, (x_test_win.shape[0], -1)))
                    pred_label = np.append(pred_label, np.expand_dims(y_pred, -1), axis = -1)
                    
                # for this window, with the same random seed, compute chance
                clf = LogisticRegression(max_iter = 1000, random_state = random_seed)
                clf.fit(np.reshape(x_train_win, (x_train_win.shape[0],-1)), np.random.permutation(y_train))

                y_pred = clf.predict(np.reshape(x_test_win, (x_test_win.shape[0], -1)))
                pred_label_chance = np.append(pred_label_chance, np.expand_dims(y_pred, -1), axis = -1)
        
            
            # fold accuracy for this repeat cycle
            if repeat == 0:
                acc_matrix_binary = test_label == pred_label # n_trials x pred_stream_len; binary values for correct and incorrect predictions
                acc_stream = np.sum(acc_matrix_binary, axis = 0) / acc_matrix_binary.shape[0] # 1x pred_stream_len
                print(f'Fold {fold}, Repetition {repeat}, Word {words[fold]} decoding accuracy computed')
                results_matrix = np.append(results_matrix, np.expand_dims(acc_stream, 0), axis = 0)

            # chance accuracy for this repeat cycle
            acc_matrix_binary = test_label == pred_label_chance # n_trials x pred_stream_len; binary values for correct and incorrect predictions
            acc_stream = np.sum(acc_matrix_binary, axis = 0) / acc_matrix_binary.shape[0] # 1x pred_stream_len
            print(f'Fold {fold}, Repetition {repeat}, Word {words[fold]} chance computed')
            results_matrix_chance = np.append(results_matrix_chance, np.expand_dims(acc_stream, 0), axis = 0)

    
    # print('Accuracy across folds and repetitions, shape:', results_matrix.shape)
    # print('Mean Accuracy across folds and repetitions:', np.mean(results_matrix, axis = 0))
    # print('Chance Accuracy across folds and repetitions:', results_matrix_chance.shape)
    # print('Mean Chance Accuracy across folds and repetitions:', np.mean(results_matrix_chance, axis = 0))

    return results_matrix, results_matrix_chance


def train_logistic_regression_word(x_spikepow, y_word, y_amp):
    pred_stream_len = (int((x_spikepow.shape[1] - args.stream_window_len) / args.stream_window_stride)) + 1
    results_matrix = np.empty((0, pred_stream_len)) # (n_folds * n_repeats_per_fold) x pred_stream_len; (first n_repeat entries belong to the first fold)
    results_matrix_chance = np.empty((0, pred_stream_len))
    
    # cross validation (each fold has data from a loudness, test on unseen loudness)
    for fold in range(len(amplitudes)):
        print('Fold ', fold, amplitudes[fold])
        
        # inds in this fold
        test_inds = np.argwhere(y_amp.squeeze() == fold).squeeze()
        train_inds = np.argwhere(y_amp.squeeze() != fold).squeeze()

        # check non-overlap between train and test inds
        for ind in test_inds:
            assert ind not in train_inds
        for ind in train_inds:
            assert ind not in test_inds

        # test data
        x_test = x_spikepow[test_inds, :, :]
        y_test = y_word[test_inds, :]

        # train data
        x_train = x_spikepow[train_inds, :, :]
        y_train = y_word[train_inds, :].squeeze()

        test_label = np.repeat(y_test, pred_stream_len, axis = 1) # n_trials x pred_stream_len # same gt test label for all repetitions
        for repeat in range(args.n_repeats_per_fold):
 
            pred_label = np.empty((x_test.shape[0], 0)) # n_trials x pred_stream_len
            pred_label_chance = np.empty((x_test.shape[0], 0))
            for win in range(args.stream_window_len, x_train.shape[1] + 1, args.stream_window_stride):
                x_train_win = x_train[:, win - args.stream_window_len: win, :]
                x_test_win = x_test[:, win - args.stream_window_len: win, :]

                # for this window, set a random seed, train and evaluate performance
                if repeat == 0:
                    random_seed = np.random.randint(0,10000)
                    clf = LogisticRegression(max_iter = 1000, random_state = random_seed) # same performance across repeats as it is the same data!
                    clf.fit(np.reshape(x_train_win, (x_train_win.shape[0],-1)), y_train)

                    y_pred = clf.predict(np.reshape(x_test_win, (x_test_win.shape[0], -1)))
                    pred_label = np.append(pred_label, np.expand_dims(y_pred, -1), axis = -1)
                    
                # for this window, with the same random seed, compute chance
                clf = LogisticRegression(max_iter = 1000, random_state = random_seed)
                clf.fit(np.reshape(x_train_win, (x_train_win.shape[0],-1)), np.random.permutation(y_train))

                y_pred = clf.predict(np.reshape(x_test_win, (x_test_win.shape[0], -1)))
                pred_label_chance = np.append(pred_label_chance, np.expand_dims(y_pred, -1), axis = -1)
        
            # fold accuracy for this repeat cycle
            if repeat == 0:
                acc_matrix_binary = test_label == pred_label # n_trials x pred_stream_len; binary values for correct and incorrect predictions
                acc_stream = np.sum(acc_matrix_binary, axis = 0) / acc_matrix_binary.shape[0] # 1x pred_stream_len
                print(f'Fold {fold}, Repetition {repeat}, Loudness {amplitudes[fold]} decoding accuracy computed')
                results_matrix = np.append(results_matrix, np.expand_dims(acc_stream, 0), axis = 0)
                
            # chance accuracy for this repeat cycle
            acc_matrix_binary = test_label == pred_label_chance # n_trials x pred_stream_len; binary values for correct and incorrect predictions
            acc_stream = np.sum(acc_matrix_binary, axis = 0) / acc_matrix_binary.shape[0] # 1x pred_stream_len
            print(f'Fold {fold}, Repetition {repeat}, Loudness {amplitudes[fold]} chance computed')
            results_matrix_chance = np.append(results_matrix_chance, np.expand_dims(acc_stream, 0), axis = 0)

    # print('Accuracy across folds and repetitions, shape:', results_matrix.shape)
    # print('Mean Accuracy across folds and repetitions:', np.mean(results_matrix, axis = 0))
    # print('Chance Accuracy across folds and repetitions:', results_matrix_chance.shape)
    # print('Mean Chance Accuracy across folds and repetitions:', np.mean(results_matrix_chance, axis = 0))

    return results_matrix, results_matrix_chance



def plot_striding_performance(fold_acc_matrix_speech, fold_acc_matrix_cue, fold_acc_matrix_trial_end,
                              fold_acc_matrix_speech_chance, fold_acc_matrix_cue_chance, fold_acc_matrix_trial_end_chance):
    
    mean_acc_speech = np.mean(fold_acc_matrix_speech, axis = 0)
    sem_acc_speech = np.std(fold_acc_matrix_speech, axis = 0) / np.sqrt(fold_acc_matrix_speech.shape[0])

    mean_acc_cue = np.mean(fold_acc_matrix_cue, axis = 0)
    sem_acc_cue = np.std(fold_acc_matrix_cue, axis = 0) / np.sqrt(fold_acc_matrix_cue.shape[0])

    mean_acc_trial_end = np.mean(fold_acc_matrix_trial_end, axis = 0)
    sem_acc_trial_end = np.std(fold_acc_matrix_trial_end, axis = 0) / np.sqrt(fold_acc_matrix_trial_end.shape[0])

    mean_acc_speech_chance = np.mean(fold_acc_matrix_speech_chance, axis = 0)
    sem_acc_speech_chance = np.std(fold_acc_matrix_speech_chance, axis = 0) / np.sqrt(fold_acc_matrix_speech_chance.shape[0])

    mean_acc_cue_chance = np.mean(fold_acc_matrix_cue_chance, axis = 0)
    sem_acc_cue_chance = np.std(fold_acc_matrix_cue_chance, axis = 0) / np.sqrt(fold_acc_matrix_cue_chance.shape[0])

    mean_acc_trial_end_chance = np.mean(fold_acc_matrix_trial_end_chance, axis = 0)
    sem_acc_trial_end_chance = np.std(fold_acc_matrix_trial_end_chance, axis = 0) / np.sqrt(fold_acc_matrix_trial_end_chance.shape[0])

    print('Mean accuracies shapes (speech, cue, trial end):', mean_acc_speech.shape, mean_acc_cue.shape, mean_acc_trial_end.shape)
    print('Mean chance accuracies shapes (speech, cue, trial end):', mean_acc_speech_chance.shape, mean_acc_cue_chance.shape, mean_acc_trial_end_chance.shape)
    print('SEM accuracies shapes (speech, cue, trial end):', sem_acc_speech.shape, sem_acc_cue.shape, sem_acc_trial_end.shape)
    print('SEM chance accuracies shapes (speech, cue, trial end):', sem_acc_speech_chance.shape, sem_acc_cue_chance.shape, sem_acc_trial_end_chance.shape)

    # max value during speech
    max_value = np.max(mean_acc_speech)
    max_value_ind = np.argmax(mean_acc_speech)

    cue_onset_ind = (args.bins_before_trial_start - args.stream_window_len) / args.stream_window_stride 
    speech_onset_ind = (args.nbins_before_onset - args.stream_window_len) / args.stream_window_stride
    trial_end_onset_ind = (args.bins_before_trial_end - args.stream_window_len) / args.stream_window_stride
    print('Cue, speech and trial end onset in plot:', cue_onset_ind, speech_onset_ind, trial_end_onset_ind)

    # MNE time cluster permutation test
    all_acc = np.concatenate([mean_acc_cue, mean_acc_speech, mean_acc_trial_end], axis = 0)
    all_acc = np.expand_dims(all_acc, 0) # 1 x n_timepoints
    all_chance_repeats = np.concatenate([fold_acc_matrix_cue_chance, fold_acc_matrix_speech_chance, fold_acc_matrix_trial_end_chance], axis = 1) # n_repeats x n_timepoints

    # F_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
    #     X = [all_acc, all_chance_repeats],
    #     threshold = None,
    #     n_permutations = 1000,
    #     tail = 1,
    #     out_type = 'indices',
    #     adjacency = None,
    # )

    # cue_sig_cluster_start = []
    # for i, cluster in enumerate(clusters):
    #     start, end = cluster[0][0], cluster[0][-1]
    #     print(f'Start and end time bin of cluster {i}:', start, end)
    #     print(f'P-value of cluster {i}:', cluster_pv[i])
    #     if cluster_pv[i] < 0.05:
    #         cue_sig_cluster_start.append(start)

    # print('Significant cluster starts at time bins:', cue_sig_cluster_start)
    # print('First time point of significant decoding (ms):', f'{int((cue_sig_cluster_start[0] - cue_onset_ind)*args.stream_window_stride*bin_size_ms)} ms')
    # print('Time point of max accuracy during speech (ms):', f'{int((max_value_ind - speech_onset_ind)*args.stream_window_stride*bin_size_ms)} ms')

    fig, ax = plt.subplots(1,3,figsize=(15,4), gridspec_kw={'width_ratios': [len(mean_acc_cue), len(mean_acc_speech), len(mean_acc_trial_end)]})
    if args.participant == 't15':
        fontsize = 16
    elif args.participant == 't16':
        fontsize = 15

    # cue period
    decoder_label = 'Loudness decoder'
    if args.classify_word:
        decoder_label = 'Word decoder'
    ax[0].plot(mean_acc_cue, label = decoder_label, color = my_color, linewidth = linewidth)
    ax[0].fill_between(np.arange(len(mean_acc_cue)),
                                mean_acc_cue - sem_acc_cue,
                                mean_acc_cue + sem_acc_cue,
                                alpha = 0.5, label = '_hidden', color = my_color)

    ax[0].plot(mean_acc_cue_chance, label = 'Chance', color = 'black', linestyle = '--', linewidth = linewidth)
    ax[0].fill_between(np.arange(len(mean_acc_cue_chance)),
                                mean_acc_cue_chance - sem_acc_cue_chance,
                                mean_acc_cue_chance + sem_acc_cue_chance,
                                alpha = 0.5, label = 'hidden', color = 'black')

    ax[0].set_ylim([0,1])
    ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100], fontsize = fontsize)
    ax[0].set_ylabel('Accuracy (%)', fontsize = fontsize)
    ax[0].set_xticks([])

    for pos in ['right', 'top', 'bottom']: 
        ax[0].spines[pos].set_visible(False)
    if args.participant == 't16':
        ax[0].set_xticks(np.arange(0, len(mean_acc_cue), 50)+10, (np.arange(-(args.bins_before_trial_start-args.stream_window_len),args.bins_after_trial_start, 50)+10) * 10, fontsize = fontsize)
        ax[0].spines['bottom'].set_visible(True)

    ax[0].scatter(cue_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[0].text(cue_onset_ind, -0.08, "Cue", fontsize = fontsize, ha = "center")

    # # time cluster permutation
    # ax[0].scatter(cue_sig_cluster_start[0], 0.02, s = scattersize, color = my_color, marker = '*')
    # if args.participant == 't15':
    #     ax[0].text(cue_sig_cluster_start[0], -0.08, f'{int((cue_sig_cluster_start[0] - cue_onset_ind)*args.stream_window_stride*bin_size_ms)} ms', fontsize = fontsize, ha = "center")


    # go period
    ax[1].plot(mean_acc_speech, label = decoder_label, color = my_color, linewidth = linewidth)
    ax[1].fill_between(np.arange(len(mean_acc_speech)),
                                mean_acc_speech - sem_acc_speech,
                                mean_acc_speech + sem_acc_speech,
                                alpha = 0.5, label = '_hidden', color = my_color)

    ax[1].plot(mean_acc_speech_chance, label = 'Chance', color = 'black', linestyle = '--', linewidth = linewidth)
    ax[1].fill_between(np.arange(len(mean_acc_speech_chance)),
                                mean_acc_speech_chance - sem_acc_speech_chance,
                                mean_acc_speech_chance + sem_acc_speech_chance,
                                alpha = 0.5, label = 'hidden', color = 'black')
    
    # add vertical line from where we can significantly decode loudness
    ax[1].vlines(max_value_ind, 0, max_value, color='black', linewidth=3, alpha = 0.4)
    ax[1].text(max_value_ind, max_value + 0.05, f'{max_value * 100:.1f}%', fontsize = fontsize, ha = 'center')
    if args.participant == 't16':
        ax[1].scatter(max_value_ind, 0.02, s = scattersize, color = 'black')
    
    ax[1].set_ylim([0,1])
    for pos in ['right', 'top', 'left', 'bottom']: 
        ax[1].spines[pos].set_visible(False) 
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    if args.participant == 't16':
        ax[1].set_xticks(np.arange(0, len(mean_acc_speech), 50) + 10, (np.arange(-(args.nbins_before_onset-args.stream_window_len), args.nbins_after_onset, 50) + 10) * 10, fontsize = fontsize)
        ax[1].spines['bottom'].set_visible(True)
        ax[1].set_xlabel('Time (ms)', fontsize = fontsize)

    ax[1].scatter(speech_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[1].text(speech_onset_ind, -0.17, "Speech\nonset", fontsize = fontsize, ha = "center")
    
    
    # trial end period
    ax[2].plot(mean_acc_trial_end, label = decoder_label, color = my_color, linewidth = linewidth)
    ax[2].fill_between(np.arange(len(mean_acc_trial_end)),
                                mean_acc_trial_end - sem_acc_trial_end,
                                mean_acc_trial_end + sem_acc_trial_end,
                                alpha = 0.5, label = '_hidden', color = my_color)

    ax[2].plot(mean_acc_trial_end_chance, label = 'Chance', color = 'black', linestyle = '--', linewidth = linewidth)
    ax[2].fill_between(np.arange(len(mean_acc_trial_end_chance)),
                                mean_acc_trial_end_chance - sem_acc_trial_end_chance,
                                mean_acc_trial_end_chance + sem_acc_trial_end_chance,
                                alpha = 0.5, label = '_hidden', color = 'black')
    
    ax[2].set_ylim([0,1])
    for pos in ['right', 'top', 'left', 'bottom']: 
        ax[2].spines[pos].set_visible(False) 
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    if args.participant == 't16':
        ax[2].set_xticks(np.arange(0, len(mean_acc_trial_end), 50)[1:], np.arange(0, len(mean_acc_trial_end), 50)[1:] * 10, fontsize = fontsize)
        ax[2].spines['bottom'].set_visible(True)

    ax[2].scatter(trial_end_onset_ind, 0.02, s = scattersize, color = 'black')

        
    trial_end_label = "Speech\noffset"
    ax[2].text(trial_end_onset_ind, -0.16, trial_end_label, fontsize = fontsize, ha = "center")
    if args.participant == 't15':
        ax[2].hlines(0.01, xmin = len(mean_acc_trial_end) - (50 / args.stream_window_stride), xmax = len(mean_acc_trial_end), linewidth = 3, color = 'black')
        ax[2].text(len(mean_acc_trial_end) - (25 / args.stream_window_stride), -0.08, "500 ms", ha = "center", fontsize = fontsize)
    
    plt.subplots_adjust(wspace=0.02) 
    fig.tight_layout()
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1), fontsize = fontsize)
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_striding_performance_{formatted_datetime}.png', format='png')

    return


if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms', 'spikepow', 'threshcross', 'speech_onsets', 'speech_offsets'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=150, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=150, help = 'number of bins after speech onset')
    parser.add_argument('--stream_window_len', type=int, default=40, help = 'stream window length')
    parser.add_argument('--stream_window_stride', type=int, default=10, help = 'stream window stride')
    parser.add_argument('--bins_before_trial_start', type=int, default=100, help = 'number of bins to consider before start of the trial')
    parser.add_argument('--bins_after_trial_start', type=int, default = 100, help = 'number of bins to consider after the start of the trial')
    parser.add_argument('--bins_before_trial_end', type=int, default=100, help = 'number of bins to consider before end of trial')
    parser.add_argument('--bins_after_trial_end', type=int, default = 100, help = 'number of bins to consider after end of trial')
    parser.add_argument('--n_repeats_per_fold', type=int, default=5, help = 'number of times each fold is modeled with a different random seed, or chance is computed per fold')
    parser.add_argument('--classify_word', action='store_true', help = 'whether to classify word instead of loudness')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_data):
        os.makedirs(args.savepath_data, exist_ok=True)
    
    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)
    
    print('Running ol_striding_classification.py')
    print(args)

    # load data
    print('Loading data ...')
    data = load_rdbmat(args.participant, args.session, args.required_keys)

    # get training data
    print('Loading training data ...')
    x_spikepow_speech, x_spikepow_cue, x_spikepow_trial_end, x_threshcross_speech, x_threshcross_cue, x_threshcross_trial_end, y_word, y_amp = get_training_data(data)
    # save data
    with open(f'{args.savepath_data}{args.participant}_{args.session}_striding_processed_data_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump({
            'x_spikepow_speech': x_spikepow_speech,
            'x_spikepow_cue': x_spikepow_cue,
            'x_spikepow_trial_end': x_spikepow_trial_end,
            'x_threshcross_speech': x_threshcross_cue,
            'x_threshcross_cue': x_threshcross_cue,
            'x_threshcross_trial_end': x_threshcross_trial_end,
            'y_word': y_word,
            'y_amp': y_amp
        }, f)

    print('Number of repeats per fold (or number of chance computations per fold):', args.n_repeats_per_fold)

    if args.participant == 't15':
        x_neural_feat_cue = np.concatenate([x_spikepow_cue, x_threshcross_cue], axis = -1)
        x_neural_feat_speech = np.concatenate([x_spikepow_speech, x_threshcross_speech], axis = -1)
        x_neural_feat_trial_end = np.concatenate([x_spikepow_trial_end, x_threshcross_trial_end], axis = -1)
    elif args.participant == 't16': # consider only speecch arrays
        x_neural_feat_cue = np.concatenate([x_spikepow_cue[:, :, :128], x_threshcross_cue[:, :, :128]], axis = -1)
        x_neural_feat_speech = np.concatenate([x_spikepow_speech[:, :, :128], x_threshcross_speech[:, :, :128]], axis = -1)
        x_neural_feat_trial_end = np.concatenate([x_spikepow_trial_end[:, :, :128], x_threshcross_trial_end[:, :, :128]], axis = -1)
    
    # logistic regression classification - cue
    #--------------------------------------------------
    print('Training logistic regression on cue - cross-validation ...')
    if not args.classify_word:
        print('Classifying loudness labels ...')
        fold_acc_matrix_cue, fold_acc_matrix_cue_chance = train_logistic_regression(x_neural_feat_cue, y_word, y_amp) # n_folds x pred_stream_len
    else:
        print('Classifying word labels ...')
        fold_acc_matrix_cue, fold_acc_matrix_cue_chance = train_logistic_regression_word(x_neural_feat_cue, y_word, y_amp) # n_folds x pred_stream_len
        
    # save results
    with open(f'{args.savepath_data}{args.participant}_{args.session}_striding_fold_acc_{formatted_datetime}_cue.pkl', 'wb') as f:
        pkl.dump({
            'fold_acc_matrix_cue': fold_acc_matrix_cue,
            'fold_acc_matrix_cue_chance': fold_acc_matrix_cue_chance,
        }, f)

    # logistic regression classification - speech
    #--------------------------------------------------
    print('Training logistic regression on speech - cross-validation ...')
    if not args.classify_word:
        print('Classifying loudness labels ...')
        fold_acc_matrix_speech, fold_acc_matrix_speech_chance = train_logistic_regression(x_neural_feat_speech, y_word, y_amp) # n_folds x pred_stream_len
    else:
        print('Classifying word labels ...')
        fold_acc_matrix_speech, fold_acc_matrix_speech_chance = train_logistic_regression_word(x_neural_feat_speech, y_word, y_amp) # n_folds x pred_stream_len
        
    # save results
    with open(f'{args.savepath_data}{args.participant}_{args.session}_striding_fold_acc_{formatted_datetime}_speech.pkl', 'wb') as f:
        pkl.dump({
            'fold_acc_matrix_speech': fold_acc_matrix_speech,
            'fold_acc_matrix_speech_chance': fold_acc_matrix_speech_chance,
        }, f)

    # logistic regression classification - end
    #--------------------------------------------------
    print('Training logistic regression on trial end - cross-validation ...')
    if not args.classify_word:
        print('Classifying loudness labels ...')
        fold_acc_matrix_trial_end, fold_acc_matrix_trial_end_chance = train_logistic_regression(x_neural_feat_trial_end, y_word, y_amp) # n_folds x pred_stream_len 
    else:
        print('Classifying word labels ...')
        fold_acc_matrix_trial_end, fold_acc_matrix_trial_end_chance = train_logistic_regression_word(x_neural_feat_trial_end, y_word, y_amp) # n_folds x pred_stream_len

    # save results
    with open(f'{args.savepath_data}{args.participant}_{args.session}_striding_fold_acc_{formatted_datetime}_trialend.pkl', 'wb') as f:
        pkl.dump({
            'fold_acc_matrix_trial_end': fold_acc_matrix_trial_end,
            'fold_acc_matrix_trial_end_chance': fold_acc_matrix_trial_end_chance
        }, f)

    # # save all results
    with open(f'{args.savepath_data}{args.participant}_{args.session}_striding_fold_acc_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump({
            'fold_acc_matrix_cue': fold_acc_matrix_cue,
            'fold_acc_matrix_speech': fold_acc_matrix_speech,
            'fold_acc_matrix_trial_end': fold_acc_matrix_trial_end,
            'fold_acc_matrix_cue_chance': fold_acc_matrix_cue_chance,
            'fold_acc_matrix_speech_chance': fold_acc_matrix_speech_chance,
            'fold_acc_matrix_trial_end_chance': fold_acc_matrix_trial_end_chance
        }, f)

    # plot results
    plot_striding_performance(fold_acc_matrix_speech, fold_acc_matrix_cue, fold_acc_matrix_trial_end,
                              fold_acc_matrix_speech_chance, fold_acc_matrix_cue_chance, fold_acc_matrix_trial_end_chance)

    # save scipt args
    print('Saving scripts args and script ...')
    script_name = sys.argv[0]
    with open(script_name, 'r') as f:
        script_content = f.read()

    with open(f'{args.savepath_data}{args.participant}_{args.session}_striding_performance_{formatted_datetime}.log', 'w') as f:
        f.write(str(args))
        f.write('\n\n----------\n\n')
        f.write(script_content)

    print('DONE!')