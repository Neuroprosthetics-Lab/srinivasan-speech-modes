import argparse
import os
import numpy as np
import scipy
from pathlib import Path
from session_metadata import incorrect_trials
import math
from functions import get_audio_onset_offset
import pickle as pkl
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import sys

'''
example run cmd: 
python ol_striding_performance.py --participant <> --session <> --nbins_before_onset 150 --nbins_after_onset 150 --stream_window_len 40 --stream_window_stride 1 --savepath_fig <> 
    --savepath_data <> --n_repeats_per_fold 100 --bins_before_trial_end 50 --bins_after_trial_end 200
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
n_channels = 256
bin_size_ms = 10
fs = 30000
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going']

delay_onset_raw_threshcross_from_pre_cue = 100 # rdbmat has 1s of neural features before trial start included "raw_threshcross_from_pre_cue"
trial_end_onset_raw_threshcross_from_post_end = -300 # rdbmat has 3s of neural features after trial end included "spikepow_from_post_end"

# plotting
fontsize = 16
scattersize = 70
linewidth = 4
my_color = 'green'
trial_end_flag = -1
speech_offset_flag = -1

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
    if args.participant == 't15':
        valid_trial_inds = [i for i in range(n_trials) if 'DO NOTHING' not in data['cue'][i] and max(np.squeeze(data['predaudio16k'])[i]) != 0]
    elif args.participant == 't16':
        valid_trial_inds = [i for i in range(n_trials) if 'DO NOTHING' not in data['cue'][i]]
    print('Total number of usable trials:', len(valid_trial_inds))

    for ind in valid_trial_inds:
        # current data
        cue = data['cue'][ind].strip()
        amp_label = [amplitudes.index(cue.split(':')[0])]
        word_label = [words.index(cue.split(':')[-1].strip())]
        spikepow = np.squeeze(data['spikepow_from_delay'])[ind]
        prev_spikepow = np.squeeze(data['spikepow_from_pre_cue'])[ind]
        next_spikepow = np.squeeze(data['spikepow_with_post_end'])[ind]
        threshcross = np.squeeze(data['threshcross_from_delay'])[ind]
        prev_threshcross = np.squeeze(data['threshcross_from_pre_cue'])[ind]
        next_threshcross = np.squeeze(data['threshcross_with_post_end'])[ind]
        delay_duration_ms = np.squeeze(data['delay_duration_ms'])[ind]
        binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)

        if ind == valid_trial_inds[0]:
            print('current_dataset_spikepow_shape (from delay, from 1s prev, to 3s next)', spikepow.shape, prev_spikepow.shape, next_threshcross.shape)

        # append the duration after trial end to spikepow
        # print('spikepow shape from delay to trial end:', spikepow.shape)
        spikepow = np.concatenate([spikepow, next_spikepow[trial_end_onset_raw_threshcross_from_post_end:,:]], 0)
        threshcross = np.concatenate([threshcross, next_threshcross[trial_end_onset_raw_threshcross_from_post_end:, :]], 0)
        # print('spikepow shape from delay to 3s after trial end:', spikepow.shape)
            
        if args.participant == 't15':
            predaudio16k = np.squeeze(data['predaudio16k'])[ind]
            # speech onset
            start_ind, end_ind = get_audio_onset_offset(predaudio16k, display_audio = False, 
                                                                mic_audio = None, cue = None, 
                                                                intersegment_duration = 3000, 
                                                                amplitude_percentage = 0.1) # returns ind at 30k
        elif args.participant == 't16':
            start_ind = np.squeeze(data['speech_onsets'])[ind]
            end_ind = np.squeeze(data['speech_offsets'])[ind]

        start_ind = math.floor((start_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index
        end_ind = math.ceil((end_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index

        if np.expand_dims(spikepow[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :], 0).shape[1] == args.nbins_before_onset + args.nbins_after_onset:
            if np.expand_dims(next_spikepow[(end_ind - args.bins_before_trial_end): (end_ind + args.bins_after_trial_end), :], 0).shape[1] == args.bins_before_trial_end + args.bins_after_trial_end:
            
                # go period spikepow
                temp_spikepow = spikepow[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :] # shape (time_bins x 256)
                x_spikepow_speech = np.append(x_spikepow_speech, np.expand_dims(temp_spikepow, 0), axis = 0)
                temp_threshcross = threshcross[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :] # shape (time_bins x 256)
                x_threshcross_speech = np.append(x_threshcross_speech, np.expand_dims(temp_threshcross, 0), axis = 0)
            
                # delay period spikepow, with some threshold crossings before cue onset
                temp_spikepow = prev_spikepow[(delay_onset_raw_threshcross_from_pre_cue - args.bins_before_trial_start): (delay_onset_raw_threshcross_from_pre_cue + args.bins_after_trial_start), :] 
                x_spikepow_cue = np.append(x_spikepow_cue, np.expand_dims(temp_spikepow, axis = 0), axis = 0)
                temp_threshcross = prev_threshcross[(delay_onset_raw_threshcross_from_pre_cue - args.bins_before_trial_start): (delay_onset_raw_threshcross_from_pre_cue + args.bins_after_trial_start), :] 
                x_threshcross_cue = np.append(x_threshcross_cue, np.expand_dims(temp_threshcross, axis = 0), axis = 0)
                
                # add neural features around speech offset
                speech_offset_flag = 1
                temp_spikepow = next_spikepow[end_ind - args.bins_before_trial_end: end_ind + args.bins_after_trial_end, :]
                temp_threshcross = next_threshcross[end_ind - args.bins_before_trial_end: end_ind + args.bins_after_trial_end, :]
                x_spikepow_trial_end = np.append(x_spikepow_trial_end, np.expand_dims(temp_spikepow, axis = 0), axis = 0)
                x_threshcross_trial_end = np.append(x_threshcross_trial_end, np.expand_dims(temp_threshcross, axis = 0), axis = 0)

                # add labels
                y_word_speech = np.append(y_word_speech, np.expand_dims(word_label, 0), axis = 0)
                y_amp_speech = np.append(y_amp_speech, np.expand_dims(amp_label, 0), axis = 0)

    print('Spikepow, threshcross (speech, cue, trial_end) and label shapes for model:', x_spikepow_speech.shape, x_spikepow_cue.shape, x_spikepow_trial_end.shape,
          x_threshcross_speech.shape, x_threshcross_cue.shape, x_threshcross_trial_end.shape, y_word_speech.shape, y_amp_speech.shape)
    
    return x_spikepow_speech, x_spikepow_cue, x_spikepow_trial_end, x_threshcross_speech, x_threshcross_cue, x_threshcross_trial_end, y_word_speech, y_amp_speech



def train_logistic_regression(x_spikepow, y_word, y_amp):

    pred_stream_len = (int((x_spikepow.shape[1] - args.stream_window_len) / args.stream_window_stride)) + 1
    print('Predicted stream length:', pred_stream_len)
    results_matrix = np.empty((0, pred_stream_len)) # (n_folds * n_repeats_per_fold) x pred_stream_len; (first n_repeat entries belong to the first fold)
    # "results_matrix" will have only 1 repetition as performance does not change with repetitions!
    results_matrix_chance = np.empty((0, pred_stream_len))
    
    # cross validation (each fold has data from a word, test on unseen word)
    for fold in range(len(words)):
        print('Fold ', fold, words[fold])
        
        # inds in this fold
        test_inds = np.argwhere(y_word.squeeze() == fold).squeeze()
        train_inds = np.argwhere(y_word.squeeze() != fold).squeeze()
        print('Number of train and test trials:', len(train_inds), len(test_inds))

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

        print('Train and test data shapes:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        test_label = np.repeat(y_test, pred_stream_len, axis = 1) # n_trials x pred_stream_len # same gt test label for all repetitions
        for repeat in range(args.n_repeats_per_fold):
 
            pred_label = np.empty((x_test.shape[0], 0)) # n_trials x pred_stream_len
            pred_label_chance = np.empty((x_test.shape[0], 0))
            for win in range(args.stream_window_len, x_train.shape[1] + 1, args.stream_window_stride):
                print(f'Training window: ({win - args.stream_window_len}, {win})')
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
        
            print('Test and predicted label shapes (n_trials x pred_stream_len):', test_label.shape, pred_label.shape)

            # fold accuracy for this repeat cycle
            if repeat == 0:
                acc_matrix_binary = test_label == pred_label # n_trials x pred_stream_len; binary values for correct and incorrect predictions
                acc_stream = np.sum(acc_matrix_binary, axis = 0) / acc_matrix_binary.shape[0] # 1x pred_stream_len
                print(f'Fold {fold}, Repetition {repeat}, Word {words[fold]}, accuracy: {acc_stream}')
                results_matrix = np.append(results_matrix, np.expand_dims(acc_stream, 0), axis = 0)

            # chance accuracy for this repeat cycle
            acc_matrix_binary = test_label == pred_label_chance # n_trials x pred_stream_len; binary values for correct and incorrect predictions
            acc_stream = np.sum(acc_matrix_binary, axis = 0) / acc_matrix_binary.shape[0] # 1x pred_stream_len
            print(f'Fold {fold}, Repetition {repeat}, Word {words[fold]}, chance accuracy: {acc_stream}')
            results_matrix_chance = np.append(results_matrix_chance, np.expand_dims(acc_stream, 0), axis = 0)

    
    print('Accuracy across folds and repetitions, shape:', results_matrix.shape)
    print('Mean Accuracy across folds and repetitions:', np.mean(results_matrix, axis = 0))
    print('Chance Accuracy across folds and repetitions:', results_matrix_chance.shape)
    print('Mean Chance Accuracy across folds and repetitions:', np.mean(results_matrix_chance, axis = 0))

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

    # max value during speech
    print(mean_acc_speech)
    max_value = np.max(mean_acc_speech)
    max_value_ind = np.argmax(mean_acc_speech)
    print('Max value and ind', max_value, max_value_ind)

    # where decoded accuracy is greater than 80% (arbitrary threshold)
    arb_value_ind = np.argwhere(mean_acc_speech > 0.8).squeeze()[0]
    arb_value = mean_acc_speech[arb_value_ind]

    # min value during trial end
    min_value = np.min(mean_acc_trial_end)
    min_value_ind = np.argmin(mean_acc_trial_end)
    print('Min value and ind', min_value, min_value_ind)
    mean_acc_trial_end = mean_acc_trial_end[:min_value_ind + 1]
    mean_acc_trial_end_chance = mean_acc_trial_end_chance[:min_value_ind + 1]
    sem_acc_trial_end = sem_acc_trial_end[:min_value_ind + 1]
    sem_acc_trial_end_chance = sem_acc_trial_end_chance[:min_value_ind + 1]

    cue_onset_ind = (args.bins_before_trial_start - args.stream_window_len) / args.stream_window_stride 
    speech_onset_ind = (args.nbins_before_onset - args.stream_window_len) / args.stream_window_stride
    trial_end_onset_ind = (args.bins_before_trial_end - args.stream_window_len) / args.stream_window_stride
    print('Cue, speech and trial end onset in plot:', cue_onset_ind, speech_onset_ind, trial_end_onset_ind)
    
    fig, ax = plt.subplots(1,3,figsize=(15,4), gridspec_kw={'width_ratios': [len(mean_acc_cue), len(mean_acc_speech), len(mean_acc_trial_end)]})
    if args.participant == 't15':
        fontsize = 16
    elif args.participant == 't16':
        fontsize = 15

    # cue period
    ax[0].plot(mean_acc_cue, label = 'Loudness decoder', color = my_color, linewidth = linewidth)
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
    ax[0].scatter(cue_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[0].text(cue_onset_ind, -0.08, "Cue", fontsize = fontsize, ha = "center")


    # go period
    ax[1].plot(mean_acc_speech, label = 'Loudness decoder', color = my_color, linewidth = linewidth)
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
        ax[1].text(max_value_ind + 0.5, -0.05, f'{int((max_value_ind - speech_onset_ind)*args.stream_window_stride*bin_size_ms)} ms', fontsize = fontsize, ha = 'left')
    
    ax[1].scatter(arb_value_ind, 0.02, s = scattersize, color = 'black')
    if args.participant == 't15':
        ax[1].text(arb_value_ind, -0.09, f'{int((arb_value_ind - speech_onset_ind)*args.stream_window_stride*bin_size_ms)} ms', fontsize = fontsize, ha = 'center')
    elif args.participant == 't16':
        ax[1].text(arb_value_ind - 0.5, -0.04, f'{int((arb_value_ind - speech_onset_ind)*args.stream_window_stride*bin_size_ms)} ms', fontsize = fontsize, ha = 'right')
    
    ax[1].vlines(arb_value_ind, 0, arb_value, color = 'black', linewidth = 3, alpha = 0.4)
    ax[1].text(arb_value_ind, arb_value + 0.07, f'{arb_value * 100:.1f}%', fontsize = fontsize, ha = 'center')
    
    ax[1].set_ylim([0,1])
    for pos in ['right', 'top', 'left', 'bottom']: 
        ax[1].spines[pos].set_visible(False) 
    ax[1].set_yticks([])
    ax[1].set_xticks([])

    ax[1].scatter(speech_onset_ind, 0.02, s = scattersize, color = 'black')
    ax[1].text(speech_onset_ind, -0.17, "Speech\nonset", fontsize = fontsize, ha = "center")
    
    
    # trial end period
    ax[2].plot(mean_acc_trial_end, label = 'Loudness decoder', color = my_color, linewidth = linewidth)
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

    ax[2].scatter(trial_end_onset_ind, 0.02, s = scattersize, color = 'black')
    if speech_offset_flag:
        trial_end_label = "Speech offset"
    elif trial_end_flag:
        trial_end_label = "End of trial\nNext cue"

    ax[2].text(trial_end_onset_ind, -0.07, trial_end_label, fontsize = fontsize, ha = "center")
    ax[2].hlines(0.01, xmin = len(mean_acc_trial_end) - (50 / args.stream_window_stride), xmax = len(mean_acc_trial_end), linewidth = 3, color = 'black')
    ax[2].text(len(mean_acc_trial_end) - (25 / args.stream_window_stride), -0.08, "500 ms", ha = "center", fontsize = fontsize)
    
    plt.subplots_adjust(wspace=0.02) 
    fig.tight_layout()

    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1), fontsize = fontsize)
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_striding_performance_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_striding_performance_{formatted_datetime}.png', format='png')

    return



if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms', 'predaudio16k', 'spikepow_from_delay', 
                                                               'spikepow_from_pre_cue', 'threshcross_from_delay', 'threshcross_from_pre_cue',
                                                               'spikepow_with_post_end', 'threshcross_with_post_end'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--stream_window_len', type=int, default=None, help = 'stream window length')
    parser.add_argument('--stream_window_stride', type=int, default=None, help = 'stream window stride')
    parser.add_argument('--bins_before_trial_start', type=int, default=100, help = 'number of bins to consider before start of the trial')
    parser.add_argument('--bins_after_trial_start', type=int, default = 100, help = 'number of bins to consider after the start of the trial')
    parser.add_argument('--bins_before_trial_end', type=int, default=100, help = 'number of bins to consider before end of trial')
    parser.add_argument('--bins_after_trial_end', type=int, default = 100, help = 'number of bins to consider after end of trial')
    parser.add_argument('--n_repeats_per_fold', type=int, default=1, help = 'number of times each fold is modeled with a different random seed, or chance is computed per fold')
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
    
    print('Running ol_striding_performance.py')
    print(args)

    # load data
    print('Loading data ...')
    data = load_rdbmat(args.participant, args.session, args.required_keys)

    # get training data
    print('Loading training data ...')
    x_spikepow_speech, x_spikepow_cue, x_spikepow_trial_end, x_threshcross_speech, x_threshcross_cue, x_threshcross_trial_end, y_word, y_amp = get_training_data(data)

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
    fold_acc_matrix_cue, fold_acc_matrix_cue_chance = train_logistic_regression(x_neural_feat_cue, y_word, y_amp) # n_folds x pred_stream_len

    # logistic regression classification - speech
    #--------------------------------------------------
    print('Training logistic regression on speech - cross-validation ...')
    fold_acc_matrix_speech, fold_acc_matrix_speech_chance = train_logistic_regression(x_neural_feat_speech, y_word, y_amp) # n_folds x pred_stream_len

    # logistic regression classification - end
    #--------------------------------------------------
    print('Training logistic regression on trial end - cross-validation ...')
    fold_acc_matrix_trial_end, fold_acc_matrix_trial_end_chance = train_logistic_regression(x_neural_feat_trial_end, y_word, y_amp) # n_folds x pred_stream_len


    # plot results
    plot_striding_performance(fold_acc_matrix_speech, fold_acc_matrix_cue, fold_acc_matrix_trial_end,
                              fold_acc_matrix_speech_chance, fold_acc_matrix_cue_chance, fold_acc_matrix_trial_end_chance)