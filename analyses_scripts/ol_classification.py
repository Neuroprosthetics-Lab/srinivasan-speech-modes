import argparse
import os
import scipy
import numpy as np
import pickle as pkl
from pathlib import Path 
import math
from functions import get_audio_onset_offset
from datetime import datetime
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sys
import seaborn as sns

'''
Example cmd:
For t15,
    python ol_classification.py --participant t15 --session word-loudness --nbins_before_onset 60 --nbins_after_onset 60 --savepath_fig ../analyses_figures/t15/word-loudness/ol_classification/ --savepath_data ../analyses_figures_data/t15/word-loudness/ol_classification/
For t16,
    python ol_classification.py --participant t16 --session word-loudness --nbins_before_onset 60 --nbins_after_onset 60 --savepath_fig ../analyses_figures/t16/word-loudness/ol_classification/ --savepath_data ../analyses_figures_data/t16/word-loudness/ol_classification/

Sample neural data will be loaded from ../sample_neural_data/{participant}/{session}/.
Results obtained from this script will be saved in savepath_data.
Figures generated using those results will be saved in savepath_fig.
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
n_channels = 256
n_electrodes_per_array = 64
bin_size_ms = 10
fs = 30000
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
# words = ['be', 'my', 'know', 'do', 'have', 'going'] # use when all the data is provided via Dryad
words = ['be', 'my', ] # use with sample data provided in Github

arrays = {
    't15': ['M1', 'v6v','d6v','55b'], # correct_electrode_mapping = 0
    't16': ['55b', '6v'],#, 'HK1', 'HK2'],
}

# plotting
fontsize = 15
linewidth = 5
my_color = 'navy'
my_color_all_array = 'green'
bar_width = 1.2
array_plotting_order = {
    't15': ['55b', 'd6v', 'M1', 'v6v'], # using_correct_electrode_mapping = 0
    't16': ['55b', '6v'],#,'HK1','HK2'] # only speech arrays needed
}

#--------------------------------------------
# functions
#--------------------------------------------
def load_rdbmat(participant, session, required_keys):
    # load data
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



def get_training_data(data):

    # get data around speech onset
    x_spikepow = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels)) # n_trials x time bins x channels
    x_threshcross = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels)) # n_trials x time bins x channels
    y_word = np.empty((0, 1)) # n_trials x word label 
    y_amp = np.empty((0, 1)) # n_trials x amplitude label
    
    n_trials = len(data['cue'])
    if args.participant == 't15':
        valid_trial_inds = [i for i in range(n_trials) if 'DO NOTHING' not in data['cue'][i] and max(np.squeeze(data['predaudio16k'])[i]) != 0]
    elif args.participant in ['t16', 't19']:
        valid_trial_inds = [i for i in range(n_trials) if 'DO NOTHING' not in data['cue'][i]]
    print('Total number of usable trials:', len(valid_trial_inds))

    for ind in valid_trial_inds:
        
        # current data
        cue = data['cue'][ind].strip()
        amp_label = [amplitudes.index(cue.split(':')[0])]
        word_label = [words.index(cue.split(':')[-1].strip())]
        spikepow = np.squeeze(data['spikepow_from_delay'])[ind]
        threshcross = np.squeeze(data['threshcross_from_delay'])[ind]
        delay_duration_ms = np.squeeze(data['delay_duration_ms'])[ind]
        binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)
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
            
        # binned start and end ind
        start_ind = math.floor((start_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index
        end_ind = math.ceil((end_ind/fs) * (1000/bin_size_ms)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index


        if np.expand_dims(spikepow[binned_delay_duration + (start_ind - args.nbins_before_onset): binned_delay_duration + (start_ind + args.nbins_after_onset), :], 0).shape[1] == args.nbins_before_onset + args.nbins_after_onset:
            
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



def train_logistic_regression(x_neural_feat, y_word, y_amp):

    results = [] # length == n_folds x n_repeats (first n_repeat entries belong to first fold)
    results_cf = [] # for confusion matrices
    results_chance = [] # length == n_folds x n_repeats (first n_repeat entries belong to first fold)

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
        x_test = x_neural_feat[test_inds, :, :]
        y_test = y_amp[test_inds, :]

        # train data
        x_train = x_neural_feat[train_inds, :, :]
        y_train = y_amp[train_inds, :].squeeze()

        print('Train and test data shapes:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        for repeat in range(args.n_repeats_per_fold):
            print(f'Fold {fold} Repetition {repeat}')
            
            # build and fit a classifier with a different seed
            if repeat == 0:
                random_seed = np.random.randint(0, 10000)
                clf = LogisticRegression(max_iter = 1000, random_state = random_seed) # random seed doesn't chance performance!!! as it is the same data
                clf.fit(np.reshape(x_train, (x_train.shape[0],-1)), y_train)
                y_pred = clf.predict(np.reshape(x_test, (x_test.shape[0], -1)))

                acc_binary = y_test.squeeze() == y_pred # binary values for correct and incorrect predictions
                acc = np.sum(acc_binary) / len(acc_binary)
                print(f'Fold: {fold}, Repetition: {repeat}, Word: {words[fold]}, Accuracy: {acc}')
                results.append(acc)

                # compute confusion matrix for this fold and repeat
                cf = confusion_matrix(y_test.squeeze(), y_pred, normalize = 'true')
                print(f'Fold: {fold}, Repetition: {repeat}, Word: {words[fold]}, Confusion Matrix: {cf}')
                results_cf.append(cf)

            # compute chance given the same random seed but fit on shuffled data
            clf = LogisticRegression(max_iter = 1000, random_state = random_seed)
            clf.fit(np.reshape(x_train, (x_train.shape[0],-1)), np.random.permutation(y_train))
            y_pred = clf.predict(np.reshape(x_test, (x_test.shape[0], -1)))

            acc_binary = y_test.squeeze() == y_pred # binary values for correct and incorrect predictions
            acc = np.sum(acc_binary) / len(acc_binary)
            print(f'Fold: {fold}, Repetition: {repeat}, Word: {words[fold]}, Chance Accuracy: {acc}')
            results_chance.append(acc)

    print('Accuracy across folds and repetitions:', results)
    print('Mean Accuracy across folds and repetitions:', np.mean(results))
    print('Chance Accuracy across folds and repetitions:', results_chance)
    print('Mean Chance Accuracy across folds and repetitions:', np.mean(results_chance))
    print('Mean Confusion Matrix across folds:', np.mean(results_cf, axis = 0))
    return results, results_chance, results_cf


def plot_performance_accuracy(fold_accuracies):

    fontsize = 17
    # plot all arrays and per array performance
    fig = plt.figure(figsize = (5.5,5.5))

    all_chance = np.mean(fold_accuracies['all_chance'])
    arrays_to_plot = copy.deepcopy(array_plotting_order[args.participant])
    arrays_to_plot.append('all')

    plot_acc = [np.mean(fold_accuracies[arr]) for arr in arrays_to_plot]
    plot_std = [np.std(fold_accuracies[arr]) for arr in arrays_to_plot]
    plot_sem = [np.std(fold_accuracies[arr])/np.sqrt(len(fold_accuracies[arr])) for arr in arrays_to_plot]
    plot_acc_str = [str(s*100) for s in plot_acc] # accuracy values as string

    if args.participant =='t15':
        x_range_for_plot = np.arange(0, len(arrays_to_plot)-1+2, 1.5)
    elif args.participant == 't16':
        x_range_for_plot = np.arange(0, len(arrays_to_plot), 1.5)
        print(x_range_for_plot)

    # plot each array
    plt.bar(x_range_for_plot, plot_acc[:-1], color = my_color, width = bar_width, label = '_hidden')
    plt.errorbar(x_range_for_plot, plot_acc[:-1], yerr = plot_std[:-1], fmt = 'none', color = 'black', elinewidth = 3)
    for i in range(len(plot_acc[:-1])):
        plt.text(x_range_for_plot[i], plot_acc[i]-0.08, plot_acc_str[i][:4], fontsize = fontsize, color = 'white', ha = 'center')


    # all arrays
    plt.hlines(plot_acc[-1], xmin = x_range_for_plot[0] - 1, xmax = x_range_for_plot[-1] + 1, linestyle = '--', color = my_color_all_array, linewidth = linewidth, label = f'All arrays ({plot_acc[-1] * 100:.1f}%)') # all arrays

    # chance (for all arrays)
    plt.hlines(all_chance, xmin = x_range_for_plot[0] - 1, xmax = x_range_for_plot[-1] + 1, linestyle = '--', color = 'black', linewidth = linewidth, label = f'Chance ({all_chance * 100:.1f}%)') # all array chance

    # compute if array performance is significantly above chance (% of times performance is above chance)
    p_value_per_array = []
    for arr in arrays_to_plot[:-1]:
        p_value_per_array.append(1 - sum([1 for chance_acc in fold_accuracies[f'{arr}_chance'] if np.mean(fold_accuracies[arr]) > chance_acc]) / len(fold_accuracies[f'{arr}_chance']))

    print('P-value per array:', p_value_per_array)
    for i in range(len(p_value_per_array)):
        if p_value_per_array[i] < 0.05:
            plt.text(x_range_for_plot[i] + 0.2 , plot_acc[i] + 0.001, '*', fontsize = fontsize, color = 'black', ha = 'center', fontweight = 'bold')
    
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 

    plt.ylim([0, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100], fontsize = fontsize)
    plt.ylabel('Accuracy (%)', fontsize = fontsize)
    if args.participant == 't16': # make the label for 55b as 55b/PEF
        arrays_to_plot[0] = '55b/PEF'
    plt.xticks(x_range_for_plot, arrays_to_plot[:-1], fontsize = fontsize)
    plt.xlim([x_range_for_plot[0] - 1, x_range_for_plot[-1] + 1])
    
    plt.legend(bbox_to_anchor=(0.3, 1.2), loc='upper left', fontsize = fontsize)
    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_ol_classification_acc_{formatted_datetime}.png', format='png')

    return


def plot_performance_confusion_matrix(cf, mean_acc = None):

    mean_cf = np.mean(cf, axis = 0)

    fig = plt.figure(figsize=(6, 5))
    ax = sns.heatmap(mean_cf*100, annot=True, fmt=".1f", cmap='bone_r', vmin = 0, vmax = 100,
                xticklabels = amplitudes, yticklabels = amplitudes, annot_kws={"size": fontsize, "color": "white"})
    plt.xticks(np.arange(len(amplitudes)) + 0.5, amplitudes, fontsize = fontsize - 1)
    plt.yticks(np.arange(len(amplitudes)) + 0.5, amplitudes, fontsize = fontsize - 1)
    plt.xlabel('Predicted', fontsize = fontsize)
    plt.ylabel('True', fontsize = fontsize)

    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)

    # Loop through annotations and change font color dynamically
    cmap = plt.cm.get_cmap('bone_r')
    for text in ax.texts:
        value = float(text.get_text())  # Get numerical value of the cell
        if value == 0:
            text.set_text(int(value))
        text.set_color("black" if value < 50 else "white")  # Use black for light backgrounds, white for dark

    if mean_acc is not None:
        plt.title(f'Accuracy: {mean_acc*100:.2f}%', fontsize = fontsize)

    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_ol_classification_cf_{formatted_datetime}.png', format='png')

    return



if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms', 'predaudio16k', 'spikepow_from_delay', 'threshcross_from_delay'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--n_repeats_per_fold', type=int, default=1, help = 'number of times each fold is modeled with a different random seed, or chance is computed per fold')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if args.participant == 't16':
        args.required_keys.extend(['speech_onsets', 'speech_offsets'])
        args.required_keys.remove('predaudio16k')

    if not os.path.exists(args.savepath_data):
        os.makedirs(args.savepath_data, exist_ok=True)
    
    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)
    
    print('Running ol_classification_performance.py')
    print(args)

    # load data
    print('Loading data ...')
    data = load_rdbmat(args.participant, args.session, args.required_keys)

    # get training data
    print('Loading training data ...')
    x_spikepow, x_threshcross, y_word, y_amp = get_training_data(data)
    # save training data
    with open(f'{args.savepath_data}{args.participant}_{args.session}_ol_classification_processed_data_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump({
            'x_spikepow': x_spikepow,
            'x_threshcross': x_threshcross,
            'y_word': y_word,
            'y_amp': y_amp
        }, f)


    fold_accuracies = {}    
    print('Number of repeats per fold (or number of chance computations per fold):', args.n_repeats_per_fold)

    # logistic regression classification - all arrays
    # --------------------------------------------------

    if args.participant in ['t15', 't19']:
        x_neural_feat = np.concatenate([x_spikepow, x_threshcross], axis = -1)
    elif args.participant == 't16': # append only speech arrays
        x_neural_feat = np.concatenate([x_spikepow[:, :, :128], x_threshcross[:, :, :128]], axis = -1)
    print('All arrays: Spikepow + threshcross concatenated shape:', x_neural_feat.shape)

    print('All arrays: Training logistic regression')
    fold_acc, fold_acc_chance, fold_cf = train_logistic_regression(x_neural_feat, y_word, y_amp) # n_acc_values == n_folds 
    fold_accuracies['all'] = fold_acc
    fold_accuracies['all_chance'] = fold_acc_chance
    fold_accuracies['all_cf'] = fold_cf

    print('All arrays SPB + THX results')
    print(fold_acc, np.mean(fold_acc))

    # save results
    with open(f'{args.savepath_data}{args.participant}_{args.session}_ol_classification_allarrays_acc_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump({
            'all': fold_acc,
            'all_chance': fold_acc_chance,
            'all_cf': fold_cf,
        }, f)

    # logistic regression classification - per array
    #--------------------------------------------------

    for array_ind in range(len(arrays[args.participant])):
        print('Current array: ', arrays[args.participant][array_ind])

        array_electrode_start = array_ind * n_electrodes_per_array
        array_electrode_end = array_ind * n_electrodes_per_array + n_electrodes_per_array

        print('Electrode start and end (0-indexed):', array_electrode_start, array_electrode_end)

        x_neural_feat = np.concatenate([x_spikepow[:, :, array_electrode_start: array_electrode_end],
                                        x_threshcross[:, :, array_electrode_start: array_electrode_end]],
                                        axis = -1)
        
        print(f'Array {arrays[args.participant][array_ind]}: Spikepow + threshcross concatenated shape:', x_neural_feat.shape)

        print(f'Array {arrays[args.participant][array_ind]}: Training logistic regression')
        fold_acc, fold_acc_chance, fold_cf = train_logistic_regression(x_neural_feat, y_word, y_amp) # n_acc_values == n_folds 
        fold_accuracies[f'{arrays[args.participant][array_ind]}'] = fold_acc
        fold_accuracies[f'{arrays[args.participant][array_ind]}_chance'] = fold_acc_chance
        fold_accuracies[f'{arrays[args.participant][array_ind]}_cf'] = fold_cf

        # save results
        with open(f'{args.savepath_data}{args.participant}_{args.session}_ol_classification_{arrays[args.participant][array_ind]}_acc_{formatted_datetime}.pkl', 'wb') as f:
            pkl.dump({
                f'{arrays[args.participant][array_ind]}': fold_acc,
                f'{arrays[args.participant][array_ind]}_chance': fold_acc_chance,
            }, f)

    # save all results
    with open(f'{args.savepath_data}{args.participant}_{args.session}_ol_classification_acc_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump(fold_accuracies, f)

    # plot results (accuracies)
    plot_performance_accuracy(fold_accuracies)

    # plot results (confusion matrix)
    plot_performance_confusion_matrix(fold_accuracies['all_cf'], mean_acc = np.mean(fold_accuracies['all']))
    
    # save scipt args
    print('Saving scripts args and script ...')
    script_name = sys.argv[0]
    with open(script_name, 'r') as f:
        script_content = f.read()

    with open(f'{args.savepath_data}{args.participant}_{args.session}_ol_classification_performance_{formatted_datetime}.log', 'w') as f:
        f.write(str(args))
        f.write('\n\n----------\n\n')
        f.write(script_content)