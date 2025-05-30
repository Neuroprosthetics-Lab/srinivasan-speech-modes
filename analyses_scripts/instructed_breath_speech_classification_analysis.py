import argparse
import os
import scipy
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
from session_metadata import incorrect_trials
import pickle as pkl
from datetime import datetime
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

'''
example run cmd:
python instructed_breath_speech_classification_analysis.py --participant <> --session <> --nbins_before_onset <> --nbins_after_onset <> --savepath_fig <> --savepath_data <> --n_chances 100
'''

#--------------------------------------------
# global variables
#--------------------------------------------
breath_types = ['NORMAL', 'DEEP']
speech_types = ['NORMAL', 'LOUD']
bin_size_ms = 10
fs = 30000
n_channels = 256

# plotting
my_color = 'navy'
bar_width = 0.7
fontsize = 14


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


def get_breath_training_data(data):

    # collect breath w.r.t. start of exhalation 
    # as we want to compare it with the activity during speech onset which happens during exhalation

    n_trials = len(data['cue'])

    x_spikepow = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))
    x_threshcross = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))
    y = np.empty((0,1))
    for tr in range(n_trials):

        # get current data
        spikepow = np.squeeze(data['spikepow_from_delay'])[tr]
        threshcross = np.squeeze(data['threshcross_from_delay'])[tr]
        delay_duration_ms = np.squeeze(data['delay_duration_ms'])[tr]
        binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)
        binned_breath_troughs = np.squeeze(data['binned_breath_min_loc'])[tr].squeeze()
        binned_breath_peaks = np.squeeze(data['binned_breath_max_loc'])[tr].squeeze(axis = 0) 
        cue = data['cue'][tr]

        for breath_type in breath_types:
            if breath_type in cue:

                if args.participant == 't15':
                    for i in range(len(binned_breath_troughs) - 1): # exhalation start is trough
                        if binned_delay_duration + (binned_breath_troughs[i] - args.nbins_before_onset) >= 0 and binned_delay_duration + (binned_breath_troughs[i] + args.nbins_after_onset) <= threshcross.shape[0]:
                            
                            temp_spb = spikepow[binned_delay_duration + (binned_breath_troughs[i] - args.nbins_before_onset): binned_delay_duration + (binned_breath_troughs[i] + args.nbins_after_onset), :] 
                            x_spikepow = np.append(x_spikepow, np.expand_dims(temp_spb, 0), axis = 0)
                            
                            temp_thx = threshcross[binned_delay_duration + (binned_breath_troughs[i] - args.nbins_before_onset): binned_delay_duration + (binned_breath_troughs[i] + args.nbins_after_onset), :] 
                            x_threshcross = np.append(x_threshcross, np.expand_dims(temp_thx, 0), axis = 0)

                            y = np.append(y, np.expand_dims([breath_types.index(breath_type)], 0), axis = 0)

                elif args.participant == 't16':
                    for i in range(len(binned_breath_peaks) - 1): # exhalation start is peak
                        if binned_delay_duration + (binned_breath_peaks[i] - args.nbins_before_onset) >= 0 and binned_delay_duration + (binned_breath_peaks[i] + args.nbins_after_onset) <= threshcross.shape[0]:
                            
                            temp_spb = spikepow[binned_delay_duration + (binned_breath_peaks[i] - args.nbins_before_onset): binned_delay_duration + (binned_breath_peaks[i] + args.nbins_after_onset), :] 
                            x_spikepow = np.append(x_spikepow, np.expand_dims(temp_spb, 0), axis = 0)
            
                            temp_thx = threshcross[binned_delay_duration + (binned_breath_peaks[i] - args.nbins_before_onset): binned_delay_duration + (binned_breath_peaks[i] + args.nbins_after_onset), :] 
                            x_threshcross = np.append(x_threshcross, np.expand_dims(temp_thx, 0), axis = 0)

                            y = np.append(y, np.expand_dims([breath_types.index(breath_type)], 0), axis = 0)
                

    print(f'Spikepow, threshcross, breath_label shapes:', x_spikepow.shape, x_threshcross.shape, y.shape)
    for breath_type in breath_types:
        print(f'Number of {breath_type} data: ', np.where(y == breath_types.index(breath_type))[0].shape)
    return x_spikepow, x_threshcross, y



def get_speech_training_data(data):

    n_trials = len(data['cue'])

    x_spikepow = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))
    x_threshcross = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels))
    y = np.empty((0,1))
    for tr in range(n_trials):
        # get current data
        spikepow = np.squeeze(data['spikepow_from_delay'])[tr]
        threshcross = np.squeeze(data['threshcross_from_delay'])[tr]
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
        
        assert len(words) == len(start_ind) == len(end_ind)

        for i in range(len(start_ind)):
            if start_ind[i] != -1 and end_ind[i] != 0: # there is a valid annotation for this word, automatically rejects only breath trials which have -1 annotation
                if binned_delay_duration + start_ind[i] - args.nbins_before_onset >= 0 and binned_delay_duration + start_ind[i] + args.nbins_after_onset < threshcross.shape[0]:
                    
                    temp_spb = spikepow[binned_delay_duration + start_ind[i] - args.nbins_before_onset: binned_delay_duration + start_ind[i] + args.nbins_after_onset, :]
                    x_spikepow = np.append(x_spikepow, np.expand_dims(temp_spb, 0), axis = 0)
                
                    temp_thx = threshcross[binned_delay_duration + start_ind[i] - args.nbins_before_onset: binned_delay_duration + start_ind[i] + args.nbins_after_onset, :]
                    x_threshcross = np.append(x_threshcross, np.expand_dims(temp_thx, 0), axis = 0)

                    if words[i].isupper():
                        y = np.append(y, np.expand_dims([speech_types.index('LOUD')], 0), axis = 0)
                    elif words[i].islower():
                        y = np.append(y, np.expand_dims([speech_types.index('NORMAL')], 0), axis = 0)

    print(f'Spikepow, threshcross, breath_label shapes:', x_spikepow.shape, x_threshcross.shape, y.shape)
    for speech_type in speech_types:
        print(f'Number of {speech_type} data: ', np.where(y == speech_types.index(speech_type))[0].shape)
    return x_spikepow, x_threshcross, y


def train_logistic_regression(x_breath, y_breath, x_speech, y_speech):

    train_test_acc = {}
    train_test_acc_chance = {}
    train_test_cf = {}
    for train_condition in ['breath', 'speech']:
        for test_condition in ['breath', 'speech']:
            train_test_acc[f'{train_condition}_{test_condition}'] = []
            train_test_cf[f'{train_condition}_{test_condition}'] = []
            train_test_acc_chance[f'{train_condition}_{test_condition}'] = []
    
    data_set = [
        [x_breath, y_breath, x_speech, y_speech],
        [x_speech, y_speech, x_breath, y_breath],
    ]
    data_condition = [
        ['breath', 'speech'],
        ['speech', 'breath']
    ]

    for i in range(len(data_set)):

        x1, y1, x2, y2 = data_set[i][0], data_set[i][1], data_set[i][2], data_set[i][3]
        train_cond, test_cond = data_condition[i][0], data_condition[i][1]
        print(f'Training on {train_cond}')

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(kf.split(x1, y1)):
            print(f'Fold {fold + 1}')
            
            x_train, x_test = x1[train_idx], x1[test_idx]
            y_train, y_test = y1[train_idx], y1[test_idx]
            y_train = y_train.squeeze()
            print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
            print('Train labels:', np.unique(y_train, return_counts=True))
            print('Test labels:', np.unique(y_test, return_counts=True))
        
            # Train logistic regression model
            random_seed = np.random.randint(0, 10000)
            clf = LogisticRegression(max_iter = 1000, random_state = random_seed)
            clf.fit(np.reshape(x_train, (x_train.shape[0],-1)), y_train)
            y_pred = clf.predict(np.reshape(x_test, (x_test.shape[0], -1)))

            # test on same condition
            acc_binary = y_test.squeeze() == y_pred # binary values for correct and incorrect predictions
            acc = np.sum(acc_binary) / len(acc_binary)
            cf = confusion_matrix(y_test.squeeze(), y_pred, normalize = 'true')
            train_test_acc[f'{train_cond}_{train_cond}'].append(acc)
            train_test_cf[f'{train_cond}_{train_cond}'].append(cf)

            # test on cross condition
            print(x2.shape, y2.shape)
            y_pred = clf.predict(np.reshape(x2, (x2.shape[0], -1)))
            acc_binary = y2.squeeze() == y_pred # binary values for correct and incorrect predictions
            acc = np.sum(acc_binary) / len(acc_binary)
            cf = confusion_matrix(y2.squeeze(), y_pred, normalize = 'true')
            train_test_acc[f'{train_cond}_{test_cond}'].append(acc)
            train_test_cf[f'{train_cond}_{test_cond}'].append(cf)

    for key in train_test_acc.keys():
        train_test_acc[key] = np.array(train_test_acc[key])
        print(f'Accuracy for {key}:', train_test_acc[key], train_test_acc[key].shape)     
    
    return train_test_acc, train_test_cf

def train_logistic_regression_chance(x_breath, y_breath, x_speech, y_speech):
    train_test_acc_chance = {}
    
    for train_condition in ['breath', 'speech']:
        for test_condition in ['breath', 'speech']:

            train_test_acc_chance[f'{train_condition}_{test_condition}'] = [] # list of list, each sub-list len = n_folds, len overall list = n_chance
    
    data_set = [
        [x_breath, y_breath, x_speech, y_speech],
        [x_speech, y_speech, x_breath, y_breath],
    ]
    data_condition = [
        ['breath', 'speech'],
        ['speech', 'breath']
    ]

    for i in range(len(data_set)):

        x1, y1, x2, y2 = data_set[i][0], data_set[i][1], data_set[i][2], data_set[i][3]
        train_cond, test_cond = data_condition[i][0], data_condition[i][1]
        print(f'Training on {train_cond}')

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for chance in range(args.n_chances):
            print(f'Chance {chance + 1}')

            acc_same_condition_across_folds = [] # n_folds
            acc_cross_condition_across_folds = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(x1, y1)):
                print(f'Fold {fold + 1}')
                
                x_train, x_test = x1[train_idx], x1[test_idx]
                y_train, y_test = y1[train_idx], y1[test_idx]
                y_train = y_train.squeeze()
                print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
            
                # Train logistic regression model
                random_seed = np.random.randint(0, 10000)
                clf = LogisticRegression(max_iter = 1000, random_state = random_seed)
                clf.fit(np.reshape(x_train, (x_train.shape[0],-1)), np.random.permutation(y_train))
                y_pred = clf.predict(np.reshape(x_test, (x_test.shape[0], -1)))

                # test on same condition
                acc_binary = y_test.squeeze() == y_pred # binary values for correct and incorrect predictions
                acc = np.sum(acc_binary) / len(acc_binary)
                acc_same_condition_across_folds.append(acc)
                
                
                # test on cross condition
                print(x2.shape, y2.shape)
                y_pred = clf.predict(np.reshape(x2, (x2.shape[0], -1)))
                acc_binary = y2.squeeze() == y_pred # binary values for correct and incorrect predictions
                acc = np.sum(acc_binary) / len(acc_binary)
                acc_cross_condition_across_folds.append(acc)
                
            train_test_acc_chance[f'{train_cond}_{train_cond}'].append(acc_same_condition_across_folds)
            train_test_acc_chance[f'{train_cond}_{test_cond}'].append(acc_cross_condition_across_folds)

    for key in train_test_acc_chance.keys():
        train_test_acc_chance[key] = np.array(train_test_acc_chance[key])
        print(f'Chance accuracy for {key}:', train_test_acc_chance[key], train_test_acc_chance[key].shape)
    
    return train_test_acc_chance
                    


def plot_results(acc, chance_acc):

    fig = plt.figure(figsize= (5,5))

    # Function to determine significance level
    def get_p_label(p):
        if p < 0.05:
            return '*'
        else:
            return 'ns'  # Not significant

    # plot accuracies as bar plot
    for condition in acc.keys():
        plt.bar(list(acc.keys()).index(condition), np.mean(acc[condition]) * 100, color = my_color, width = bar_width, alpha = 0.8)
        plt.text(list(acc.keys()).index(condition), np.mean(acc[condition]) * 100 - 8, f'{(np.mean(acc[condition]) * 100):.1f}', fontsize = fontsize, color = 'white', ha = 'center')
        plt.hlines(np.mean(np.mean(chance_acc[condition], 0)) * 100, xmin = list(acc.keys()).index(condition) - (2*bar_width/3), xmax = list(acc.keys()).index(condition) + (2*bar_width/3), linestyle = '--', color = 'black', linewidth = 2, label = [f'Chance' if condition == 'breath_breath' else '_hidden'][0])

        # compute p-value
        mean_acc_across_folds = np.mean(acc[condition]) # one value
        mean_shuffle_acc_folds = np.mean(chance_acc[condition], axis = -1) # n_chances
        p_value = 1 - (np.sum(mean_acc_across_folds > mean_shuffle_acc_folds) / len(mean_shuffle_acc_folds)) # p-value

        # # how many times is the actual accuracy above chance?
        # threshold = np.percentile(shuffle_acc, 95)
        # n_above_chance = np.sum(np.array(acc[condition]) > np.mean(chance_acc[condition]))

        significance = get_p_label(p_value)
        plt.text(list(acc.keys()).index(condition), np.mean(acc[condition]) * 100 + 5, significance, ha="center", va="bottom", fontsize = fontsize, fontweight="bold")

    plt.ylim([0, 100])
    plt.yticks(np.arange(0, 101, 25), np.arange(0, 101, 25), fontsize = fontsize)
    plt.ylabel('Accuracy (%)', fontsize = fontsize)
    xtick_labels = []
    for k in list(acc.keys()):
        if k.split('_')[0] == 'speech':
            xtick_labels.append('loudness'+ '-' + k.split('_')[1])
        elif k.split('_')[1] == 'speech':
            xtick_labels.append(k.split('_')[0] + '-' + 'loudness')
        else:
            xtick_labels.append(k.split('_')[0] + '-' + k.split('_')[1])

    plt.xticks(np.arange(len(acc)), xtick_labels, fontsize = fontsize, rotation = 20)
    plt.xlabel('Train-test condition', fontsize = fontsize)

    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 
    
    if args.participant == 't15':
        plt.legend(fontsize = fontsize)

    plt.title(f'{args.participant.upper()}', fontsize = fontsize + 2)
    fig.tight_layout()
    # plt.show()

    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_breath_speech_classification_acc_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_breath_speech_classification_acc_{formatted_datetime}.png', format='png')

    return

def plot_results_cf(cf):

    fig, ax = plt.subplots(int(len(cf)/2), int(len(cf)/2), figsize=(5, 5))
    ax = ax.ravel()
    for i in range(len(cf)):
        key = list(cf.keys())[i]
        if key.split('_')[1] == 'breath':
            xticklabels = yticklabels = breath_types
            xticklabels[0] = yticklabels[0] = 'NORMALLY' # for legend purposes
            xticklabels[1] = yticklabels[1] = 'DEEPLY'
        elif key.split('_')[1] =='speech':
            xticklabels = yticklabels = speech_types

        im = sns.heatmap(np.mean(cf[key], axis = 0) * 100, annot=True, fmt=".1f", cmap='bone_r', vmin = 0, vmax = 100,
                xticklabels = xticklabels, yticklabels = yticklabels, annot_kws={"size": fontsize}, ax = ax[i],
                cbar = False)
        ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize = fontsize - 2)
        ax[i].set_yticklabels(ax[i].get_yticklabels(), fontsize = fontsize - 2)

        if i == 0:
            ax[i].set_ylabel('Breath decoder', fontsize = fontsize)
            ax[i].set_title('Tested on breath', fontsize = fontsize)
        elif i == 1:
            ax[i].set_title('Tested on loudness', fontsize = fontsize)
        elif i == 2:
            ax[i].set_ylabel('Loudness decoder', fontsize = fontsize)
        
    plt.suptitle(f'{args.participant.upper()}', fontsize = fontsize + 2)
    # if args.participant == 't15':
    #     # plot colorbar
    #     cbar_ax = fig.add_axes([0.97, 0.05, 0.01, 0.8])  # [left, bottom, width, height]
    #     fig.colorbar(im.get_children()[0], cax=cbar_ax)

    fig.tight_layout()
    # plt.show()

    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_breath_speech_classification_cf_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_breath_speech_classification_cf_{formatted_datetime}.png', format='png')

    return


if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'delay_duration_ms', 'spikepow_from_delay', 'threshcross_from_delay'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--n_chances', type=int, default=1, help = 'number of times each fold is modeled with a different random seed, or chance is computed per fold')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_data):
        os.mkdir(args.savepath_data)
    
    if not os.path.exists(args.savepath_fig):
        os.mkdir(args.savepath_fig)
    
    print('Running instructed_breath_speech_classification_analysis.py')
    print(args)

    # Instructed breath trials
    # ----------------------------------------------------------
    # load only instructed breathing data
    required_keys = args.required_keys + ['binned_breath_min_loc', 'binned_breath_max_loc']
    
    print('Loading data ...') # load data with respect to exhalation as anchor point
    data = load_rdbmat(args.participant, args.session, required_keys)

    # get training data
    print('Getting model data ...')
    x_spikepow_breath, x_threshcross_breath, y_breath = get_breath_training_data(data)

    # Speech trials
    # ----------------------------------------------------------
    # load only speech data
    required_keys = args.required_keys + ['speech_onsets', 'speech_offsets']
    print('Loading data ...')
    data = load_rdbmat(args.participant, args.session, required_keys)

    # get training data
    print('Getting model data ...')
    x_spikepow_speech, x_threshcross_speech, y_speech = get_speech_training_data(data)


    # train model and test with same condition or cross condition
    x_neural_feat_breath = np.concatenate([x_spikepow_breath, x_threshcross_breath], axis = -1)
    x_neural_feat_speech = np.concatenate([x_spikepow_speech, x_threshcross_speech], axis = -1)
    if args.participant == 't16':
        x_neural_feat_breath = np.concatenate([x_spikepow_breath[:, :, :128], x_threshcross_breath[:, :, :128]], axis = -1)
        x_neural_feat_speech = np.concatenate([x_spikepow_speech[:, :, :128], x_threshcross_speech[:, :, :128]], axis = -1)
    print('Input neural features for breath and speech shape:', x_neural_feat_breath.shape, x_neural_feat_speech.shape)

    acc, cf = train_logistic_regression(x_neural_feat_breath, y_breath, x_neural_feat_speech, y_speech)
    chance_acc = train_logistic_regression_chance(x_neural_feat_breath, y_breath, x_neural_feat_speech, y_speech)


    # # save data
    with open(f'{args.savepath_data}{args.participant}_{args.session}_classification_results_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump({
            'acc': acc,
            'acc_chance': chance_acc,
            'cf': cf,
        }, f)

    # plot accuracies
    plot_results(acc, chance_acc)

    # plot confusion matrices
    plot_results_cf(cf)