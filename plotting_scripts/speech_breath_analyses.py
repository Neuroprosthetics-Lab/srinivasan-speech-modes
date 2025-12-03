# This script plots the analyses of sentence-loudness and instructed breath tasks during which a breath belt was worn by the participant.
# This script plots the breath belt expansion during both tasks, psths from example electrodes and decoding accuracy under different task conditions.

import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
import pandas as pd
import seaborn as sns
from scipy.stats import ranksums

'''
Example cmd:
For t15,
    python speech_breath_analyses.py --participant t15 --nbins_before_onset 150 --nbins_after_onset 150 --savepath_data ../plotting_data/t15/speech-breath/speech_breath_analyses/ --savepath_fig ../plotting_figures/t15/speech-breath/speech_breath_analyses/
For t16,
    python ./speech_breath_analyses.py --participant t16 --nbins_before_onset 150 --nbins_after_onset 150 --savepath_data ../plotting_data/t16/speech-breath/speech_breath_analyses/ --savepath_fig ../plotting_figures/t16/speech-breath/speech_breath_analyses/
'''

#--------------------------------------------
# global variables
#--------------------------------------------
breath_types = ['NORMALLY', 'DEEPLY']
speech_types = ['NORMAL', 'LOUD']
linewidth = 3

# plotting
scatter_size = 30
fontsize = 13
my_color_breath = [(230/255, 143/255, 172/255), (153/255, 15/255, 75/255)]
my_color_speech = [(86/255, 180/255, 233/255), (31/255, 120/255, 180/255)]
my_color = 'navy'
bar_width = 0.7
arrays = {
    't15': ['v6v', 'M1', '55b', 'd6v'], # correct_electrode_mapping = 1
    't16': ['55b/PEF', '6v', 'HK1', 'HK2'],
}

#--------------------------------------------
# functions
#--------------------------------------------
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
        plt.text(0.5, 0.5, "breathe", ha='center', fontsize=fontsize)
        plt.text(2.5, 0.5, "loudness level", ha='center', fontsize=fontsize)
    elif args.participant == 't16':
        plt.text(0.5, -1.5, "breathe", ha='center', fontsize=fontsize)
        plt.text(2.5, -1.5, "loudness level", ha='center', fontsize=fontsize)

    plt.suptitle(f'{args.participant.upper()}', fontsize = fontsize)
    fig.tight_layout()
    # plt.show()
    plt.savefig(f'{args.savepath_fig}{args.participant}_breath_expansion_statistics.png', format='png')
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
    plt.savefig(f'{args.savepath_fig}{args.participant}_belt_plot.png', format='png')
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

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05) 
    # plt.show()
    plt.savefig(f'{args.savepath_fig}{args.participant}_psth.png', format='png')

    return      


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
    plt.savefig(f'{args.savepath_fig}{args.participant}_breath_speech_classification_acc.png', format='png')

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
    plt.savefig(f'{args.savepath_fig}{args.participant}_breath_speech_classification_cf.png', format='png')

    return


if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()
    
    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)

    #-------------------------------------------------
    # load breath belt expansion
    print('Loading breath belt data during instructed breathing task....')
    with open(f'{args.savepath_data}{args.participant}_breath_belt_expansion_during_breath.pkl', 'rb') as f:
        breath_expansion_during_breath = pkl.load(f)
        breath_expansion_breath = breath_expansion_during_breath['breath_expansion']
        breath_belt_aligned_to_breath = breath_expansion_during_breath['aligned_breath_trough']

    print('Loading breath belt data during speech task...')
    with open(f'{args.savepath_data}{args.participant}_breath_belt_expansion_during_speech.pkl', 'rb') as f:
        breath_expansion_during_speech = pkl.load(f)
        breath_expansion_speech = breath_expansion_during_speech['breath_expansion']
        breath_belt_aligned_to_speech = breath_expansion_during_speech['aligned_breath_to_speech']

    print('Plotting breath belt expansion and statistics...')
    plot_joint_breath_belt(breath_belt_aligned_to_breath, my_color_breath, breath_belt_aligned_to_speech, my_color_speech)
    plot_joint_breath_expansion_statistics(breath_expansion_breath, my_color_breath, breath_expansion_speech, my_color_speech)
    
    #-------------------------------------------------
    # load breath and speech psth
    print('Loading speech thx...')
    with open(f'{args.savepath_data}{args.participant}_aligned_speech_thx.pkl', 'rb') as f:
        data = pkl.load(f)
        avg_speech_thx = data['avg_speech_thx']
        sem_speech_thx = data['sem_speech_thx']

    print('Loading breath thx...')
    with open(f'{args.savepath_data}{args.participant}_aligned_breath_thx.pkl', 'rb') as f:
        data = pkl.load(f)
        avg_breath_thx = data['avg_breath_thx']
        sem_breath_thx = data['sem_breath_thx']

    # plot particular channel psth
    print('Plotting particular channels ...')
    plt_channel = {
        't15': [141, 97, 34], # channel (0-indexed), ordered according to implanted arrays
        't16': [5, 67, 36], # channel (0-indexed), ordered according to implanted arrays (only speech arrays considered)
    }
    plot_psth_given_channel(avg_breath_thx, sem_breath_thx, avg_speech_thx, sem_speech_thx, plt_channel[args.participant]) # 0-indexed channel

    #--------------------------------------------------
    # load classification accuracy (neural decoder trained on speech or breath and tested within same-task or cross-task)
    print('Loading neural decoder classification results...')
    with open(f'{args.savepath_data}{args.participant}_classification_results.pkl', 'rb') as f:
        clf_results = pkl.load(f)
        acc = clf_results['acc']
        acc_chance = clf_results['acc_chance']
        cf = clf_results['cf']
    
    print('Plotting neural decoder classification accuracy and confusion matrix...')
    plot_results(acc, acc_chance)
    plot_results_cf(cf)

    print('DONE!')