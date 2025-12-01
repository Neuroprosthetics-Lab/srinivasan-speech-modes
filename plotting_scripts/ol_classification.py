# This script plots the open-loop classification performance results.

import argparse
import os
import numpy as np
import pickle as pkl
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import seaborn as sns

'''
Example cmd:
For t15,
    python ol_classification.py --participant t15 --session word-loudness --nbins_before_onset 60 --nbins_after_onset 60 --savepath_data ../plotting_data/t15/word-loudness/ol_classification/ --savepath_fig ../plotting_figures/t15/word-loudness/ol_classification/
For t16,
    python ol_classification.py --participant t16 --session word-loudness --nbins_before_onset 60 --nbins_after_onset 60 --savepath_data ../plotting_data/t16/word-loudness/ol_classification/ --savepath_fig ../plotting_figures/t16/word-loudness/ol_classification/

Data will be loaded from the specified savepath_data directory.
Figures will be saved in the specified savepath_fig directory.
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going']
arrays = {
    't15': ['M1', 'v6v','d6v','55b'],
    't16': ['55b', '6v'],#, 'HK1', 'HK2'],
    't19': ['M1-s', '55b-p', '55b-a', 'M1-i'],
}

# plotting
fontsize = 15
linewidth = 5
my_color = 'navy'
my_color_all_array = 'green'
bar_width = 1.2
array_plotting_order = {
    't15': ['55b', 'd6v', 'M1', 'v6v'], 
    't16': ['55b', '6v'],#,'HK1','HK2'] # only speech arrays needed
}

#--------------------------------------------
# functions
#--------------------------------------------
def plot_performance_accuracy(fold_accuracies):

    fontsize = 17
    # plot all arrays and per array performance
    fig = plt.figure(figsize = (5.5,5.5))

    all_chance = fold_accuracies['all_chance_mean']
    arrays_to_plot = copy.deepcopy(array_plotting_order[args.participant])
    arrays_to_plot.append('all')

    plot_acc = [fold_accuracies[f'{arr}_mean'] for arr in arrays_to_plot]
    plot_std = [fold_accuracies[f'{arr}_std'] for arr in arrays_to_plot]
    plot_acc_str = [str(s*100) for s in plot_acc] # accuracy values as string

    if args.participant == 't15':
        x_range_for_plot = np.arange(0, len(arrays_to_plot)-1+2, 1.5)
    elif args.participant == 't16':
        x_range_for_plot = np.arange(0, len(arrays_to_plot), 1.5)

    # plot each array
    plt.bar(x_range_for_plot, plot_acc[:-1], color = my_color, width = bar_width, label = '_hidden')
    plt.errorbar(x_range_for_plot, plot_acc[:-1], yerr = plot_std[:-1], fmt = 'none', color = 'black', elinewidth = 3)
    for i in range(len(plot_acc[:-1])):
        plt.text(x_range_for_plot[i], plot_acc[i]-0.08, plot_acc_str[i][:4], fontsize = fontsize, color = 'white', ha = 'center')


    # all arrays
    plt.hlines(plot_acc[-1], xmin = x_range_for_plot[0] - 1, xmax = x_range_for_plot[-1] + 1, linestyle = '--', color = my_color_all_array, linewidth = linewidth, label = f'All arrays ({plot_acc[-1] * 100:.1f}%)') # all arrays

    # chance (for all arrays)
    plt.hlines(all_chance, xmin = x_range_for_plot[0] - 1, xmax = x_range_for_plot[-1] + 1, linestyle = '--', color = 'black', linewidth = linewidth, label = f'Chance ({all_chance * 100:.1f}%)') # all array chance

    for i, arr in enumerate(arrays_to_plot[:-1]):
        if fold_accuracies[f'{arr}_pval'] < 0.05:
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
    plt.savefig(f'{args.savepath_fig}{args.participant}_ol_classification_acc_{formatted_datetime}.png', format='png')

    return


def plot_performance_confusion_matrix(cf, mean_acc = None):

    mean_cf = fold_accuracies['all_cf_mean']
    print(mean_cf.shape)

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
    plt.savefig(f'{args.savepath_fig}{args.participant}_ol_classification_cf_{formatted_datetime}.png', format='png')

    return



if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()
    
    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok = True)
    
    print('Running ol_classification.py')
    print(args)

    ### temp!!!
    with open(f'{args.savepath_data}{args.participant}_ol_classification_results.pkl', 'rb') as f:
        fold_accuracies = pkl.load(f) 
        print(fold_accuracies.keys())

    # plot results (accuracies)
    print('Plotting accuracies...')
    plot_performance_accuracy(fold_accuracies)

    # plot results (confusion matrix)
    print('Plotting confusion matrix...')
    plot_performance_confusion_matrix(fold_accuracies['all_cf_mean'], mean_acc = fold_accuracies['all_mean'])
    
    print('DONE!')