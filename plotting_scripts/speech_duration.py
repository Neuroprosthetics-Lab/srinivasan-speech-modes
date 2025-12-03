# This script plots the speech duration across different words and loudness levels, 
# and decoding accuracy when using minimum speech duration neural data.

import argparse
import os
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pickle as pkl
import math
from datetime import datetime
from scipy.stats import ranksums
from itertools import combinations
import seaborn as sns

'''
Example cmd (when run from this directory; provide python script path appropriately if run from different directory):
For t15,
    python speech_duration.py --participant t15 --session word-loudness --savepath_data ../plotting_data/t15/word-loudness/speech_duration/ --savepath_fig ../plotting_figures/t15/word-loudness/speech_duration/
For t16,
    python speech_duration.py --participant t16 --session word-loudness --savepath_data ../plotting_data/t16/word-loudness/speech_duration/ --savepath_fig ../plotting_figures/t16/word-loudness/speech_duration/

Data will be loaded from the specified savepath_data directory.
Figures will be saved in the specified savepath_fig directory.
'''

fs = 30000
bin_size_ms = 10
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going']
fontsize = 17
target_color = [
    (0.9254902, 0.12156863, 0.14117647),
    (0.98431373, 0.72941176, 0.07058824),
    (0.384, 0.682, 0.2),
    (0.43137255, 0.79607843, 0.85490196),
    [0.45568627, 0.31764706, 0.63529412],
    (0.84705882, 0.2627451, 0.59215686)
]
star_step = 0.1
markersize = 8
markeredgewidth = 2

#--------------------------------------------
# functions
#--------------------------------------------
def plot_speech_duration(stats):

    plt.figure(figsize=(15,7))
    all_positions = []
    x_tick_positions = []

    for i, word in enumerate(words):
        word_data = stats[word]
        positions = np.arange(len(amplitudes)) + i*(len(amplitudes) + 1)  # space between words
        all_positions.extend(positions)
        bplot = plt.boxplot(
            [word_data[l] for l in amplitudes],
            positions=positions,
            patch_artist=True,
            showfliers=True
        )

        # Set color for all bar plots for this word, saturation and linewidth per loudness
        for patch, l in zip(bplot['boxes'], amplitudes):
            patch.set_facecolor('None')
            patch.set_edgecolor(target_color[words.index(word)])
            patch.set_linewidth(linewidth[amplitudes.index(l)])
            patch.set_alpha(alpha[amplitudes.index(l)])

        for i, whisker in enumerate(bplot['whiskers']):
            box_index = i // 2  # two whiskers per box
            whisker.set_color(target_color[words.index(word)])
            whisker.set_linewidth(linewidth[box_index])

        for i, cap in enumerate(bplot['caps']):
            box_index = i // 2  # two caps per box
            cap.set_color(target_color[words.index(word)])
            cap.set_linewidth(linewidth[box_index])

        for i, median in enumerate(bplot['medians']):
            box_index = i  # one median per box
            median.set_color(target_color[words.index(word)])
            median.set_linewidth(linewidth[box_index])   
        
        for i, flier in enumerate(bplot['fliers']):
            flier.set_alpha(alpha[i])
            flier.set_markeredgecolor(target_color[words.index(word)])
            flier.set_markersize(markersize)
            flier.set_markeredgewidth(markeredgewidth)

        x_tick_positions.append(positions.mean())

    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    # Create custom legend handles for loudness
    if args.participant == 't15':
        loudness_handles = []
        for l in range(len(amplitudes)):
            line = mlines.Line2D([], [], color='gray', linewidth=linewidth[l], alpha=alpha[l],
                                label=amplitudes[l])
            loudness_handles.append(line)

        plt.legend(handles=loudness_handles, loc='lower right', fontsize = fontsize)

    for i, word in enumerate(words):
        prev_line_y = -1
        word_data = stats[word]
        for (l1, l2) in combinations(amplitudes, 2):
            x1 = i*(len(amplitudes)+1) + amplitudes.index(l1)
            x2 = i*(len(amplitudes)+1) + amplitudes.index(l2)
            # Base y from current max
            y = max(
                np.max(word_data[l1]),
                np.max(word_data[l2]),
            ) + star_step

            if prev_line_y != -1:
                y = prev_line_y + star_step

            stat, p_val = ranksums(word_data[l1], word_data[l2])

            # bonferroni correction (p_val < significance_level / n_comparisons for significant results)
            if args.participant == 't16':
                n_comparisons = math.comb(len(amplitudes)-1, 2) # 3C2 = 3, MIME is not considered
            elif args.participant == 't15':
                n_comparisons = math.comb(len(amplitudes), 2) # 4C2 = 6

            if not (args.participant == 't16' and l1 == 'MIME'):
                # Draw line
                plt.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], c='k')
                # Add text
                if p_val < 0.001/n_comparisons:
                    star = '***'
                elif p_val < 0.01/n_comparisons:
                    star = '**'
                elif p_val < 0.05/n_comparisons:
                    star = '*'
                else:
                    star = 'ns'
                plt.text((x1+x2)/2, y + 0.02, star, ha='center', va='bottom', fontsize = fontsize - 2)
                
                prev_line_y = y 

    plt.xticks(x_tick_positions, words, fontsize = fontsize)
    plt.ylabel('Attempted speech duration (ms)', fontsize = fontsize)
    plt.yticks(np.arange(0, 2, 0.2), [f'{int(num * 1000)}' for num in np.arange(0, 2, 0.2)], fontsize = fontsize)
    plt.title(args.participant.upper(), fontsize = fontsize + 2)
    plt.tight_layout()
    # plt.show()

    plt.savefig(f'{args.savepath_fig}/speech_duration_{args.participant}_{args.session}_{formatted_datetime}.png', format='png')

    return

def plot_performance_confusion_matrix(mean_cf, mean_acc = None):

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
        plt.title(f'Accuracy: {mean_acc*100:.1f}%', fontsize = fontsize)

    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_min_duration_loudness_classification_cf_{formatted_datetime}.png', format='png')

    return
         

if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if args.participant == 't15':
        alpha = [0.25, 0.5, 0.75, 1]
        linewidth = [1.5, 2.5, 3.5, 4.5]
    elif args.participant == 't16':
        alpha = [0, 0.5, 0.75, 1] # make alpha for MIME 0
        linewidth = [0, 2.5, 3.5, 4.5] # make linewidth for MIME 0

    
    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)

    print('Running speech_duration.py...')

    # save speech duration stats
    print('Loading speech duration stats...')
    with open(f'{args.savepath_data}{args.participant}_speech_duration_stats.pkl', 'rb') as f:
        speech_duration_stats = pkl.load(f)

    # plot duration stats
    print('Plotting speech duration stats...')
    plot_speech_duration(speech_duration_stats)

    # plot decoding accuracy and confusion matrix when decoding from minimum duration
    print('Loading decoding performance using minimum speech duration neural data...')
    with open(f'{args.savepath_data}{args.participant}_min_duration_classification_acc.pkl', 'rb') as f:
        data = pkl.load(f)

    plot_performance_confusion_matrix(data['mean_cf'], data['mean_acc'])

    print('DONE!')