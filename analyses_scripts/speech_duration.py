import argparse
import os
import scipy
import numpy as np
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from functions import get_audio_onset_offset
import pickle as pkl
import math
from datetime import datetime
from scipy.stats import ranksums
from itertools import combinations

'''
Example cmd:
For t15,
    python speech_duration.py --participant t15 --session word-loudness --savepath_fig ../analyses_figures/t15/word-loudness/speech_duration/ --savepath_data ../analyses_figures_data/t15/word-loudness/speech_duration/
For t16,
    python speech_duration.py --participant t16 --session word-loudness --savepath_fig ../analyses_figures/t16/word-loudness/speech_duration/ --savepath_data ../analyses_figures_data/t16/word-loudness/speech_duration/

Sample neural data will be loaded from ../sample_neural_data/{participant}/{session}/.
Results obtained from this script will be saved in savepath_data.
Figures generated using those results will be saved in savepath_fig.
'''

fs = 30000
bin_size_ms = 10
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going'] # use when all the data is provided via Dryad
fontsize = 17
target_color = [
    (0.9254902, 0.12156863, 0.14117647),
    (0.98431373, 0.72941176, 0.07058824),
    # [0.57254902, 0.78431373, 0.24313725], 
    (0.384, 0.682, 0.2),
    (0.43137255, 0.79607843, 0.85490196),
    # (0.26529412, 0.40686275, 0.72490196),
    [0.45568627, 0.31764706, 0.63529412],
    (0.84705882, 0.2627451, 0.59215686)
]
star_step = 0.1
markersize = 8
markeredgewidth = 2

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



def compute_speech_duration(data):

    speech_duration_stats = {} # duration in ms

    trial_counter = 0

    for word in words:
        speech_duration_stats[word] = {}
        for amp in amplitudes:
            speech_duration_stats[word][amp] = []
            inds = [i for i in range(len(data['cue'])) if amp in data['cue'][i] and word in data['cue'][i]]
                
            for ind in inds:

                start_ind = np.squeeze(data['speech_onsets'])[ind].squeeze()
                end_ind = np.squeeze(data['speech_offsets'])[ind].squeeze()
                # start ind and end ind in seconds
                start_ind = start_ind/fs 
                end_ind = end_ind/fs 

                 # duration in ms
                if args.participant == 't15' and (end_ind - start_ind) < 0.2: # less than 0.2 s unlikely a successful attempt of speech
                    continue
                speech_duration_stats[word][amp].append((end_ind - start_ind))
                trial_counter += 1
    
    min_duration_across_all_trials = 100
    for word in words:
        for amp in amplitudes:
            print(f'Min, max, mean duration (s) for {word} {amp}: {np.min(speech_duration_stats[word][amp])}, {np.max(speech_duration_stats[word][amp])}, {np.mean(speech_duration_stats[word][amp])}')
            if np.min(speech_duration_stats[word][amp]) < min_duration_across_all_trials:
                min_duration_across_all_trials = np.min(speech_duration_stats[word][amp])

    return speech_duration_stats, min_duration_across_all_trials


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
            # if not (args.participant == 't16' and l == 'MIME'):
            patch.set_facecolor('None')
            patch.set_edgecolor(target_color[words.index(word)])
            patch.set_linewidth(linewidth[amplitudes.index(l)])
            patch.set_alpha(alpha[amplitudes.index(l)])

        for i, whisker in enumerate(bplot['whiskers']):
            # if not (args.participant == 't16' and l == 'MIME'):
            box_index = i // 2  # two whiskers per box
            whisker.set_color(target_color[words.index(word)])
            whisker.set_linewidth(linewidth[box_index])

        for i, cap in enumerate(bplot['caps']):
            # if not (args.participant == 't16' and l == 'MIME'):
            box_index = i // 2  # two caps per box
            cap.set_color(target_color[words.index(word)])
            cap.set_linewidth(linewidth[box_index])

        for i, median in enumerate(bplot['medians']):
            # if not (args.participant == 't16' and l == 'MIME'):
            box_index = i  # one median per box
            median.set_color(target_color[words.index(word)])
            median.set_linewidth(linewidth[box_index])   
        
        for i, flier in enumerate(bplot['fliers']):
            # print(i, target_color[words.index(word)], alpha[i])
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
         

if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['cue', 'speech_onsets', 'speech_offsets'], help = 'keys to load from rdbmat files')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if args.participant == 't15':
        alpha = [0.25, 0.5, 0.75, 1]
        linewidth = [1.5, 2.5, 3.5, 4.5]
    elif args.participant == 't16':
        alpha = [0, 0.5, 0.75, 1] # make alpha for MIME 0
        linewidth = [0, 2.5, 3.5, 4.5] # make linewidth for MIME 0

    if not os.path.exists(args.savepath_data):
        os.makedirs(args.savepath_data, exist_ok=True)
    
    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)

    print('Running speech_duration.py')
    print(args)

    # load data
    data = load_rdbmat(args.participant, args.session, args.required_keys)

    # compute speech duration
    print('Computing speech duration ...')
    speech_duration_stats, min_duration = compute_speech_duration(data)
    print('Min duration across all word-loudness trials:', min_duration) # t15 = 306 ms. t16 = 155 ms

    # save speech duration stats
    with open(f'{args.savepath_data}/speech_duration_stats_{args.participant}_{args.session}_{formatted_datetime}.pkl', 'wb') as f:
        pkl.dump(speech_duration_stats, f)

    # plot duration stats
    plot_speech_duration(speech_duration_stats)
