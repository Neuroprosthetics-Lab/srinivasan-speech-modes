# This script plots the spike sorting results: single-unit psths, waveform and tuning to different task parameters
import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime

'''
Example cmd:
For t15,
    python single_unit_psth_tuning.py --participant t15 --session word-loudness --nbins_before_onset 100 --nbins_after_onset 50 --nbins_before_offset 25 --nbins_after_offset 25 --plot_speech_offset --savepath_data ../plotting_data/t15/word-loudness/single_unit_neurons/ --savepath_fig ../plotting_figures/t15/word-loudness/single_unit_neurons/
For t16,
    python single_unit_psth_tuning.py --participant t16 --session word-loudness --nbins_before_onset 100 --nbins_after_onset 50 --nbins_before_offset 25 --nbins_after_offset 30 --plot_speech_offset --savepath_data ../plotting_data/t16/word-loudness/single_unit_neurons/ --savepath_fig ../plotting_figures/t16/word-loudness/single_unit_neurons/

'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
required_binned_delay_duration = 100 # required delay duration bins for plotting
required_binned_delay_duration_before_cue = 10

# plotting
my_color = [
    (167/255, 185/255, 207/255),
    (114/255, 159/255, 207/255),
    (53/255, 126/255, 221/255),
    (0, 79/255, 158/255),
]
fontsize = 12 # general for all psth plots; not for select channels
scatter_size = 30
delay_duration = 100
linewidth = 2

arrays = {
    't15': ['M1', 'v6v','d6v','55b'],
    't16': ['55b/PEF', '6v', 'HK1', 'HK2'],
}

#--------------------------------------------
# functions
#--------------------------------------------
def plot_psth_and_waveform_given_neuron(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, avg_end_thx, sem_end_thx, spike_unit_ids, spike_channels, waveforms, plt_neuron):

    num_subplots = 2
    width_ratios = [required_binned_delay_duration_before_cue + required_binned_delay_duration, args.nbins_before_onset + args.nbins_after_onset]
    if args.plot_speech_offset:
        num_subplots = 3
        width_ratios.extend([args.nbins_before_offset + args.nbins_after_offset])

    # plot one example electrode per array
    if args.participant in ['t15', 't19']:
        fontsize = 14
        if args.plot_speech_offset:
            fontsize = 12
        fig, ax = plt.subplots(len(plt_neuron), num_subplots, figsize = (6, 8), 
                            gridspec_kw={'width_ratios': width_ratios}) # 'width_ratios': [n_time_bins_plotted in left subplot, n_time_bins plotted in right subplot]
    
    elif args.participant == 't16':
        fontsize = 14
        if args.plot_speech_offset:
            fontsize = 13
        fig, ax = plt.subplots(len(plt_neuron), num_subplots, figsize = (6, 4), # (6, 4) originally
                            gridspec_kw={'width_ratios': width_ratios}) # 'width_ratios': [n_time_bins_plotted in left subplot, n_time_bins plotted in right subplot]
    
    for subplot in range(len(plt_neuron)):
        # delay period
        for a in range(len(amplitudes)):
            ax[subplot, 0].plot(avg_delay_thx[a, :, plt_neuron[subplot]].T, color = my_color[a], linewidth = linewidth)
            ax[subplot, 0].fill_between(np.arange(required_binned_delay_duration_before_cue + required_binned_delay_duration),
                                avg_delay_thx[a, :, plt_neuron[subplot]].T - sem_delay_thx[a, :, plt_neuron[subplot]].T,
                                avg_delay_thx[a, :, plt_neuron[subplot]].T + sem_delay_thx[a, :, plt_neuron[subplot]].T,
                                alpha = 0.5, label='_hidden', color = my_color[a], linewidth = linewidth)
        
        ax[subplot, 0].set_ylim([0, 110])
        if subplot == 0:
            ax[subplot, 0].set_yticks([0, 110], [0, 110], fontsize = fontsize)
            ax[subplot, 0].set_ylabel('Firing rate (Hz)', fontsize = fontsize)
        else:
            ax[subplot, 0].set_yticks([])
        if args.participant != 't16':
            # ax[subplot, 0].set_xticks([])
            ax[subplot, 0].set_xticks(np.arange(10, required_binned_delay_duration + 11, 50), [k*10 if k != 0 else '' for k in (np.arange(10, required_binned_delay_duration + 11, 50) - 10)], fontsize = fontsize, rotation = 90)
            if subplot != len(plt_neuron) - 1:
                ax[subplot, 0].tick_params(labelbottom = False)
        elif args.participant == 't16':
            ax[subplot, 0].set_xticks(np.arange(10, required_binned_delay_duration + 11, 50), [k*10 if k != 0 else '' for k in (np.arange(10, required_binned_delay_duration + 11, 50) - 10)], fontsize = fontsize, rotation = 90)
            if subplot != len(plt_neuron) - 1:
                ax[subplot, 0].tick_params(labelbottom = False)
        if args.participant != 't16':
            for pos in ['right', 'top', 'bottom']: 
                ax[subplot, 0].spines[pos].set_visible(False) 
            if subplot == len(plt_neuron) - 1:
                ax[subplot, 0].spines['bottom'].set_visible(True)
        elif args.participant == 't16':
            for pos in ['right', 'top', 'bottom']: 
                ax[subplot, 0].spines[pos].set_visible(False) 
            if subplot == len(plt_neuron) - 1:
                ax[subplot, 0].spines['bottom'].set_visible(True)

        ax[subplot, 0].scatter(required_binned_delay_duration_before_cue, 3.5, color='black', s=scatter_size)
        if subplot == len(plt_neuron) - 1:
            if args.participant == 't15':
                ax[subplot, 0].text(required_binned_delay_duration_before_cue, - 12, 'Cue', color='black', ha = 'center', fontsize = fontsize)
            elif args.participant == 't16':
                ax[subplot, 0].text(required_binned_delay_duration_before_cue, - 16, 'Cue', color='black', ha = 'center', fontsize = fontsize)

        # go period
        for a in range(len(amplitudes)):
            ax[subplot, 1].plot(avg_go_thx[a, :, plt_neuron[subplot]].T, label = amplitudes[a], color = my_color[a], linewidth = linewidth)
            ax[subplot, 1].fill_between(np.arange(args.nbins_before_onset + args.nbins_after_onset),
                                avg_go_thx[a, :, plt_neuron[subplot]].T - sem_go_thx[a, :, plt_neuron[subplot]].T,
                                avg_go_thx[a, :, plt_neuron[subplot]].T + sem_go_thx[a, :, plt_neuron[subplot]].T,
                                alpha = 0.5, label = '_hidden', color = my_color[a], linewidth = linewidth)

        ax[subplot, 1].set_ylim([0, 110])
        ax[subplot, 1].set_yticks([])
        if args.participant != 't16':
            for pos in ['right', 'top', 'left', 'bottom']: 
                ax[subplot, 1].spines[pos].set_visible(False)
            if subplot == len(plt_neuron) - 1:
                ax[subplot, 1].spines['bottom'].set_visible(True) 
        elif args.participant == 't16':
            for pos in ['right', 'top', 'left', 'bottom']: 
                ax[subplot, 1].spines[pos].set_visible(False)
            if subplot == len(plt_neuron) - 1:
                ax[subplot, 1].spines['bottom'].set_visible(True)
        if args.participant != 't16':
            # ax[subplot, 1].set_xticks([])
            ax[subplot, 1].set_xticks(np.arange(0, avg_go_thx.shape[1] + 1, 50), [k*10 if k != 0 else '' for k in (np.arange(0, avg_go_thx.shape[1] + 1, 50) - args.nbins_before_onset)], fontsize = fontsize, rotation = 90)
            if subplot != len(plt_neuron) - 1:
                ax[subplot, 1].tick_params(labelbottom = False)
        elif args.participant == 't16':
            ax[subplot, 1].set_xticks(np.arange(0, avg_go_thx.shape[1] + 1, 50), [k*10 if k != 0 else '' for k in (np.arange(0, avg_go_thx.shape[1] + 1, 50) - args.nbins_before_onset)], fontsize = fontsize, rotation = 90)
            if subplot != len(plt_neuron) - 1:
                ax[subplot, 1].tick_params(labelbottom = False)
                

        ax[subplot, 1].scatter(args.nbins_before_onset, 3.5, color='black', s=scatter_size)
        if subplot == len(plt_neuron) - 1:
            if args.participant == 't15':
                ax[subplot, 1].text(args.nbins_before_onset, -25, 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
                ax[subplot, 1].set_xlabel('Time (ms)', fontsize = fontsize)
            elif args.participant == 't16':
                ax[subplot, 1].text(args.nbins_before_onset, -29, 'Speech\nonset', color='black', ha='center', fontsize = fontsize)
                ax[subplot, 1].set_xlabel('Time (ms)', fontsize = fontsize)
            elif args.participant == 't19':
                ax[subplot, 1].text(args.nbins_before_onset, -12, 'Go', color='black', ha='center', fontsize = fontsize)
            
        if not args.plot_speech_offset:
            ax[subplot, 1].hlines(0, xmin = len(avg_go_thx[a, :, plt_neuron[subplot]]) - 50, xmax = len(avg_go_thx[a, :, plt_neuron[subplot]]), linewidth = 4, color = 'black')
            if subplot == len(plt_neuron) - 1:
                if args.participant == 't15':
                    ax[subplot, 1].text(len(avg_go_thx[a, :, plt_neuron[subplot]]) - 25, -14, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
                elif args.participant == 't16':
                    ax[subplot, 1].text(len(avg_go_thx[a, :, plt_neuron[subplot]]) - 25, -14, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
                elif args.participant == 't19':
                    ax[subplot, 1].text(len(avg_go_thx[a, :, plt_neuron[subplot]]) - 25, -12, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
        elif args.plot_speech_offset:
            # plot the 3rd column of psth w.r.t. speech offset, add speech offset and 500 ms marker
            # psth
            for a in range(len(amplitudes)):
                ax[subplot, 2].plot(avg_end_thx[a, :, plt_neuron[subplot]].T, label = amplitudes[a], color = my_color[a], linewidth = linewidth)
                ax[subplot, 2].fill_between(np.arange(args.nbins_before_offset + args.nbins_after_offset),
                                    avg_end_thx[a, :, plt_neuron[subplot]].T - sem_end_thx[a, :, plt_neuron[subplot]].T,
                                    avg_end_thx[a, :, plt_neuron[subplot]].T + sem_end_thx[a, :, plt_neuron[subplot]].T,
                                    alpha = 0.5, label = '_hidden', color = my_color[a], linewidth = linewidth)

            ax[subplot, 2].set_ylim([0, 110])
            if args.participant != 't16':
                for pos in ['right', 'top', 'left', 'bottom']: 
                    ax[subplot, 2].spines[pos].set_visible(False) 
                if subplot == len(plt_neuron) - 1:
                    ax[subplot, 2].spines['bottom'].set_visible(True)
            elif args.participant == 't16':
                for pos in ['right', 'top', 'left', 'bottom']: 
                    ax[subplot, 2].spines[pos].set_visible(False) 
                if subplot == len(plt_neuron) - 1:
                    ax[subplot, 2].spines['bottom'].set_visible(True)
            ax[subplot, 2].set_yticks([])
            if args.participant != 't16':
                # ax[subplot, 2].set_xticks([])
                ax[subplot, 2].set_xticks(np.arange(0, avg_end_thx.shape[1], 25), [k*10 if k != 0 else '' for k in (np.arange(0, avg_end_thx.shape[1], 25) - args.nbins_before_offset)], fontsize = fontsize, rotation = 90) # ms labels
                if subplot != len(plt_neuron) - 1:
                    ax[subplot, 2].tick_params(labelbottom = False)
            elif args.participant == 't16':
                ax[subplot, 2].set_xticks(np.arange(0, avg_end_thx.shape[1], 25), [k*10 if k != 0 else '' for k in (np.arange(0, avg_end_thx.shape[1], 25) - args.nbins_before_offset)], fontsize = fontsize, rotation = 90) # ms labels
                if subplot != len(plt_neuron) - 1:
                    ax[subplot, 2].tick_params(labelbottom = False)

            # speech offset marker
            ax[subplot, 2].scatter(args.nbins_before_offset, 3.5, color='black', s=scatter_size)
            if subplot == len(plt_neuron) - 1:
                if args.participant == 't15':
                    ax[subplot, 2].text(args.nbins_before_offset, -25, 'Speech\noffset', color='black', ha='center', fontsize = fontsize)
                elif args.participant == 't16':
                    ax[subplot, 2].text(args.nbins_before_offset, -29, 'Speech\noffset', color='black', ha='center', fontsize = fontsize)
                elif args.participant == 't19':
                    ax[subplot, 2].text(args.nbins_before_offset, -12, 'End', color='black', ha='center', fontsize = fontsize)
            
        # electrode number
        ax[subplot, 1].text(-130, 70, f'Electrode {spike_channels[plt_neuron[subplot]] + 1} (array {arrays[args.participant][math.floor(spike_channels[plt_neuron[subplot]]/64)]})', color = 'darkorange', fontsize = fontsize)

    # legend
        legend_x_loc = args.nbins_before_onset + args.nbins_after_onset
        if args.plot_speech_offset:
            legend_x_loc = args.nbins_before_offset + args.nbins_after_offset
        if args.participant == 't15':
            for a in range(4):
                plt.text(legend_x_loc, 90 - a * 15, amplitudes[a], ha = "right", fontsize = fontsize, color = my_color[a])
        elif args.participant == 't16':
            for a in range(4):
                plt.text(legend_x_loc, 230 - a * 15, amplitudes[a], ha = "right", fontsize = fontsize, color = my_color[a])
        elif args.participant == 't19':
            for a in range(4):
                plt.text(legend_x_loc, 100 - a * 15, amplitudes[a], ha = "right", fontsize = fontsize, color = my_color[a])

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05) 
    plt.subplots_adjust(hspace=0.1)
    # plt.show()
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_psth_neurons.png', format='png')

    # plot waveforms
    # plot 1ms before 1 ms after spike
    x_samples = 90 # 1ms before to 2 ms after spike is fetched by default (90 samples)
    for unit_idx in plt_neuron:
        unit_id = spike_unit_ids[unit_idx]
        ch = spike_channels[unit_idx]
        plt.figure(figsize = (1.5,1.5))
        curr_waveform = waveforms[ch][unit_id][:x_samples]
        plt.plot(curr_waveform,linewidth = 4, color = 'black') # plot 1ms before to 1 ms after spike
        for pos in ['top', 'bottom', 'right', 'left']:
            plt.gca().spines[pos].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        plt.vlines(-1, min(curr_waveform), min(curr_waveform) + 10, linewidth = 2, color = 'black')
        plt.hlines(max(curr_waveform) + 3, len(curr_waveform) - 30, len(curr_waveform), linewidth = 2, color = 'black')
        plt.text(len(curr_waveform) - 15, max(curr_waveform) + 5, '1 ms', fontsize = fontsize - 2)
        plt.text(-6, min(curr_waveform), '10 μV', rotation = 90, fontsize = fontsize - 2)
        plt.title(f'Neuron_ID {unit_id}', fontsize = fontsize)
        # plt.show()
        plt.savefig(f'{args.savepath_fig}{args.participant}_electrode_{ch+1}_neuron_id_{unit_id}.png', format='png')

    return


def plot_neuron_tuning(data):

    fontsize = 14
    my_color = [ # ordered similar to marg names
    (34/255, 54/255, 150/255), # loudness, blue
    (150/255, 54/255, 34/255), # word, red
    (150/255, 34/255, 150/255), # both word and loudness, purple
    (128/255, 128/255, 128/255), # neither, gray
    ]
    mylabels = ['Only\nloudness', 'Only\nwords', 'Both words\nand loudness', 'Neither']
    n_neurons = len(data['only_loud_units'] + data['only_word_units'] + data['both_loud_word_units'] + data['neither_loud_word_units'])
    y = np.array((len(data['only_loud_units'])/n_neurons, len(data['only_word_units'])/n_neurons, len(data['both_loud_word_units'])/n_neurons, len(data['neither_loud_word_units'])/n_neurons))
    plt.figure(figsize=(5,4))
    plt.pie(y, labels = mylabels, autopct='%1.1f%%', colors=my_color, textprops={'fontsize': fontsize + 1}, shadow = False, startangle = 0, wedgeprops={'alpha': 0.8})
    plt.title(f'{args.participant.upper()}\n\nTotal number of neurons: {n_neurons}\n% of neurons tuned to\n', fontsize = fontsize + 3)
    # plt.show()
    plt.savefig(f'{args.savepath_fig}{args.participant}_neurons_tuning_pie_chart.png', format='png')

    return
        


if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--nbins_before_onset', type=int, default=None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=None, help = 'number of bins after speech onset')
    parser.add_argument('--nbins_before_offset', type=int, default=None, help='number of bins before speech offset')
    parser.add_argument('--nbins_after_offset', type=int, default=None, help='number of bins after speech offset')
    parser.add_argument('--plot_speech_offset', action='store_true', help='whether to plot psth around speech offset; default psth around cue and speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)

    print('Running single_unit_psth_tuning.py')
    print(args)

    # load single-unit waveforms
    print('Loading single-unit waveforms...')
    with open(f'{args.savepath_data}{args.participant}_unit_waveform.pkl', 'rb') as f:
        waveforms = pkl.load(f) # key is ch, value is a dict of key with unit_id and value as waveform mean

    # load single unit psth
    print('Loading single-unit psths...')
    with open(f'{args.savepath_data}{args.participant}_unit_psth.pkl', 'rb') as f:
        data = pkl.load(f)
        avg_go_thx = data['avg_go_thx']
        sem_go_thx = data['sem_go_thx']
        avg_delay_thx = data['avg_delay_thx']
        sem_delay_thx = data['sem_delay_thx']
        avg_end_thx = data['avg_end_thx']
        sem_end_thx = data['sem_end_thx']
        spike_unit_ids = data['spike_unit_ids']
        spike_channels = data['spike_channels']

    # plot particular channel psth
    print('Plotting particular neurons ...')
    plt_neuron = {
        't15': [180, 131, 46, 76], # these are neuron indices (0-indexed), ordered according to implanted arrays
        't16': [54, 84], # these are neuron indices (0-indexed), ordered according to implanted arrays
    }

    plot_psth_and_waveform_given_neuron(avg_go_thx, sem_go_thx, avg_delay_thx, sem_delay_thx, avg_end_thx, sem_end_thx, spike_unit_ids, spike_channels, waveforms, plt_neuron[args.participant]) # 0-indexed channel

    # plot neuron tuning pie chart
    print('Loading single-unit word and loudness tuning...')
    with open(f'{args.savepath_data}{args.participant}_unit_tuning.pkl', 'rb') as f:
        data = pkl.load(f)
        
    plot_neuron_tuning(data)
