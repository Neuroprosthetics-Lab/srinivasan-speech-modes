from sklearn.decomposition import PCA
import scipy
import numpy as np
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import pickle as pkl
from sklearn.preprocessing import StandardScaler

'''
Example cmd (when run from this directory; provide python script path appropriately if run from different directory):
For t15,
    python pca.py --participant t15 --session word-loudness --nbins_before_onset 75 --nbins_after_onset 75 --savepath_data ../analyses_figures_data/t15/word-loudness/pca/ --savepath_fig ../analyses_figures/t15/word-loudness/pca/
For t16,
    python pca.py --participant t16 --session word-loudness --nbins_before_onset 75 --nbins_after_onset 75 --savepath_data ../analyses_figures_data/t16/word-loudness/pca/ --savepath_fig ../analyses_figures/t16/word-loudness/pca/


Neural data will be loaded from ../analyses_data/{participant}_{session}/.
Results obtained from this script will be saved in savepath_data.
Figures generated from this script will be saved in savepath_fig.
'''

#---------------------------------------------------
# global variables
#---------------------------------------------------
pre_delay_nbins_in_spikepow = 100
bin_size_ms = 10
n_channels = {
     't15': 256,
     't16': 128
}
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going'] 

amplitude_color = [
    (167/255, 185/255, 207/255),
    (114/255, 159/255, 207/255),
    (53/255, 126/255, 221/255),
    (0, 79/255, 158/255),
]
word_color = [
    (0.9254902, 0.12156863, 0.14117647),
    (0.98431373, 0.72941176, 0.07058824),
    # [0.57254902, 0.78431373, 0.24313725],  
    (0.384, 0.682, 0.2),
    (0.43137255, 0.79607843, 0.85490196),
    # (0.26529412, 0.40686275, 0.72490196),
    [0.45568627, 0.31764706, 0.63529412],  
    (0.84705882, 0.2627451, 0.59215686)
]
fontsize = 40
scatter_size = 3000

#--------------------------------------------
# functions
#--------------------------------------------
def load_data():

    # load data
    data_path = f'../analyses_data/{args.participant}_{args.session}/'
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
                for key in args.required_keys:
                    data[key] = data_temp_required[key]
            else:
                for key in args.required_keys:
                    data[key] = np.append(data[key], data_temp_required[key], axis = -1)

    data['cue'] = np.squeeze(data['cue']) # squeeze cue numpy array shape

    print('Data loaded ...')
    for key in data:
        print(key, data[key].shape)
    
    return data


def preprocess_data(data):
    # load spikepow around speech onset, get average spike for word-amp combination
    n_trials = len(data['cue'])
    word_amp_trial_avg = []
    word_amp_trial_avg_thx = []
    word_label = []
    amp_label = []
    for word in words:
        for amp in amplitudes:
            print(word, amp)
            inds = [k for k in range(n_trials) if amp in data['cue'][k] and word in data['cue'][k]]
            
            x_spikepow = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels[args.participant]))
            x_threshcross = np.empty((0, args.nbins_before_onset + args.nbins_after_onset, n_channels[args.participant]))
            for k in inds:
                    spikepow = np.squeeze(data['spikepow'])[k]
                    delay_duration_ms = np.squeeze(data['delay_duration_ms'])[k]
                    binned_delay_duration = int(np.squeeze(delay_duration_ms) / bin_size_ms)
                    spikepow = spikepow[pre_delay_nbins_in_spikepow + binned_delay_duration:, :] # extract from delay period


                    start_ind = np.squeeze(data['speech_onsets'])[k].squeeze()
                    end_ind = np.squeeze(data['speech_offsets'])[k].squeeze()
                    start_ind = math.floor((start_ind/30000) * (1000/10)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index
                    end_ind = math.ceil((end_ind/30000) * (1000/10)) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index

                    temp_spikepow = spikepow[start_ind - args.nbins_before_onset: start_ind + args.nbins_after_onset, :n_channels[args.participant]] 
                    if temp_spikepow.shape[0] == args.nbins_before_onset + args.nbins_after_onset:
                            # add spikepow and threshcross around speech onset; shape (time_bins x 256)
                            x_spikepow = np.append(x_spikepow, np.expand_dims(temp_spikepow, 0), axis = 0)

            print(x_spikepow.shape)
            
            if x_spikepow.shape[0] != 0:
                    x_spikepow_mean = np.mean(x_spikepow, axis = 0) # avg across trials
                    assert not np.isnan(np.sum(x_spikepow_mean))
                    word_amp_trial_avg.append(x_spikepow_mean)
                    word_label.append(words.index(word))
                    amp_label.append(amplitudes.index(amp))

    amp_word_trial_avg = np.array(word_amp_trial_avg)

    return amp_word_trial_avg, word_label, amp_label

def compute_pca(amp_word_trial_avg):
    # compute pca all components, but take only 3 for plotting
    pca = PCA(n_components=3)

    # no averaging across time; n_samples == n_conditions x time (25 x 50), n_features == n_channels (256)
    X = amp_word_trial_avg.reshape(-1, n_channels[args.participant])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_pca = pca.fit_transform(X_scaled) 
    x_recon = pca.inverse_transform(x_pca)

    # reshape it back to n_trials x time x n_pca_components
    x_pca = x_pca.reshape(amp_word_trial_avg.shape[0], -1, 3)
    x_recon = x_recon.reshape(amp_word_trial_avg.shape[0], -1, n_channels[args.participant])

    # averge across time
    x_pca = np.mean(x_pca, axis=1)
    x_ax = x_pca[:, 0]
    y_ax = x_pca[:, 1]
    z_ax = x_pca[:, 2]

    # explained variance
    x_fitted = pca.fit(X_scaled)
    print('PCA explained variance ratio:', x_fitted.explained_variance_ratio_)

    with open(f'{args.savepath_data}{args.participant}_{args.session}_{args.nbins_before_onset}_{args.nbins_after_onset}_pca_weights.pkl', 'wb') as f:
        pkl.dump({'pca_weights': pca.components_}, f)

    return x_ax, y_ax, z_ax, x_fitted.explained_variance_ratio_


def plot_pca_projections(x_ax, y_ax, z_ax, expl_var, amp_label, word_label):
     
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_ax, y_ax, z_ax, color=np.array(amplitude_color)[amp_label], s=scatter_size, alpha = 0.8)

    ax.set_xlabel(f'PC 1 ({expl_var[0]*100:.1f}%)', fontsize = fontsize)
    ax.set_ylabel(f'PC 2 ({expl_var[1]*100:.1f}%)', fontsize = fontsize)
    ax.set_zlabel(f'PC 3 ({expl_var[2]*100:.1f}%)', fontsize = fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if args.participant == 't15':
        ax.view_init(elev = -140, azim = 120, roll = 0)
    elif args.participant == 't16':
        ax.view_init(elev = 70, azim = -120, roll = 8) 
    ax.grid(False)

    fig.tight_layout()
    # save figure
    plt.savefig(f'{args.savepath_fig}pca_loudness_{formatted_datetime}.png', format='png')

    # word based color coding
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_ax, y_ax, z_ax, color=np.array(word_color)[word_label], s=scatter_size, alpha = 0.8)
    ax.set_xlabel('PC 1', fontsize = fontsize)
    ax.set_ylabel('PC 2', fontsize = fontsize)
    ax.set_zlabel('PC 3', fontsize = fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if args.participant == 't15':
        ax.view_init(elev = -140, azim = 120, roll = 0)
    elif args.participant == 't16':
        ax.view_init(elev = 70, azim = -120, roll = 8) 
    ax.grid(False)

    fig.tight_layout()
    # save figure
    plt.savefig(f'{args.savepath_fig}pca_word_{formatted_datetime}.png', format='png')

    return


if __name__ == "__main__":
     
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--required_keys', type=list, default=['spikepow', 'cue', 'delay_duration_ms', 'speech_onsets', 'speech_offsets'], help = 'keys to load from rdbmat files')
    parser.add_argument('--nbins_before_onset', type=int, default=75, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default=75, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_fig):
        os.makedirs(args.savepath_fig, exist_ok=True)
    if not os.path.exists(args.savepath_data):
        os.makedirs(args.savepath_data, exist_ok=True)

    # load data
    print('Loading data...')
    data = load_data()

    # preprocess data
    print('Preprocessing data...')
    amp_word_trial_avg, word_label, amp_label = preprocess_data(data)

    # compute pca
    print('Computing PCA...')
    x_ax, y_ax, z_ax, expl_var = compute_pca(amp_word_trial_avg)

    # plot pca projections
    print('Plotting PCA projections...')
    plot_pca_projections(x_ax, y_ax, z_ax, expl_var, amp_label, word_label)

    print('DONE!')