import argparse
import numpy as np
import scipy
import math
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

'''
Example run command:
python plot_dPCA_results.py --participant <> --session <> --nbins_before_onset 75 --nbins_after_onset 75
'''

#-----------------------
# global variables
#-----------------------
amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD']
words = ['be', 'my', 'know', 'do', 'have', 'going']

marg_names = ['Word', 'Loudness', 'Condition\nindependent', 'Interaction']
num_pcs_plot = 3
time = np.arange(-0.75, 0.75, 0.01)

# plotting
marg_plotting_order = ['Condition\nindependent', 'Word', 'Loudness', 'Interaction']
marg_plotting_order_ind = [2, 0, 1, 3]
target_color = [
    (0.9254902, 0.12156863, 0.14117647),
    (0.98431373, 0.72941176, 0.07058824),
    # [0.57254902, 0.78431373, 0.24313725],  # Commented out in MATLAB
    (0.384, 0.682, 0.2),
    (0.43137255, 0.79607843, 0.85490196),
    # (0.26529412, 0.40686275, 0.72490196),
    [0.45568627, 0.31764706, 0.63529412],  # Commented out in MATLAB
    (0.84705882, 0.2627451, 0.59215686)
]
hue = [0.25, 0.5, 0.75, 1]
linewidth = [0.5, 1, 1.5, 2]
scatter_size = 30
fontsize = 15
my_explode = [0.1] * len(marg_names)
my_color = [ # ordered similar to marg names
    (150/255, 54/255, 34/255), # word, red
    (34/255, 54/255, 150/255), # loudness, blue
    (128/255, 128/255, 128/255), # condition independent, gray
    (150/255, 34/255, 150/255), # interaction, purple
]

#-----------------------
# functions
#-----------------------
def plot_projections(projections, which_marg, explained_var):

    fig, ax = plt.subplots(len(marg_names), num_pcs_plot, figsize = (12,6))
    for marg in range(len(marg_names)):
        
        marg_ind_to_plot = marg_plotting_order_ind[marg]
        components = [i for i, val in enumerate(which_marg.squeeze()) if val == marg_ind_to_plot]
        comp_min = 1000; comp_max = -1
        
        for comp in range(min(num_pcs_plot, len(components))):
            for d1 in range(len(words)):
                for d2 in range(len(amplitudes)):

                    # Plot the data
                    ax[marg, comp].plot(
                        time,
                        np.squeeze(projections[components[comp], d1, d2, :]),
                        linestyle = '-',
                        color=target_color[d1],
                        alpha = hue[d2],
                        linewidth = linewidth[d2],
                        label=f"{amplitudes[d2]} {words[d1]}"
                    )

                    comp_min = min(comp_min, np.min(np.squeeze(projections[components[comp], d1, d2, :])))
                    comp_max = max(comp_max, np.max(np.squeeze(projections[components[comp], d1, d2, :])))

            print(marg_ind_to_plot, comp, comp_min, comp_max, components[comp], explained_var[marg_ind_to_plot,components[comp]])
            
            ax[marg, comp].set_ylim([math.floor(comp_min), math.ceil(comp_max)])
            ax[marg, comp].scatter(0, math.floor(comp_min) + (math.ceil(comp_max) * 2/20), color='black', s=scatter_size)
            ax[marg, comp].text(0.7, math.ceil(comp_max), f"{explained_var[marg_ind_to_plot,components[comp]]:.1f}%", fontsize = fontsize, ha = 'center')
            ax[marg, comp].text(-0.65, math.floor(comp_min) + 0.07 * (math.ceil(comp_max) - math.floor(comp_min)), f"{components[comp] + 1}", fontsize = fontsize, ha = 'center')
            ax[marg, comp].scatter(-0.65, math.floor(comp_min) + 0.16 * (math.ceil(comp_max) - math.floor(comp_min)), s = 400, color = 'black', facecolor = 'none')
            
            if comp != 0:
                for pos in ['right', 'top', 'bottom']: 
                    ax[marg, comp].spines[pos].set_visible(False) 
                ax[marg, comp].set_xticks([])
                ax[marg, comp].set_yticks([])

            if comp == 0:
                for pos in ['right', 'top', 'bottom',]: 
                    ax[marg, comp].spines[pos].set_visible(False)
                    # ax[marg, comp].set_yticks([-math.ceil(comp_max), 0, math.ceil(comp_max)],
                    #                         [-math.ceil(comp_max), 0, math.ceil(comp_max)], fontsize = fontsize)
                    ax[marg, comp].set_yticks([math.floor(comp_min), 0, math.ceil(comp_max)],
                                            [math.floor(comp_min), 0, math.ceil(comp_max)], fontsize = fontsize)
                    ax[marg, comp].set_xticks([])
                    ax[marg, comp].set_ylabel(marg_names[marg_ind_to_plot], fontsize = fontsize)

            if marg == 3 and comp == 2:
                ax[marg, comp].hlines(-math.ceil(comp_max), xmin = 0.25, xmax = 0.75, linewidth = 3, color = 'black')
                ax[marg, comp].text(0.5, -1.5, '500 ms', color = 'black', ha = 'center', fontsize = fontsize)
            
            if marg == 3 and comp == 0:
                ax[marg, comp].text(0, -1.5, 'Speech onset', color = 'black', ha = 'center', fontsize = fontsize)

            if marg == 0 and comp == 0:
                ax[marg, comp].text(0.7, math.ceil(comp_max) + 3, 'Explained variance', color = 'black', ha = 'center', fontsize = fontsize)
                # ax[marg, comp].text(-0.65, math.floor(comp_min) - 0.4 * (math.ceil(comp_max) - math.floor(comp_min)), 'Component', color = 'black', ha = 'center', fontsize = fontsize)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05) 
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_dpca_projections_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_dpca_projections_{formatted_datetime}.png', format='png')

    return


def plot_explained_variance(explained_var):

    fontsize = 13
    
    fig = plt.figure(figsize=(5,5))
    plt.subplot(2,1,1)

    # pie chart
    expl_var_comp = np.sum(explained_var, axis = -1)
    marg_names_with_percent = []
    for i in range(len(marg_names)):
        marg_names_with_percent.append(f'{marg_names[i]}\n({(expl_var_comp[i]/sum(expl_var_comp))*100:.1f}%)')
    print(marg_names_with_percent)
    plt.pie(expl_var_comp, labels = marg_names_with_percent, #autopct='%1.1f%%',
            colors = my_color, explode = my_explode, startangle = 140, shadow = False, textprops={'fontsize': fontsize}, labeldistance=1.3)

    # stacked bar plot
    plt.subplot(2, 1, 2)
    num_comps_to_plot = 15
    bottom = np.zeros(num_comps_to_plot)  # Initialize bottom for stacking
    for i in range(len(marg_names)):
        plt.bar(np.arange(num_comps_to_plot), explained_var[i, :num_comps_to_plot], label=marg_names[i], color=my_color[i], bottom=bottom)
        bottom += explained_var[i, :num_comps_to_plot]
    
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    plt.ylim([0, math.ceil(bottom[0])])
    plt.yticks(np.arange(0, math.ceil(bottom[0]), 10), np.arange(0, math.ceil(bottom[0]), 10), fontsize = fontsize)
    plt.xticks([0, 4, 9, 14], [1, 5, 10, 15], fontsize = fontsize)
    plt.xlabel('dPCA component', fontsize = fontsize)
    plt.ylabel('Variance (%)', fontsize = fontsize)

    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_dpca_variance_plot_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_dpca_variance_plot_{formatted_datetime}.png', format='png')

    return


def plot_correlation(corr_matrix, sig_ind):
    
    fontsize = 19
    fig = plt.figure(figsize = (5,5))
    num_comps_to_plot = 15
    custom_cmap = LinearSegmentedColormap.from_list("dark_bwr", ["#002147", "white", "#800000"])  # Dark navy blue → White → Maroon
    img = plt.imshow(corr_matrix[:num_comps_to_plot, :num_comps_to_plot], cmap = custom_cmap, vmin = -1, vmax = 1)
    ind_to_plot = []
    for k in range(len(sig_ind[0])):
        if sig_ind[0][k] < num_comps_to_plot and sig_ind[1][k] < num_comps_to_plot:
            ind_to_plot.append(k)
    plt.scatter(sig_ind[1][ind_to_plot], sig_ind[0][ind_to_plot], color='k', marker='*')
    plt.xticks([0, 4, 9, 14], [1, 5, 10, 15], fontsize = fontsize)
    plt.yticks([0, 4, 9, 14], [1, 5, 10, 15], fontsize = fontsize)
    plt.xlabel('dPCA component', fontsize = fontsize)
    plt.ylabel('dPCA component', fontsize = fontsize)
    cbar = fig.colorbar(img, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_ticks([-1, 0, 1])
    for pos in ['right', 'top', 'left', 'bottom']: 
        plt.gca().spines[pos].set_visible(False)
    
    fig.tight_layout()
    # plt.show()

    # save figure
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_dpca_correlation_matrix_{formatted_datetime}.svg', format='svg', dpi=1200)
    plt.savefig(f'{args.savepath_fig}{args.participant}_{args.session}_dpca_correlation_matrix_{formatted_datetime}.png', format='png')

    return

if __name__ == "__main__":

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=str, default=None, help='participant id')
    parser.add_argument('--session', type=str, default=None, help = 'session id')
    parser.add_argument('--nbins_before_onset', type=int, default = None, help = 'number of bins before speech onset')
    parser.add_argument('--nbins_after_onset', type=int, default = None, help = 'number of bins after speech onset')
    parser.add_argument('--savepath_data', type=str, default='../figures_data/', help = 'path to save processed data from this script')
    parser.add_argument('--savepath_fig', type=str, default='../figures/', help = 'path to save figures from this script')
    args = parser.parse_args()

    if not os.path.exists(args.savepath_fig):
        os.mkdir(args.savepath_fig)

    if args.participant == 't15':
        n_channels = 256
    elif args.participant == 't16':
        n_channels = 128

    print('Running plot_dPCA_results.py')
    print(args)

    # load projections
    projections = scipy.io.loadmat(f'{args.savepath_data}{args.participant}_{args.session}_{args.nbins_before_onset}_{args.nbins_after_onset}_avg_firingrates_projections.mat')
    projections = projections['ProjectionAverage']
    print('Projections shape', projections.shape)

    # load which margins the projections belong to
    which_marg = scipy.io.loadmat(f'{args.savepath_data}{args.participant}_{args.session}_{args.nbins_before_onset}_{args.nbins_after_onset}_which_marg.mat')
    which_marg = which_marg['whichMarg'] - 1 # subtract 1 as matlab indices are 1-indexed
    print('Which marginals shape', which_marg.shape)

    # load explained variance by components
    explained_var = scipy.io.loadmat(f'{args.savepath_data}{args.participant}_{args.session}_{args.nbins_before_onset}_{args.nbins_after_onset}_explained_var.mat')
    explained_var = explained_var['margVar']
    # print(explained_var)
    print('Explained variance shape', explained_var.shape)

    # # plot projections
    plot_projections(projections, which_marg, explained_var)

    # plot explained variance piechart and bar plot
    plot_explained_variance(explained_var)

    # load correlation, decoder dot product, kendall p-value
    correlation = scipy.io.loadmat(f'{args.savepath_data}{args.participant}_{args.session}_{args.nbins_before_onset}_{args.nbins_after_onset}_correlations.mat')
    correlation = correlation['a']
    print('Correlation shape', correlation.shape)

    decoder_dot_product = scipy.io.loadmat(f'{args.savepath_data}{args.participant}_{args.session}_{args.nbins_before_onset}_{args.nbins_after_onset}_decoder_matrix_dot_product.mat')
    decoder_dot_product = decoder_dot_product['b']
    print('Decoder dot product shape', decoder_dot_product.shape)

    pval = scipy.io.loadmat(f'{args.savepath_data}{args.participant}_{args.session}_{args.nbins_before_onset}_{args.nbins_after_onset}_kendall_pvalue.mat')
    pval = pval['psp']
    print('Pvalue shape', pval.shape)

    map_matrix = np.tril(correlation, -1) + np.triu(decoder_dot_product) #lower triangle is correlation, upper triangle is decoder dot product
    # components with significant difference in dot product
    indices = np.where((abs(np.triu(decoder_dot_product, 1)) > 3.3 / np.sqrt(n_channels)) & (pval < 0.001))
    # print(indices[0])
    # print(indices[1])

    # plot correlation matrix
    plot_correlation(map_matrix, indices)

    # check how many pairs of of components in the dot product belong to word-loudness and how many are significantly orthogonal
    which_marg_word = np.where(which_marg.squeeze() == marg_names.index('Word'))[0]
    which_marg_loudness = np.where(which_marg.squeeze() == marg_names.index('Loudness'))[0]
    print(which_marg_word, which_marg_loudness)

    n_word_loudness_axis_pairs = len(which_marg_word) * len(which_marg_loudness)
    print(f'Number of word-loudness axis pairs: {n_word_loudness_axis_pairs}')
    print(f'Number of word-loudness axis pairs significantly non-orthogonal: {np.sum((abs(decoder_dot_product[which_marg_word, :][:, which_marg_loudness]) > 3.3 / np.sqrt(n_channels)) & (pval[which_marg_word, :][:, which_marg_loudness] < 0.001))}')