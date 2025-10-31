# srinivasan-speech-modes
This repository houses codes related to analyzing intracortical neural activity of speech modes and loudness.

## Installation
**Requirements:** Code has been tested in Python 3.9.
```
scipy 1.13.1
numpy 2.0.1
scikit-learn 1.5.1
noisereduce 3.0.3
```

## Generating manuscript results
To generate the results and plots we have reported in the manuscript, you can run the codes in the folder ```plot_results```. The python scripts in the folder read the results we obtained by analyzing the data and generate plots. The sub-folder ```results``` has the results we obtained (such as classification accuracies, firing rates etc.) obtained by analyzing the data as .pkl files.

Scripts to run for:

1. psths:
2. significant channels tuned to loudness or words:
3. pca:
4. dpca:
5. temporal loudness classification along trial:
6. overall loudness classification:
7. speech and breath analyses:
8. breath and speech psth:
9. speech and breath classification:

## Running analyses
Sample processed neural data from one of the participant is in ```data``` folder. We will upload the 

To run an analysis script in ```analyses_scripts``` folder, execute the example run command provided in the script with the required data. Note: these scripts will only generate the correct results when the entire dataset is provided. The results obtained using sample data could be wrong. Here, we provide the analyses scripts to just let you know how we implemented the analyses

Run the following scripts to generate these figures
1. Fig 1C psth -- psth.py
2. Fig 1D significant channels tuned to loudness -- significant_loudness_channels.py
3. Fig 2A pca -- t15_pca_electrodes.ipynb
4. Fig 2B-D dPCA -- plot_dPCA_results.py
5. Fig 3A loudness classification along trial -- ol_striding_classification.py
6. Fig 3B loudness classification -- ol_classification_performance.py
7. Supp Fig 3A-B speech and breath analyses -- instructed_breath_speech_breath_belt_analysis.py
8. Supp Fig 3C speech and breath psth -- instructed_breath_speech_psth.py
9. Supp Fig 3D-E speech and breath classification -- instructed_breath_speech_classification_analysis.py
