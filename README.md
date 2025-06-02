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

## Running analyses
Sample processed neural data from one of the participant is in ```data``` folder.

To run an analysis script in ```analyses_scripts``` folder, execute the example run command provided in the script with the required data.

Run the following scripts to generate these figures
1. Fig 1C psth -- psth.py
2. Fig 1D significant channels tuned to loudness -- significant_loudness_channels.py
3. Fig 2A pca -- t15_pca_electrodes.ipynb
4. Fig 2B-D dPCA -- plot_dPCA_results.py
5. Fig 3A loudness classification along trial -- ol_striding_classification.py
6. Fig 3B loudness classification -- ol_classification_performance.py
7. Supp Fig 3 speech and breath analyses -- instructed_breath_speech_breath_belt_analysis.py, instructed_breath_speech_classification_analysis.py
