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
To generate the results and plots we have reported in the manuscript, you can run the codes in the folder ```plotting_scripts```. The python scripts in the folder read the results we obtained by analyzing the data and generate plots. The sub-folder ```plotting_data``` has the results we obtained (such as classification accuracies, firing rates etc.) obtained by analyzing the data as .pkl files.

Scripts to run for:

1. psths: psth.py
2. significant channels tuned to loudness or words: ch_encoding.py
