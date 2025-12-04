import numpy as np
import librosa
import math
import matplotlib.pyplot as plt
from IPython.display import Audio
 
 
def get_audio_onset_offset(pred_audio, display_audio = False, mic_audio = None, cue = None, intersegment_duration = 3000, amplitude_percentage = 0.1):
    '''
    Determines the onset and offset of attempted speech based on predicted audio which has been resampled to 30kHz (same as neural data).
    inputs - 
        pred_audio (array (float)): predicted audio array of shape (samples, 1)
        display_audio [optional] (bool): to plot the audio and display an audio play button, default False
        mic_audio [optional] (array(float)):  
    '''
    
    pred_audio = np.squeeze(pred_audio)
    if mic_audio is not None:
        mic_audio = np.squeeze(mic_audio)

    # resample predicted audio to 30kHz from 16kHz
    pred_audio = librosa.resample(pred_audio, orig_sr = 16000, target_sr = 30000)


    # onset and offset detection - wherever amplitude is greater than amplitude threshold
    max_amplitude = max(pred_audio)
    amplitude_threshold = amplitude_percentage * max_amplitude
    pred_audio_ind = np.where(pred_audio > amplitude_threshold)[0]

    

    # if the duration of two adjacent indices (where amplitude > threshold) is greater than intersegment_duration, 
    # then they are the end and start of different audio segments
    start_ind = [pred_audio_ind[0]]
    end_ind = []
    for i in range(len(pred_audio_ind[1:])):
        if pred_audio_ind[i] - pred_audio_ind[i-1] > intersegment_duration:
            end_ind.append(pred_audio_ind[i-1])
            start_ind.append(pred_audio_ind[i])
    end_ind.append(pred_audio_ind[-1])

    # the longest audio segment is where the word is predicted as these are single word trials, 
    # any noise will be spurious compared to the word duration
    have_ind = 0
    max_duration = 0
    for i in range(0, len(start_ind)): # made 1 as 0, dec 3 2023
        if end_ind[i] - start_ind[i] > max_duration:
            have_ind = i
            max_duration = end_ind[i] - start_ind[i]

    if display_audio:
        # plot microphone audio
        fig1 = plt.figure(figsize = (20,3))
        if mic_audio is not None:
            plt.subplot(2,1,1)
            plt.plot(mic_audio)
            plt.title (f'microphone audio - {cue}')
            plt.xlim(0, len(mic_audio))
            plt.show()
            display(Audio(data = mic_audio, rate = 30000))

        # plot predicted audio
        plt.subplot(2,1,2)
        plt.plot(pred_audio)
        plt.title (f'predicted audio - {cue}')
        plt.xlim(0, len(pred_audio))

        # plot predicted audio onset and offset
        vad = np.zeros((len(pred_audio)))
        vad[start_ind[have_ind]: end_ind[have_ind]] = 0.8* max_amplitude
        plt.plot(vad)
        plt.show()

        display(Audio(data = pred_audio, rate = 30000))
        display(Audio(data = pred_audio[start_ind[have_ind]:end_ind[have_ind]], rate = 30000))

    return start_ind[have_ind], end_ind[have_ind]


def get_audio_aligned_data_from_trial_ids(
        data, 
        trial_indices, 
        nbins_before_audio_onset, 
        nbins_after_audio_onset,
        amplitudes = ['MIME', 'WHISPER', 'NORMAL', 'LOUD'],
        words = ['be', 'do', 'have', 'know', 'my', 'going'],
        return_word_label = True,
        ):
    
    '''
    inputs - 
        data (dict): contains the fields spikepow, threshcross, predaudio16k, sentences etc.
        trial_indices (list(int)): list of indices for selecting the trials in data
        nbins_before_audio_onset (int): number of 10ms bins before audio onset
        nbins_after_audio_onset (int): number of 10ms bins after audio onset
        amplitudes (list(str)): list of amplitudes in the dataset
        words (list(str)): list of words in the dataset
        return_word_label (bool): each trial's word is collected as an index from words list and used in a word classifier (similar to amplitude classifier); default True

    returns - 
        x_all (np.array(batch, nbins_before_audio_onset + nbins_after_audio_onset, channels)): 3D array with concatenated spike broadband power and threshold crossing (batch, time, 512)
        y_all (np.array(batch, label)): 2D array with amplitude labels [0, 1, ... , 3] (batch, label)
        y_word_all (np.array(batch, label)): 2D array with word labels in [0, 1, ... , 5] (batch, label)

    '''
    


    x_all = np.empty((0, nbins_before_audio_onset + nbins_after_audio_onset, np.squeeze(data['spikepow'])[0].shape[-1] + np.squeeze(data['threshcross'])[0].shape[-1])) # batch x window x channels
    y_all = np.empty((0,))

    if return_word_label:
        y_word_all = np.empty((0,))

    for ind in trial_indices:
        # print(ind)

        # get audio onset and offset indices at 30kHz sampling rate
        start_ind, end_ind = get_audio_onset_offset(np.squeeze(data['predaudio16k'])[ind])
        
        binned_start_ind = math.floor(start_ind * 100 / 30000) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index (1s has 100 10ms bins)
        binned_end_ind = math.ceil(end_ind * 100 / 30000) # divide by sampling rate (30kHZ), scale it to ms by multiplying with 1000, divide by 10 to get bin index (1s has 100 10ms bins)

        x = np.squeeze(data['spikepow'])[ind][binned_start_ind - nbins_before_audio_onset: binned_start_ind + nbins_after_audio_onset, :]
        x = np.append(x, np.squeeze(data['threshcross'])[ind][binned_start_ind - nbins_before_audio_onset: binned_start_ind + nbins_after_audio_onset, :], axis = -1)

        # print(x.shape[0])
        if x.shape[0] == (nbins_before_audio_onset + nbins_after_audio_onset):
            # print('yay we have the trial')
            x_all = np.append(x_all, np.expand_dims(x,axis = 0), axis = 0)
            y_all = np.append(y_all, np.expand_dims(amplitudes.index(data['cue'][ind].split(':')[0].strip()), axis = 0), axis = 0)
            if return_word_label:
                y_word_all = np.append(y_word_all, np.expand_dims(words.index(data['cue'][ind].split(':')[1].strip()), axis = 0), axis = 0)


    if return_word_label:
        return x_all, y_all, y_word_all
    
    # print(x_all.shape, y_all.shape)
    
    return x_all, y_all


def get_audio_onset_offset_sentence(pred_audio, display_audio = False, mic_audio = None, cue = None, intersegment_duration = 3000, amplitude_percentage = 0.1):
    '''
    Determines the onset and offset of attempted speech based on predicted audio which has been resampled to 30kHz (same as neural data).
    inputs - 
        pred_audio (array (float)): predicted audio array of shape (samples, 1)
        display_audio [optional] (bool): to plot the audio and display an audio play button, default False
        mic_audio [optional] (array(float)):  
    '''
    
    pred_audio = np.squeeze(pred_audio)
    if mic_audio is not None:
        mic_audio = np.squeeze(mic_audio)

    # resample predicted audio to 30kHz from 16kHz
    pred_audio = librosa.resample(pred_audio, orig_sr = 16000, target_sr = 30000)


    # onset and offset detection - wherever amplitude is greater than amplitude threshold
    max_amplitude = max(pred_audio)
    amplitude_threshold = amplitude_percentage * max_amplitude
    pred_audio_ind = np.where(pred_audio > amplitude_threshold)[0]

    

    # if the duration of two adjacent indices (where amplitude > threshold) is greater than intersegment_duration, 
    # then they are the end and start of different audio segments
    start_ind = [pred_audio_ind[0]]
    end_ind = []
    for i in range(len(pred_audio_ind[1:])):
        if pred_audio_ind[i] - pred_audio_ind[i-1] > intersegment_duration:
            end_ind.append(pred_audio_ind[i-1])
            start_ind.append(pred_audio_ind[i])
    end_ind.append(pred_audio_ind[-1])

    # the longest audio segment is where the word is predicted as these are single word trials, 
    # any noise will be spurious compared to the word duration
    have_ind = np.arange(len(start_ind))
    # max_duration = 0
    # for i in range(0, len(start_ind)): # made 1 as 0, dec 3 2023
    #     if end_ind[i] - start_ind[i] > max_duration:
    #         have_ind = i
    #         max_duration = end_ind[i] - start_ind[i]

    if display_audio:
        # plot microphone audio
        if mic_audio is not None:
            fig1 = plt.figure(figsize = (20,3))
            plt.plot(mic_audio)
            plt.title (f'microphone audio - {cue}')
            plt.xlim(0, len(mic_audio))
            plt.show()
            display(Audio(data = mic_audio, rate = 30000))

        # plot predicted audio
        fig2 = plt.figure(figsize = (20,3))
        plt.plot(pred_audio)
        plt.title (f'predicted audio - {cue}')
        plt.xlim(0, len(pred_audio))

        # plot predicted audio onset and offset
        vad = np.zeros((len(pred_audio)))
        for ind in have_ind:
            vad[start_ind[ind]: end_ind[ind]] = 0.8* max_amplitude
        plt.plot(vad)
        plt.show()

        display(Audio(data = pred_audio, rate = 30000))
        # display(Audio(data = pred_audio[start_ind[have_ind]:end_ind[have_ind]], rate = 30000))

    return start_ind, end_ind



def t16_get_audio_onset_offset_sentence(pred_audio, display_audio = False, mic_audio = None, cue = None, intersegment_duration = 3000, amplitude_percentage = 0.1):
    '''
    Determines the onset and offset of attempted speech based on predicted audio which has been resampled to 30kHz (same as neural data).
    inputs - 
        pred_audio (array (float)): predicted audio array of shape (samples, 1)
        display_audio [optional] (bool): to plot the audio and display an audio play button, default False
        mic_audio [optional] (array(float)):  
    '''
    # pred audio is mic audio, already at 30k

    pred_audio = np.squeeze(pred_audio)
    if mic_audio is not None:
        mic_audio = np.squeeze(mic_audio)

    # resample predicted audio to 30kHz from 16kHz
    # pred_audio = librosa.resample(pred_audio, orig_sr = 16000, target_sr = 30000)


    # onset and offset detection - wherever amplitude is greater than amplitude threshold
    max_amplitude = max(pred_audio)
    amplitude_threshold = amplitude_percentage * max_amplitude
    pred_audio_ind = np.where(pred_audio > amplitude_threshold)[0]

    # if the duration of two adjacent indices (where amplitude > threshold) is greater than intersegment_duration, 
    # then they are the end and start of different audio segments
    start_ind = [pred_audio_ind[0]]
    end_ind = []
    for i in range(len(pred_audio_ind[1:])):
        if pred_audio_ind[i] - pred_audio_ind[i-1] > intersegment_duration:
            end_ind.append(pred_audio_ind[i-1])
            start_ind.append(pred_audio_ind[i])
    end_ind.append(pred_audio_ind[-1])

    # the longest audio segment is where the word is predicted as these are single word trials, 
    # any noise will be spurious compared to the word duration
    have_ind = np.arange(len(start_ind))
    # max_duration = 0
    # for i in range(0, len(start_ind)): # made 1 as 0, dec 3 2023
    #     if end_ind[i] - start_ind[i] > max_duration:
    #         have_ind = i
    #         max_duration = end_ind[i] - start_ind[i]

    if display_audio:
        # plot microphone audio
        if mic_audio is not None:
            fig1 = plt.figure(figsize = (20,3))
            plt.plot(mic_audio)
            plt.title (f'microphone audio - {cue}')
            plt.xlim(0, len(mic_audio))
            plt.show()
            display(Audio(data = mic_audio, rate = 30000))

        # plot predicted audio
        fig2 = plt.figure(figsize = (20,3))
        plt.plot(pred_audio)
        plt.title (f'predicted audio - {cue}')
        plt.xlim(0, len(pred_audio))

        # plot predicted audio onset and offset
        vad = np.zeros((len(pred_audio)))
        for ind in have_ind:
            vad[start_ind[ind]: end_ind[ind]] = 0.8* max_amplitude
        plt.plot(vad)
        plt.show()

        display(Audio(data = pred_audio, rate = 30000))
        # display(Audio(data = pred_audio[start_ind[have_ind]:end_ind[have_ind]], rate = 30000))

    return start_ind, end_ind


def t16_get_audio_onset_offset(pred_audio, display_audio = False, mic_audio = None, cue = None, intersegment_duration = 3000, amplitude_percentage = 0.1):
    '''
    Determines the onset and offset of attempted speech based on predicted audio which has been resampled to 30kHz (same as neural data).
    inputs - 
        pred_audio (array (float)): predicted audio array of shape (samples, 1)
        display_audio [optional] (bool): to plot the audio and display an audio play button, default False
        mic_audio [optional] (array(float)):  
    '''
    
    # pred audio is mic audio, already at 30k

    pred_audio = np.squeeze(pred_audio)
    if mic_audio is not None:
        mic_audio = np.squeeze(mic_audio)

    # onset and offset detection - wherever amplitude is greater than amplitude threshold
    max_amplitude = max(pred_audio)
    amplitude_threshold = amplitude_percentage * max_amplitude
    pred_audio_ind = np.where(pred_audio > amplitude_threshold)[0]

    # if the duration of two adjacent indices (where amplitude > threshold) is greater than intersegment_duration, 
    # then they are the end and start of different audio segments
    start_ind = [pred_audio_ind[0]]
    end_ind = []
    for i in range(len(pred_audio_ind[1:])):
        if pred_audio_ind[i] - pred_audio_ind[i-1] > intersegment_duration:
            end_ind.append(pred_audio_ind[i-1])
            start_ind.append(pred_audio_ind[i])
    end_ind.append(pred_audio_ind[-1])

    # the longest audio segment is where the word is predicted as these are single word trials, 
    # any noise will be spurious compared to the word duration
    have_ind = 0
    max_duration = 0
    for i in range(0, len(start_ind)): # made 1 as 0, dec 3 2023
        if end_ind[i] - start_ind[i] > max_duration:
            have_ind = i
            max_duration = end_ind[i] - start_ind[i]

    if display_audio:
        # plot microphone audio
        if mic_audio is not None:
            fig1 = plt.figure(figsize = (20,3))
            plt.plot(mic_audio)
            plt.title (f'microphone audio - {cue}')
            plt.xlim(0, len(mic_audio))
            plt.show()
            display(Audio(data = mic_audio, rate = 30000))

        # plot predicted audio
        fig2 = plt.figure(figsize = (20,3))
        plt.plot(pred_audio)
        plt.title (f'predicted audio - {cue}')
        plt.xlim(0, len(pred_audio))
        plt.xticks(np.arange(0, len(pred_audio), 15000), np.arange(0, len(pred_audio)/30000, 0.5))
        plt.xlabel ('Time (s)')

        # plot predicted audio onset and offset
        vad = np.zeros((len(pred_audio)))
        vad[start_ind[have_ind]: end_ind[have_ind]] = 0.8* max_amplitude
        plt.plot(vad)
        plt.show()

        display(Audio(data = pred_audio, rate = 30000))
        display(Audio(data = pred_audio[start_ind[have_ind]:end_ind[have_ind]], rate = 30000))

    return start_ind[have_ind], end_ind[have_ind]
