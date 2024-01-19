# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
from scipy import signal, stats
import mne
from tkinter import *
import subprocess, time
eeg_status = {"message": "Not started", "status": "idle"}
eeg_model = keras.models.load_model("./models/EmotionNetV2.h5")

# Setup input buffer 

fs = 256
inputLength = 10.5 # Length of input in seconds
shiftLength = 5 # Time between epochs
samples = int(shiftLength * fs) # How many samples to gather in every cycle

bufferSize = int(128 * inputLength) # Size of buffer in samples. Enough to hold one set of downsampled input.

buffers = np.zeros((4, bufferSize)) # buffers for each of the four channels

# Push new data onto buffer, removing any old data on the end
def updateBuffer(buffer, newData):
    assert len(newData.shape) == len(buffer.shape) and buffer.shape[0] >= newData.shape[0], "Buffer shape ({}) and new data shape ({}) are not compatible.".format(buffer.shape, newData.shape)
    size = newData.shape[0]
    buffer[:-size] = buffer[size:]
    buffer[-size:] = newData
    return buffer


def eeg_inference():
    global eeg_status
    # Get the streamed data from the Muse. Blue Muse must be streaming.
    process = subprocess.Popen("muselsl stream --address 00:55:DA:B5:D5:CF", shell=True)
    print('Waiting 10 seconds for EEG stream to start...')
    time.sleep(10)

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=10, minimum=1)
   

    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    inlet = StreamInlet(streams[0], max_buflen=60, max_chunklen=int(inputLength))
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    fs = int(info.nominal_srate())
    print(fs)


    # Important info for plot
    # img = plt.imread("EmotionSpace.jpg")
    # width = img.shape[1]
    # height = img.shape[0]
    # centerX = int(width / 2) -8
    # centerY = int(height / 2) +8
    # pixPerValence = centerX / 5
    # pixPerArousal = centerY / 5

    try:
        eeg_status = {"message": "Connecting to EEG...", "status": "connecting"}
        while True:
            eeg_status = {"message": "Processing EEG data...", "status": "processing"}
            # Continuously update EEG data and attempt to measure emotions
            
            data, timestamp = inlet.pull_chunk(timeout=5, max_samples=samples)
            eeg = np.array(data).swapaxes(0,1)

            """
            The DEAP dataset has 3 processing steps which we must also apply to out data:
            1. downsample to 128 Hz.
            2. Apply a bandpass frequency filter from 4-45 Hz
            3. Average data to the common reference
            """

            # Downsample
            processedEEG = signal.resample(eeg, int(eeg.shape[1] * (128 / fs)), axis=1)

            # Apply bandpass filter from 4-45Hz
            # processedEEG = mne.filter.filter_data(processedEEG, 128, 4, 45, filter_length=512, 
            #                             l_trans_bandwidth='auto', h_trans_bandwidth='auto', 
            #                             phase='zero', fir_window='hamming', verbose=0)
            processedEEG = mne.filter.filter_data(processedEEG, 128, 4, 45, filter_length='auto', 
                                                l_trans_bandwidth='auto', h_trans_bandwidth='auto', 
                                                phase='zero', fir_window='hamming', verbose=0)


            
            # Zero mean
            processedEEG -= np.mean(processedEEG, axis=1, keepdims=True)
            
            
            # Update buffer
            for channel in range(buffers.shape[0]):
                buffers[channel] = updateBuffer(buffers[channel], processedEEG[channel])
            
        
        
            buffers_reshaped = np.expand_dims(buffers, axis=0)
            emotions = eeg_model.predict(buffers_reshaped)

            # Clip results in case they have outliers
            emotions = np.clip(emotions, 1, 9)
            
            valence = emotions[0][0]
            arousal = emotions[0][1]
            dominance = emotions[0][2]
            
            
            # Calculate Emotion Display and plot the graph
            # x = valence * pixPerValence
            # y = (10 - arousal) * pixPerArousal
            
            # plt.figure(figsize=(10,10))
            # plt.imshow(img)

            # plt.plot(x, y, color='red', marker='o', markersize=36)
            # plt.show()
            print("Valence: {:.2f}".format(valence))
            print("Arousal: {:.2f}".format(arousal))
            print("Dominance: {:.2f}".format(dominance))
    except Exception as e:
        eeg_status = {"message": f"Error: {str(e)}", "status": "error"}
            # success = True
            # return success, valence, arousal, dominance
        
    finally:
        inlet.close_stream()
        eeg_status = {"message": "EEG stream closed", "status": "closed"}