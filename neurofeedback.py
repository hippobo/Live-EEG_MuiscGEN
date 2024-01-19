# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.

The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al

Adapted from https://github.com/NeuroTechX/bci-workshop
"""



from os import wait
from matplotlib.pylab import f
import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop , resolve_streams # Module to receive EEG data
import utils  # Our own utility functions
import subprocess, time
from muselsl import record
# Handy little enum to make code more readable


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
#F8 and F7
INDEX_CHANNEL = [0,1, 2,3]


if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    process = subprocess.Popen("muselsl stream --address 00:55:DA:B5:D5:CF", shell=True)
    print('Waiting 10 seconds for EEG stream to start...')
    time.sleep(10)

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=10, minimum=1)
    # streams = resolve_streams(wait_time=2)
    # muses = list_muses()
    # stream(muses[0]['address'])

    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    # fs = int(info.nominal_srate())
    fs = 250 #for matching the faced dataset

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH),  len(INDEX_CHANNEL)))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')




    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))
            

            # Only keep the channels we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
            # print(ch_data.shape)
            # print(ch_data)


            for i, channel in enumerate(INDEX_CHANNEL):
                # Reshape the slice to be 2D with one column
                channel_buffer = eeg_buffer[:, i].reshape(-1, 1)
                channel_data = ch_data[:, i].reshape(-1, 1)

                # Update buffer for each channel
                channel_buffer, filter_state = utils.update_buffer(
                    channel_buffer, channel_data, notch=True,
                    filter_state=filter_state)

                # Reshape the updated buffer back to 1D and assign it back to eeg_buffer
                eeg_buffer[:, i] = channel_buffer.ravel()

            """ 3.2 COMPUTE BAND POWERS """
            # band_powers_all_channels = []
            
            # for channel in INDEX_CHANNEL:
            #     # Extract the data for the current channel
            #     data_epoch = utils.get_last_data(eeg_buffer[:, channel - 1].reshape(-1, 1), EPOCH_LENGTH * fs)
            #     # Compute band powers and store them in a list
            #     band_powers = utils.compute_band_powers(data_epoch, fs)
            #     band_powers_all_channels.append(band_powers)
                
            #     print(f'Channel {channel}: Delta: {band_powers[Band.Delta]}, Theta: {band_powers[Band.Theta]}, '
            #         f'Alpha: {band_powers[Band.Alpha]}, Beta: {band_powers[Band.Beta]}')
             
            band_powers_all_channels = []
           

        # Process each channel
            for i, channel in enumerate(INDEX_CHANNEL):
                # Extract the data for the current channel
                data_epoch = utils.get_last_data(eeg_buffer[:, channel - 1].reshape(-1, 1), EPOCH_LENGTH * fs)
                
                # Compute band powers for the current channel
                band_powers = utils.compute_band_powers(data_epoch, fs)
                band_powers_all_channels.append(band_powers)
                
                # print(f'Channel {channel}: Delta: {band_powers[Band.Delta]}, Theta: {band_powers[Band.Theta]}, '
                #     f'Alpha: {band_powers[Band.Alpha]}, Beta: {band_powers[Band.Beta]}')

      
            if len(band_powers_all_channels) == 2:
            # Calculate valence and arousal
                
                valence = (band_powers_all_channels[1][Band.Alpha] - band_powers_all_channels[0][Band.Alpha])
                arousal = (band_powers_all_channels[1][Band.Beta] + band_powers_all_channels[0][Band.Beta]) / \
                        (band_powers_all_channels[1][Band.Alpha] + band_powers_all_channels[0][Band.Alpha])

                
                # print("tanh val", np.tanh(valence))
                # print("tanh arousal" , np.tanh(arousal))
                tanh_val = np.tanh(valence)
                tanh_arousal = np.tanh(arousal)

              
                # # Update min and max values for normalization
                # min_valence = min(min_valence, valence)
                # max_valence = max(max_valence, valence)
                # min_arousal = min(min_arousal, arousal)
                # max_arousal = max(max_arousal, arousal)

                #     # Normalize valence and arousal to be between -1 and 1
                # # If the current min and max are not equal, normalize the value. If they are, assign 0.
                # if max_valence != min_valence:
                #     normalized_valence = 2 * (valence - min_valence) / (max_valence - min_valence) - 1
                # else:
                #     normalized_valence = 0  # No variation, consider as middle point

                # if max_arousal != min_arousal:
                #     normalized_arousal = 2 * (arousal - min_arousal) / (max_arousal - min_arousal) - 1
                # else:
                #     normalized_arousal = 0 
                # print("norm val: ", normalized_valence)
                # print("norm arousal : ", normalized_arousal)




    except KeyboardInterrupt:

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the window open