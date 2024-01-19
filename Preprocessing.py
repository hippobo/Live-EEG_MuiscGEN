# VERSION 1: 0913 Plot the topography of the emotion EEG signal
# This is the MNE PLOT VERSION of preprocessing
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from scipy.signal.windows import hann
from scipy import stats
from mne.viz.topomap import _add_colorbar, plot_topomap, _hide_frame
from mne.preprocessing import ICA
import pickle as pkl
import re
import math
from collections import Counter

# The data put into preprocessing based on 30s
class Preprocessing():

    def __init__(self, raw):
        # Modify the montage
        self.nchns = raw.info['nchan']
        self.freq = raw.info['sfreq']
        old_chn_names = raw.info['ch_names']
        new_chn_names = raw.info['ch_names'].copy()
        if 'A1' in new_chn_names:
            pass
        else:
            new_chn_names[-2] = 'A2'
            new_chn_names[-1] = 'A1'
        chan_names_dict = {old_chn_names[i]: new_chn_names[i] for i in range(32)}
        raw.rename_channels(chan_names_dict)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        # Match the corresponding montage to their index
        self.montage_index = dict(zip(np.arange(self.nchns) , new_chn_names))
        # split out the data matrix
        self.raw = raw
        # Ptr operation
        self.data = self.raw.get_data()


    def plot_eeg(self,second):
        self.raw.plot(duration=second, n_channels=32, clipping=None)

    def plot_sensors(self):
        self.raw.plot_sensors(ch_type='eeg',show_names=True)

    def band_pass_filter(self,l_freq,h_freq):
        # The default filter is a FIR filter,
        # therefore,  the input [l_freq,h_freq] is
        # [lower pass-band edge, higher pass-band edge]
        self.raw.filter(l_freq,h_freq)

    def down_sample(self,n_freq):
        self.raw.resample(n_freq)

    # Should cut the data into pieces than do the interpolation
    def bad_channels_interpolate(self,thresh1=None,thresh2=None,proportion=0.3):
        data = self.raw.get_data()
        # We found that the data shape of epochs is 3 dims
        # print(data.shape)
        if len(data.shape) > 2:
            data = np.squeeze(data)
        Bad_chns = []
        value = 0
        # Delete the much larger point
        if thresh1 != None:
            md = np.median(np.abs(data))
            value = np.where(np.abs(data) > (thresh1 * md),0,1)[0]
        if thresh2 != None:
            value = np.where((np.abs(data)) > thresh2,0,1)[0]
        # Use the standard to pick out the bad channels
        Bad_chns = np.argwhere((np.mean((1-value),axis=0)>proportion))
        if Bad_chns.size > 0:
            self.raw.info['bads'].extend([self.montage_index[str(bad)] for bad in Bad_chns] )
            print('Bad channels: ',self.raw.info['bads'])
            self.raw = self.raw.interpolate_bads()
        else:
            print('No bad channel currently')

    # You can manually exclude the ICA elements
    # Our auto preprocessing method is more effective for eye blink removal
    def eeg_ica(self, check_ica=None):
        ica = ICA(max_iter='auto',method='fastica')
        raw_ = self.raw.copy()
        ica.fit(self.raw)
        # Plot different elements of the signals
        eog_indices1, eog_score1 = ica.find_bads_eog(self.raw, ch_name='Fp1')
        eog_indices2, eog_score2 = ica.find_bads_eog(self.raw, ch_name='Fp2')
        eog_indices = list(set(eog_indices1 + eog_indices2))
        ica.exclude = eog_indices
        # ica.plot_sources(raw_)
        # Plot the components
        # ica.plot_components()
        if check_ica == True:
            print('Already use:', eog_indices)
            ica.plot_sources(raw_)
            ica.plot_components()
            eog_indices = input("Exclude ?")
            eog_indices = eog_indices.split(" ")
            eog_indices = list(map(int, eog_indices))
            ica.exclude = eog_indices
        ## Plot excluded one of the elements
        # ica.plot_overlay(raw_, exclude=[1])
        # ica.plot_properties(raw_, picks=[1, 16])

        # Exclude the elements you don't want
        ica.apply(self.raw)
        # self.raw.plot(duration = 5,n_channels = self.nchns,clipping = None)

    def average_ref(self):
        self.raw.set_eeg_reference(ref_channels='average')

    def _get_average_psd(self, energy_graph, freq_bands, sample_freq, stft_n=256):
        start_index = int(np.floor(freq_bands[0] / sample_freq * stft_n))
        end_index = int(np.floor(freq_bands[1] / sample_freq * stft_n))
        ave_psd = np.mean(energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
        return ave_psd

    def extract_psd_feature(self, window_size, freq_bands, stft_n=256):
        sample_freq = self.raw.info['sfreq']
        # Ptr operation
        self.data = self.raw.get_data()
        if len(self.data.shape) > 2:
            self.data = np.squeeze(self.data)
        n_channels, n_samples = self.data.shape
        point_per_window = int(sample_freq * window_size)
        window_num = int(n_samples // point_per_window)
        psd_feature = np.zeros((window_num, len(freq_bands), n_channels))

        for window_index in range(window_num):
            start_index, end_index = point_per_window * window_index, point_per_window * (window_index + 1)
            window_data = self.data[:, start_index:end_index]
            hdata = window_data * hann(point_per_window)
            fft_data = np.fft.fft(hdata, n=stft_n)
            energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])

            for band_index, band in enumerate(freq_bands):
                band_ave_psd = self._get_average_psd(energy_graph, band, sample_freq, stft_n)
                psd_feature[window_index, band_index, :] = band_ave_psd
        return psd_feature

# for mne
def read_data(folderPath):
    # Read the data
    rawdata = mne.io.read_raw_bdf(os.path.join(folderPath, 'data.bdf'), preload=True)
    rawdata,unit = unit_check(rawdata)
    fs = rawdata.info['sfreq']
    # Read the event
    ## read events
    try:
        events = mne.read_annotations(os.path.join(folderPath, 'evt.bdf'))
        onset = np.array([int(x * fs) for x in events.onset] )
        duration = np.array([int(x) for x in events.duration])
        trigger = np.array([i for i in events.description])
        print("Original trigger events:", trigger)
        # Impedance checking
        trigger,onset,duration,impedance = inter_impedance_inspect(trigger,onset,duration)
        trigger = np.array([int(x) for x in trigger])
        # Check wheter there is ERP data
        trigger,onset, duration, experiment = trigger_check(trigger,onset, duration)
        return trigger, onset, duration, rawdata, [unit, impedance, experiment]
    except:
        raise RuntimeError("No event was found")


def data_concat(eegData, videoData:np.array, video:int):
    fs = 250; secs = 30;
    if len(videoData.shape) > 2:
        videoData = np.squeeze(videoData)
    trigger = np.zeros((1,fs * secs))
    trigger[0][0] = video
    print('The shape of current epoch:', videoData.shape)
    if videoData.shape[1] > fs * secs:
        videoData = videoData[:,-fs*secs:]
    elif videoData.shape[1] < fs * secs:
        raise RuntimeError("The length of epoch is wrong")
    videoData = np.vstack((videoData, trigger))
    if eegData is None:
        eegData = videoData
    else:
        eegData = np.hstack((eegData, videoData))
    return eegData

def eeg_save(subject:str, eegData_trigger:np.array, filepath):
    if len(subject) == 1:
        subject = '00' + subject
    elif len(subject) == 2:
        subject = '0' + subject
    f = open(filepath + '/' + subject +'.pkl','wb')
    pkl.dump(eegData_trigger,f)
    f.close()

# DATA INSPECTION
def unit_check(rawdata):
    # The first batch and the second batch have different unit (uV and V)
    original_raw = rawdata.copy()
    # Here can use np.log to make sure the level of unit, V or uV
    data_mean = np.mean(np.abs(original_raw._data))
    unit = 'uV'
    if math.log(np.mean(np.abs(original_raw._data))) < 0:
        # print('Unit change :', math.log(data_mean))
        print('Unit change :', data_mean)
        original_raw._data = original_raw._data * 1000 * 1000
        unit = 'V'
    return original_raw, unit

def inter_impedance_inspect(trigger,onset,duration):
    if 'Start Impedance' in trigger:
        pos = np.where((trigger == 'Start Impedance') | (trigger == 'Stop Impedance'))[0]
        # Make sure that there is no impedance before the whole experiment
        trigger_left = trigger[int(pos[0]):]
        # key_trigger_left = np.where((trigger_left!=0)&(trigger_left<29))[0]
        trigger_left = np.delete(trigger_left, [0,1])
        print(trigger_left)
        if trigger_left.size == 0:
            trigger = np.delete(trigger, pos)
            onset = np.delete(onset, pos)
            duration = np.delete(duration, pos)
            print("There is an Impedance in the dataset but at the end")
            impedance = 0
        else:
            print('A impedance occurs in the middle of the data')
            print(trigger)
            raise ValueError('Please Delete This Subject')
    else:
        impedance = 1
        pass
    return trigger, onset, duration, impedance

def trigger_check(trigger,onset,duration):
    trigger_cate = list(set(list(trigger)))
    experiment = 1
    for trigg in [91,92,93,94,95,96]:
        if trigg in trigger_cate:
            experiment = 'ERP'
    if experiment == 'ERP':
        print("There is other dataset inside the data")
        real_Exper = np.where(trigger==100)[0][4]
        real_Exper_end = np.where(trigger==102)[0][-1]+1
        if real_Exper_end > real_Exper:
            pass
        else:
            real_Exper = np.where(trigger == 100)[0][3]
        trigger = trigger[real_Exper:real_Exper_end]
        onset = onset[real_Exper:real_Exper_end]
        duration = duration[real_Exper:real_Exper_end]

    return trigger, onset, duration, experiment


def plot_the_topo_corr(y_axis,x_axis,psd_corr,filename):
    pos = np.load('./pos.npy')
    new_order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,17,16]
    pos = pos[new_order][:-2]
    y_axes = len(y_axis)
    x_axes = len(x_axis)
    fig, axes = plt.subplots(y_axes, x_axes+1, figsize=(x_axes * y_axes, 50))
    cmap = 'RdBu_r'
    rows, columns = axes.shape
    for column in range(columns):
        for row in range(rows):
            _hide_frame(axes[row][column])
    for i, y in enumerate(y_axis):
        for j, x in enumerate(x_axis):
            if i == 0:
                axes[i][j].set_title(x, fontsize=50, y=1.3)
            if j == 0:
                axes[i][j].set_title(y, fontsize=50, x=-1.8, y=0.5, loc='left')
            vmin = np.min(psd_corr[y][x][0,:]); vmax = np.max(psd_corr[y][x][0,:])
            print(vmin, vmax)
            im, _ = plot_topomap(np.squeeze(psd_corr[y][x][0, :]), pos, vmin=vmin, vmax=vmax, axes=axes[i][j], cmap=cmap, show=False)
    # Plot the colorbar
    a = plt.subplot(3, x_axes+2, 2*x_axes+4, frameon=False)
    _add_colorbar(a, im, cmap=cmap, side='left')
    _hide_frame(a)
    # plt.show()
    plt.gcf().subplots_adjust(left=0.2)
    plt.savefig('../Validation/Correlation_analysis/' + filename + '.png');plt.clf();


def channel_modify(data, first_or_second):
    # data (33,210000)
    new_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                 30, 31, 17, 16, 32]
    if first_or_second == 1 :
        data = data[np.array(new_order),:]
    # # Delete the reference channel
    # data = np.vstack((data[:-3, :],data[-1,:]))
    chns = 32; fs = 250; n_vids = 28; sec = 30;
    eegdata = np.zeros((n_vids, chns, fs * sec))
    video_index = np.where(data[-1,:].T>0)[0]
    video = data[-1, video_index]
    video_arange = np.argsort(data[-1,video_index])
    video_arange_index = video_index[video_arange]
    # Modify the video order into the normal arrange
    for idx, vid in enumerate(video_arange_index):
        eegdata[idx, :, :] = data[:-1, vid:vid + fs*sec]
    return eegdata