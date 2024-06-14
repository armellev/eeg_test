#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple example script to display an associated data file.

Part of the Carney Institute's 2024 Computational Fluency Short Course.

To run from a terminal, type:

python display_data.py

The code must be in the same working directory as the data file EEG_S1_prestim.csv.
"""

#%% Import block

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import numpy as np

#%% Load data 

# Assume the working directory contains the file
datadir = "." 
datafile = "EEG_S1_prestim.csv"

datapath = os.path.join(datadir,datafile)

df = pd.read_csv(datapath)

#%% attempt to find peaks

# Find peaks in S1 & M1
S1_peaks = find_peaks(df['S1 (uV)'], height = 6, distance = 20)
M1_peaks = find_peaks(df['M1 (uV)'], height = 6)





#%% attempt to remove noise

from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Example usage:
# Replace 'eeg_data' with your actual EEG data
S1_filtered = bandpass_filter(df['S1 (uV)'], lowcut=0.5, highcut=30, fs=1000, order=5)

#%% attempt 2

from scipy.signal import butter, filtfilt, find_peaks

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter_indices(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    peaks, _ = find_peaks(filtered_data, height=0)  # Finding peaks in the filtered signal
    return peaks

# Example usage:
# Replace 'eeg_data' with your actual EEG data
peaks_indices = bandpass_filter_indices(df["S1 (uV)"], lowcut=0.5, highcut=70, fs=1000, order=5)


#%% find the frequency

# Perform Fast Fourier Transform (FFT) on S1
fft_result_S1 = np.fft.fft(df['S1 (uV)'])
n_S1 = len(fft_result_S1)
freq_S1 = np.fft.fftfreq(n_S1)  # Frequency axis

# Find the peak frequency
peak_freq_index_S1 = np.argmax(np.abs(fft_result_S1[:n_S1//2]))  # Finding the index of the maximum amplitude in the positive frequency range
peak_freq_S1 = freq_S1[peak_freq_index_S1]  # Frequency corresponding to the peak amplitude



# Perform Fast Fourier Transform (FFT) on M1
fft_result_M1 = np.fft.fft(df['M1 (uV)'])
n_M1 = len(fft_result_M1)
freq_M1 = np.fft.fftfreq(n_M1)  # Frequency axis

# Find the peak frequency
peak_freq_index_M1 = np.argmax(np.abs(fft_result_M1[:n_M1//2]))  # Finding the index of the maximum amplitude in the positive frequency range
peak_freq_M1 = freq_M1[peak_freq_index_M1]  # Frequency corresponding to the peak amplitude

# maybe try to take the info from the other result then do how many peaks over time

#%% Display the data

print("Close figure to quit")


plt.figure()
plt.plot(df["S1 (uV)"],alpha=0.7, label='S1')
#plt.plot( df["M1 (uV)"],alpha=0.7, label='M1')
plt.plot(S1_peaks[0],S1_peaks[0]*0, '.', alpha =0.7, label='S1_peaks')
#plt.plot(S1_filtered, alpha = 0.7, label = 'S1_filtered')
plt.legend()
plt.xlabel("Time (sec)")
plt.ylabel("Potential (uV)")
plt.title(f'Data in {datafile}')
plt.show()
