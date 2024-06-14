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
S1_peaks = find_peaks(df['S1 (uV)'], height = 6)
M1_peaks = find_peaks(df['M1 (uV)'], height = 6)

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
plt.plot(df["Time (sec)"], df["S1 (uV)"],alpha=0.7, label='S1')
plt.plot(df["Time (sec)"], df["M1 (uV)"],alpha=0.7, label='M1')
plt.legend()
plt.xlabel("Time (sec)")
plt.ylabel("Potential (uV)")
plt.title(f'Data in {datafile}')
plt.show()
