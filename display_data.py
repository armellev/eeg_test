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
import os

#%% Load data 

# Assume the working directory contains the file
datadir = "." 
datafile = "EEG_S1_prestim.csv"

datapath = os.path.join(datadir,datafile)

df = pd.read_csv(datapath)

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
