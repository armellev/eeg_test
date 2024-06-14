#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for practice at troubleshooting (by design, will not run as is).

Part of the Carney Institute's 2024 Computational Fluency Short Course.
"""

#%% Import block

import pandas as pd
import numpy as np

#%% Load data

# To avoid aving to deal with an external file, we will
# simulate loading data by direct construction of a
# dataframe

df = pd.DataFrame( {'marker':['hox','sonic','pv','ads1'], 
                    'perm': [0.11,-0.34,0.28,0.07],
                    'area': [1028,np.nan,1017,998],
                    'num_cells': [51,28,250,34]})

#%% Analysis 

# This is where you should start troubleshooting

def get_cell_density(numcells, area):
    '''
    Find the number of cells per area
    '''
    return numcells/area

cell_density = []
for k in range(0,4):
    area = df.iloc[k]["area"]
    numcells = df.iloc[k]["num_cells"]
    cell_density.append( get_cell_density(numcells,area))

df["cell_density"] = cell_density


