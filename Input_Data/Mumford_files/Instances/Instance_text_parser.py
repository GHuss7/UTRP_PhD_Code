# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 08:49:16 2020

@author: 17832020
"""

import pandas as pd
import os
import re
import string
from pathlib import Path
import numpy as np


path_to_main_folder = Path(os.getcwd())
file_names = os.listdir(path_to_main_folder)


""" Captures and saves all the data """
for file_name in file_names:  
    if file_name.lower().endswith(".txt"):
        print(f"{file_name}")
        matrix = np.loadtxt(file_name)
        print(np.array_repr(matrix))
        np.savetxt(f'{file_name}.csv', matrix, delimiter=',')

