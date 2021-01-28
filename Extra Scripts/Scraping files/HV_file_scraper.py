# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:50:23 2020

@author: 17832020
"""

import pandas as pd
import os
import re
import string
from pathlib import Path

path_to_main_folder = Path(os.getcwd())
result_entries = os.listdir(path_to_main_folder)
results_captured = pd.DataFrame(columns=["Folder name", "HV"])
file_name = "Stats_overall.csv"

""" Captures and saves all the data """
for results_folder_name in result_entries:  
    if os.path.isdir(path_to_main_folder / results_folder_name):
        if os.path.exists(path_to_main_folder / results_folder_name / file_name):
            stats_dataframe = pd.read_csv(path_to_main_folder / results_folder_name / file_name)
            results_captured.loc[len(results_captured)] = [results_folder_name, stats_dataframe.iloc[5,1]]

results_captured.to_csv(path_to_main_folder / f"HV_per_folder.csv")
