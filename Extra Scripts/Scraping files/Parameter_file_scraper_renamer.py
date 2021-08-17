# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:50:23 2020

@author: 17832020
"""

import pandas as pd
import os
import re
import string
import json
from pathlib import Path


file_name = "Parameters/Decisions.json"
parameter_name = "mut_update_func"
rename_bool = False
path_to_main_folder = Path(os.getcwd())
result_entries = os.listdir(path_to_main_folder)
results_captured = pd.DataFrame(columns=["Folder name", parameter_name])

""" Captures and saves all the data """
for results_folder_name in result_entries:  
    if os.path.isdir(path_to_main_folder / results_folder_name):
        full_path = path_to_main_folder / results_folder_name / file_name
        if os.path.exists(full_path):
            Decisions = json.load(open(full_path))
            parameter = Decisions[parameter_name]
            results_captured.loc[len(results_captured)] = [results_folder_name, parameter]
            if rename_bool: os.rename(path_to_main_folder/results_folder_name, path_to_main_folder/results_folder_name.replace('Mutations',f'Mutations_{Decisions[parameter_name]}'))

results_captured.to_csv(path_to_main_folder / f"{parameter_name}_per_folder.csv")

