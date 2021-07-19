# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 18:09:49 2021

@author: 17832020
"""
import pandas as pd
import os
import re
import string
import json
from pathlib import Path
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))

# Get all the results folders:
all_result_folders = os.listdir(dir_path)

# Set folders to do the operation on:
apply_list = ['Results_1_1_Mandl6_GA_Initial_solutions',
     'Results_1_2_Mumford0_GA_Initial_solutions',
     'Results_1_3_Mumford1_GA_Initial_solutions',
     'Results_2_1_Mandl6_GA_Crossover',
     'Results_2_2_Mumford0_GA_Crossover',
     'Results_2_3_Mumford1_GA_Crossover',
     'Results_3_1_Mandl6_GA_Mutations',
     'Results_3_2_Mumford0_GA_Mutations',
     'Results_3_3_Mumford1_GA_Mutations',
     'Results_4_1_Mandl6_GA_Update_mut_ratio',
     'Results_4_2_Mumford0_GA_Update_mut_ratio',
     'Results_4_3_Mumford1_GA_Update_mut_ratio']

# Ensure the directory is set to the file location

for apply_folder in apply_list: 
    current_wd = dir_path+'\\'+apply_folder
    os.chdir(current_wd)
    
    #$ Take note: This runs the script in the folder itself
    # x = subprocess.call(f"python {dir_path}\Post_analysis_UTRP_GA_results.py -dir '{current_wd}'") # -interaction nonstopmode -shel-escape
    command_str = f'python Post_analysis_UTRP_GA_results.py -dir "{current_wd}"'
    x = subprocess.call(command_str) # -interaction nonstopmode -shel-escape

    if x != 0:
    	print(f'Exit-code not 0, check result! [{apply_folder}]')
    else:
    	#os.system('start box_plots.pdf')+
        print(f"Success! [{apply_folder}]")
    