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
from PyPDF2 import PdfFileMerger, PdfFileReader

dir_path = os.path.dirname(os.path.realpath(__file__))

# Get all the results folders:
all_result_folders = os.listdir(dir_path)

dir_prefix = "C:/Users/gunth/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS/Results"

# Set folders to do the operation on:
apply_list = [
                #"Results_7_5_Mumford3_GA_Pop_size/7_5_Mumford3_GA_Pop_size_20210723_134827 GA_population_size_500",
                #"Results_7_5_Mumford3_GA_Pop_size/7_5_Mumford3_GA_Pop_size_20210723_111033 GA_population_size_600",
                #"Results_8_5_Mumford3_GA_Crossover_prob/8_5_Mumford3_GA_Crossover_prob_20210725_214537 GA_crossover_probability_0.4",
                #"Results_8_5_Mumford3_GA_Crossover_prob/8_5_Mumford3_GA_Crossover_prob_20210726_072740 GA_crossover_probability_0.5",
                #"Results_9_5_Mumford3_GA_Mut_prob/9_5_Mumford3_GA_Mut_prob_20210727_225424 GA_mutation_probability_0.95",
                #"Results_9_5_Mumford3_GA_Mut_prob/9_5_Mumford3_GA_Mut_prob_20210728_083726 GA_mutation_probability_1",
                #"Results_10_3_Mumford1_GA_Long_run/10_3_Mumford1_GA_Long_run_20210731_170218 GA_Long_run",
            ]

# Ensure the directory is set to the file location

for apply_folder in apply_list: 
    current_wd = dir_prefix+'/'+apply_folder # This directs the script to run in the relevant folders
    
    # Take note: This runs the script in the folder itself
    # x = subprocess.call(f"python {dir_path}\Post_analysis_UTRP_GA_results.py -dir '{current_wd}'") # -interaction nonstopmode -shel-escape
    command_str = f'python Recon_runs_UTRP_GA.py -dir "{current_wd}"'
    x = subprocess.call(command_str) # -interaction nonstopmode -shel-escape

    if x != 0:
        print(f'Exit-code not 0, check result! [{apply_folder}]')
        
    else:	
        print(f"Success! [{apply_folder}]")
        os.chdir(dir_path)