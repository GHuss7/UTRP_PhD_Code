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

file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = file_path
dir_path = "C:/Users/gunth/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS/Results"
if not os.path.exists(dir_path):
    dir_path = "C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS/Results"
    assert os.path.exists(dir_path)

stacked_chart_bool = False

# Get all the results folders:
all_result_folders = os.listdir(dir_path)

# Set folders to do the operation on:
apply_list = [#'Results_1_1_Mandl6_GA_Initial_solutions',
     # 'Results_1_2_Mumford0_GA_Initial_solutions',
     # 'Results_1_3_Mumford1_GA_Initial_solutions',
     # 'Results_2_1_Mandl6_GA_Crossover',
     # 'Results_2_2_Mumford0_GA_Crossover',
     # 'Results_2_3_Mumford1_GA_Crossover',
     # 'Results_3_1_Mandl6_GA_Mutations',
     # 'Results_3_2_Mumford0_GA_Mutations',
     # 'Results_3_3_Mumford1_GA_Mutations',
     # 'Results_4_1_Mandl6_GA_Update_mut_ratio',
     # 'Results_4_2_Mumford0_GA_Update_mut_ratio',
     # 'Results_4_3_Mumford1_GA_Update_mut_ratio',
     # 'Results_5_1_Mandl6_GA_repair_func',
     # 'Results_5_2_Mumford0_GA_repair_func',
     # 'Results_5_3_Mumford1_GA_repair_func',
     # 'Results_6_1_Mandl6_GA_Mut_threshold',
     # 'Results_6_2_Mumford0_GA_Mut_threshold',
     # 'Results_6_3_Mumford1_GA_Mut_threshold',
     # 'Results_7_1_Mandl6_GA_Pop_size',
     # 'Results_7_2_Mumford0_GA_Pop_size',
     # 'Results_7_3_Mumford1_GA_Pop_size',
     # 'Results_7_4_Mumford2_GA_Pop_size',
     # 'Results_7_5_Mumford3_GA_Pop_size',
     # 'Results_8_1_Mandl6_GA_Crossover_prob', #34
    # 'Results_8_2_Mumford0_GA_Crossover_prob', #35
    # 'Results_8_3_Mumford1_GA_Crossover_prob', #36
    # 'Results_8_4_Mumford2_GA_Crossover_prob', #37
    # 'Results_8_5_Mumford3_GA_Crossover_prob', #38
    # 'Results_9_1_Mandl6_GA_Mut_prob', #39
    # 'Results_9_2_Mumford0_GA_Mut_prob', #40
    # 'Results_9_3_Mumford1_GA_Mut_prob', #41
    # 'Results_9_4_Mumford2_GA_Mut_prob', #42
    # 'Results_9_5_Mumford3_GA_Mut_prob', #43
    # 'Results_21_1_Mandl6_SA_Initial_solutions', #11
    # 'Results_21_2_Mumford0_SA_Initial_solutions', #12
    # 'Results_21_3_Mumford1_SA_Initial_solutions', #13
    # 'Results_22_1_Mandl6_SA_Mut_update', #14
    # 'Results_22_2_Mumford0_SA_Mut_update', #15
    # 'Results_22_3_Mumford1_SA_Mut_update', #16
    # 'Results_23_1_Mandl6_SA_Mutations', #17
    # 'Results_23_2_Mumford0_SA_Mutations', #18
    # 'Results_23_3_Mumford1_SA_Mutations', #19
    # 'Results_24_1_Mandl6_SA_Mutations_more', #20
    # 'Results_24_2_Mumford0_SA_Mutations_more', #21
    # 'Results_24_3_Mumford1_SA_Mutations_more', #22

    'Results_33_1_Mandl6_SA_ALL_param', #47
    'Results_33_2_Mumford0_SA_ALL_param', #48
    'Results_33_3_Mumford1_SA_ALL_param', #49


     ]

# Ensure the directory is set to the file location
# Call the PdfFileMerger
mergedObject = PdfFileMerger() # create the object

for apply_folder in apply_list: 
    current_wd = dir_path+'\\'+apply_folder # This directs the script to run in the relevant folders
    #os.chdir(current_wd)
    os.chdir(file_path) # ensures the results folder' script is used
    
    # Take note: This runs the script in the folder itself
    # x = subprocess.call(f"python {file_path}\Post_analysis_UTRP_GA_results.py -dir '{current_wd}'")
    
    if apply_folder.find('_GA_') != -1:
        meta_type = 'GA'
    elif apply_folder.find('_SA_') != -1:
        meta_type = 'SA'
    else:
        meta_type = 'SA'
        print(f"ERROR! Not SA or GA folder! Setting to {meta_type}")
        

    command_str = f'python Post_analysis_UTRP_GA_results.py -dir "{current_wd}" -mt "{meta_type}" -sc {stacked_chart_bool}'
    x = subprocess.call(command_str)

    if x != 0:
        print(f'Exit-code not 0, check result! [{apply_folder}]')
        
    else:
    	#os.system('start box_plots.pdf')+
        print(f"Success! [{apply_folder}]")
        os.chdir(current_wd)
        mergedObject.append(PdfFileReader('box_plots.pdf', 'rb'))
        
#%% Save merged pdf

 
# Write all the files into a file which is named as shown below
os.chdir(dir_path)
mergedObject.write('box_plots_all.pdf')
os.system('start box_plots_all.pdf')