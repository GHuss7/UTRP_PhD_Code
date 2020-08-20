# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:48:53 2020

@author: 17832020
"""
import pandas as pd
import os
import re
import string
from pathlib import Path

def count_Run_folders(path_to_folder):
    # NB only all the folders of Runs should me in the main path, otherwise errors will occur
    result_entries = os.listdir(path_to_folder) # gets the names of all the results entries
    counter = 0
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            counter = counter + 1
    return counter


path_to_main_folder = Path(os.getcwd())
nr_of_runs = count_Run_folders(path_to_main_folder)
result_entries = os.listdir(path_to_main_folder)

prefix_for_each_csv_file = "Run_HVs_per_iteration"
df_list_of_results = []
parameters_list = []
parameters_max_length = 0 # list for storing the max run lenghts per parameter
                 
""" Determines all the max lengths for data """
for results_folder_name in result_entries:  
    if os.path.isdir(path_to_main_folder / results_folder_name):
  
        """Capture all the results into dataframes"""
        df_results_read = pd.read_csv(f"{path_to_main_folder / results_folder_name}/SA_Analysis.csv")
        
        if len(df_results_read) > parameters_max_length:
            parameters_max_length = len(df_results_read)
                        
            
df_list_of_results = []
parameters_list = []
df_list_of_results.append(pd.DataFrame())

""" Captures and saves all the data """
for results_folder_name in result_entries:  
    if os.path.isdir(path_to_main_folder / results_folder_name):
  
        """Capture all the results into dataframes"""
        df_results_read = pd.read_csv(f"{path_to_main_folder / results_folder_name}/SA_Analysis.csv")
        
        if parameters_max_length != len(df_results_read):
            df_blank_to_append = pd.concat([pd.DataFrame([""], columns=['Mean_HV']) for i in range(parameters_max_length - len(df_results_read))], ignore_index=True)
            df_results_read = df_results_read.append(df_blank_to_append, ignore_index=True)

        df_list_of_results[0][results_folder_name] = df_results_read["HV"]        


"""Print dataframes as .csv files"""
df_list_of_results[0].to_csv(path_to_main_folder / f"{prefix_for_each_csv_file}.csv")




