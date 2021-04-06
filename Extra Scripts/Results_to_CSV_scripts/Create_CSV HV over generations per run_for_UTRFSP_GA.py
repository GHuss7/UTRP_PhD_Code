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


prefix_for_each_csv_file = "UTRP_GA_ST"
spl_word = 'GA_' # initializing split word 


def count_Run_folders(path_to_folder):
    # NB only all the folders of Runs should me in the main path, otherwise errors will occur
    result_entries = os.listdir(path_to_folder) # gets the names of all the results entries
    counter = 0
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            counter = counter + 1
    return counter

def get_sens_tests_stats_from_model_runs(path_to_main_folder, nr_of_runs):
    # NB: folders should be named "Run_1" for example
    df_all_obtained_HV = pd.DataFrame()
    df_all_obtained_mean_f_1 = pd.DataFrame()
    df_all_obtained_mean_f_2 = pd.DataFrame()
    
    for i in range(nr_of_runs):
        results_file_path = path_to_main_folder / ("Run_"+str(i+1)) # sets the path
        df_generations = pd.read_csv(f"{results_file_path}/Data_generations.csv")
        
        df_all_obtained_HV["Run_"+str(i+1)] = df_generations["HV"]
        df_all_obtained_mean_f_1["Run_"+str(i+1)] = df_generations["mean_f_1"]
        df_all_obtained_mean_f_2["Run_"+str(i+1)] = df_generations["mean_f_2"]
        
    df_all_obtained_HV['Mean'] = df_all_obtained_HV.mean(axis=1)
    df_all_obtained_mean_f_1['Mean'] = df_all_obtained_mean_f_1.mean(axis=1)
    df_all_obtained_mean_f_2['Mean'] = df_all_obtained_mean_f_2.mean(axis=1)
    
    if False:
        df_all_obtained_HV.to_csv(path_to_main_folder / "Results_all_avg_HV.csv")
        df_all_obtained_mean_f_1.to_csv(path_to_main_folder / "Results_all_avg_f1.csv")
        df_all_obtained_mean_f_2.to_csv(path_to_main_folder / "Results_all_avg_f2.csv")
    
    df_runs_summary = pd.DataFrame()
    df_runs_summary["Mean_HV"] = df_all_obtained_HV["Mean"]
    df_runs_summary["Mean_f_1"] = df_all_obtained_mean_f_1["Mean"]
    df_runs_summary["Mean_f_2"] = df_all_obtained_mean_f_2["Mean"]
    df_runs_summary.to_csv(path_to_main_folder / "Results_all_runs_summary.csv")

    return df_runs_summary

path_to_main_folder = Path(os.getcwd())
nr_of_runs = count_Run_folders(path_to_main_folder)
result_entries = os.listdir(path_to_main_folder)

df_list_of_ST_results = []
parameters_list = []

df_all_cases = pd.DataFrame()

for results_folder_name in result_entries:  
    if os.path.isdir(path_to_main_folder / results_folder_name):
        nr_of_runs = count_Run_folders(path_to_main_folder / results_folder_name)
        df_runs_summary = get_sens_tests_stats_from_model_runs(path_to_main_folder / results_folder_name, nr_of_runs) # prints the runs summary
        
        """Get the substrings"""
        test_string = results_folder_name  # initializing string 
        res = test_string.partition(spl_word)[2] # partitions the string in three

        value = re.split("_", res)[-1] # gets the last number
        parameter_name = re.split("_[0-9]", res)[0] # gets the parameter name

        df_all_cases[parameter_name] = df_runs_summary["Mean_HV"]

df_all_cases.to_csv("All_cases_HV_over_generations.csv")
