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
import copy

prefix_for_each_csv_file = "UTRP_GA_Summary"
spl_word = 'GA_' # initializing split word 

def count_Run_folders(path_to_folder):
    # NB only all the folders of Runs should me in the main path, otherwise errors will occur
    result_entries = os.listdir(path_to_folder) # gets the names of all the results entries
    counter = 0
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            counter = counter + 1
    return counter


def get_sens_tests_stats_from_UTRP_GA_runs(path_to_main_folder):
    # NB: folders should be named "Run_1" for example
    nr_of_runs = count_Run_folders(path_to_main_folder)
    df_all_obtained_HV = pd.DataFrame()
    df_overall_results = pd.DataFrame(columns=["Run_nr","HV"])
    
    for i in range(nr_of_runs):
        results_file_path = path_to_main_folder / ("Run_"+str(i+1)) # sets the path
        df_all_results_per_run = pd.read_csv(f"{results_file_path}/Data_generations.csv")
        
        df_all_obtained_HV["Run_"+str(i+1)] = df_all_results_per_run["HV"]
        df_overall_results.loc[len(df_overall_results)] = ["Run_"+str(i+1),
                                                           df_all_results_per_run.iloc[-1, df_all_results_per_run.columns.get_loc("HV")]
                                                           ]
        
    df_HV_description = df_overall_results[["HV"]].describe().transpose()
    df_HV_description.columns = ["count", "mean", "std", "min", "lq", "med", "uq", "max"]
  
    IQR = abs((df_HV_description["uq"] - df_HV_description["lq"]).values)[0]
    
    """Calculate stats without outliers""" 
    df_results_cons_outliers = copy.deepcopy(df_overall_results)
    outlier_list = []
    
    for i_entry in range(len(df_results_cons_outliers["HV"])-1,-1,-1):
        if df_results_cons_outliers["HV"].iloc[i_entry] < df_HV_description["lq"].values[0] - 1.5*IQR or df_results_cons_outliers["HV"].iloc[i_entry] > 1.5*IQR + df_HV_description["uq"].values[0]:
            outlier_list.append(df_results_cons_outliers["HV"].iloc[i_entry])
            df_results_cons_outliers = df_results_cons_outliers.drop([i_entry]) 
    
    df_HV_description_outliers = df_results_cons_outliers[["HV"]].describe().transpose()
    df_HV_description_outliers.columns = ["count", "mean", "std", "min", "lq", "med", "uq", "max"]
    df_HV_description_outliers_included = copy.deepcopy(df_HV_description_outliers)
    df_HV_description_outliers_included["outliers"] = ", ".join(map(str, outlier_list))
  
    df_all_obtained_HV['Mean'] = df_all_obtained_HV.mean(axis=1) # calculates the mean of the HVs
    df_overall_results.loc[len(df_overall_results)] = df_overall_results.mean(axis=0) # calculates the mean of the HVs and iterations
    df_overall_results.iloc[-1,0] = "Means"
    
    if False:
        df_all_obtained_HV.to_csv(path_to_main_folder / "Results_all_avg_HV.csv")
    
    df_runs_summary = pd.DataFrame()
    df_runs_summary["Mean_HV"] = df_all_obtained_HV["Mean"]
    # df_runs_summary.to_csv(path_to_main_folder / "Results_all_runs_summary.csv")
    df_HV_description.to_csv(path_to_main_folder / "Results_description_HV_no_outliers.csv")
    df_HV_description_outliers.to_csv(path_to_main_folder / "Results_description_HV_with_outliers.csv")
    df_HV_description_outliers_included.to_csv(path_to_main_folder / "Results_description_HV.csv")
    df_overall_results.to_csv(path_to_main_folder / "Overall_results.csv")


path_to_main_folder = Path(os.getcwd())

result_entries = os.listdir(path_to_main_folder)
         
for results_folder_name in result_entries:  
    if os.path.isdir(path_to_main_folder / results_folder_name):
        get_sens_tests_stats_from_UTRP_GA_runs(path_to_main_folder / results_folder_name)



