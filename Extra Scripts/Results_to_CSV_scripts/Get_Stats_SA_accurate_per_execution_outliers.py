# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:46:04 2020

@author: 17832020
"""
import pandas as pd
import os
import re
import string
from pathlib import Path
import numpy as np
import copy


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


#def get_sens_tests_stats_from_UTRP_SA_runs(path_to_main_folder):
# NB: folders should be named "Run_1" for example
nr_of_runs = count_Run_folders(path_to_main_folder)
df_all_obtained_HV = pd.DataFrame()
df_overall_results = pd.DataFrame(columns=["Run_nr","HV","Epochs","Poor_epochs","Total_iterations"])

for i in range(nr_of_runs):
    results_file_path = path_to_main_folder / ("Run_"+str(i+1)) # sets the path
    df_all_results_per_run = pd.read_csv(f"{results_file_path}/SA_Analysis.csv")
    
    df_all_obtained_HV["Run_"+str(i+1)] = df_all_results_per_run["HV"]
    df_overall_results.loc[len(df_overall_results)] = ["Run_"+str(i+1),
                                                       df_all_results_per_run.iloc[-1, df_all_results_per_run.columns.get_loc("HV")],
                                                       df_all_results_per_run.iloc[-1, df_all_results_per_run.columns.get_loc("C_epoch_number")],
                                                       df_all_results_per_run.iloc[-1, df_all_results_per_run.columns.get_loc("eps_num_epochs_without_accepting_solution")],
                                                       len(df_all_results_per_run)
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
df_HV_description_outliers["outliers"] = ", ".join(map(str, outlier_list))
 
df_Total_iterations_description = pd.DataFrame(columns=["count", "mean", "std", "min", "lq", "med", "uq", "max"])
percentiles = np.percentile(df_overall_results[["Total_iterations"]], [0, 25, 50, 75, 100])

df_Total_iterations_description.loc[len(df_Total_iterations_description)] = [len(df_overall_results[["Total_iterations"]]),
                                                                                df_overall_results[["Total_iterations"]].mean()[0],
                                                                                df_overall_results[["Total_iterations"]].std()[0],
                                                                                percentiles[0],
                                                                                percentiles[1],
                                                                                percentiles[2],
                                                                                percentiles[3],
                                                                                percentiles[4]]

df_all_obtained_HV['Mean'] = df_all_obtained_HV.mean(axis=1) # calculates the mean of the HVs

if False:
    df_all_obtained_HV.to_csv(path_to_main_folder / "Results_all_avg_HV.csv")

df_runs_summary = pd.DataFrame()
df_runs_summary["Mean_HV"] = df_all_obtained_HV["Mean"]
# df_runs_summary.to_csv(path_to_main_folder / "Results_all_runs_summary.csv")
df_HV_description.to_csv(path_to_main_folder / "Results_description_HV_no_outliers.csv")
df_HV_description_outliers.to_csv(path_to_main_folder / "Results_description_HV.csv")
df_Total_iterations_description.to_csv(path_to_main_folder / "Results_description_Tot_iter.csv")
df_overall_results.to_csv(path_to_main_folder / "Overall_results.csv")