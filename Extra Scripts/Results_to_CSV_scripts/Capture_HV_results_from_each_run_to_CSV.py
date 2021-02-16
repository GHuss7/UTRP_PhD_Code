# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:22:18 2020

@author: 17832020
"""
# %% Import libraries
import os
import re
import json
import pickle
import numpy as np
import igraph as ig
import math
import pandas as pd
from collections import deque, namedtuple
from timeit import default_timer as timer
import datetime
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

#def get_sens_tests_stats_from_UTRP_SA_runs(path_to_main_folder, nr_of_runs):
# NB: folders should be named "Run_1" for example
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

    
df_Total_iterations_description = pd.DataFrame(columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max"])
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
df_overall_results.loc[len(df_overall_results)] = df_overall_results.mean(axis=0) # calculates the mean of the HVs and iterations
df_overall_results.iloc[-1,0] = "Means"

if False:
    df_all_obtained_HV.to_csv(path_to_main_folder / "Results_all_avg_HV.csv")

df_runs_summary = pd.DataFrame()
df_runs_summary["Mean_HV"] = df_all_obtained_HV["Mean"]
df_runs_summary.to_csv(path_to_main_folder / "Results_all_runs_summary.csv")
df_HV_description.to_csv(path_to_main_folder / "Results_description_HV.csv")
df_Total_iterations_description.to_csv(path_to_main_folder / "Results_description_Tot_iter.csv")
df_overall_results.to_csv(path_to_main_folder / "Overall_results.csv")
