# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:44:36 2019

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

# %% Timer parts to use when executing code
# Paste these two pieces of code above and below the block of code you want to time
if False:               # the False hinders the code from printing 
    runs = 100
    timeData = np.empty(runs)
    for timeCounter in range(runs):
        timerStart = timer()
        
        
        timerEnd = timer(); timeData[timeCounter] = timerEnd - timerStart
    print("Avg time: "+np.str(np.mean(timeData))+" sec\n Std dev: "+ np.str(np.std(timeData)) + " sec"); del timerStart, timerEnd,timeData # Time in seconds

# %% Range functions 
def rangeEx(a,b,c):
    # A function for creating a range between a and b and excluding a number c
    x = np.array(range(a,b))
    x = np.delete(x, np.argwhere(x == c))
    return x    
    
def rangeExArray(a,b,c):
    # A function for creating a range between a and b and excluding an array c
    x = np.array(range(a,b))
    for i in range(len(c)):
        x = np.delete(x, np.argwhere(x == c[i]))
    return x

def createDataTableFor_SA_Analysis():
    names_SA_Main = ["f1_ATT", "f2_TRT","Temperature","C_epoch_number", 
                  "L_iteration_per_epoch","A_num_accepted_moves_per_epoch","eps_num_epochs_without_accepting_solution","Route"]
    
    df_SA_Main_accepted = pd.DataFrame(columns = names_SA_Main)
    return df_SA_Main_accepted

def print_timedelta_duration(timedelta_obj):
    # prints the hours:minutes:seconds
    totsec = timedelta_obj.total_seconds()
    h = totsec//3600
    m = (totsec%3600) // 60
    sec =(totsec%3600)%60 #just for reference
    return "%d:%d:%d" %(h,m,sec)

# %% Directory functions
'''Gets all the pareto fronts and combines them'''
def group_pareto_fronts_from_model_runs(path_to_main_folder, parameters_input):
    # NB only all the folders of Runs should me in the main path, otherwise errors will occur
    result_entries = os.listdir(path_to_main_folder) # gets the names of all the results entries
    df_all_obtained_pareto_sets = pd.DataFrame()
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            results_file_path = path_to_main_folder / result_entries[i] # sets the path
            df_all_obtained_pareto_sets = df_all_obtained_pareto_sets.append(pd.read_csv(results_file_path / "Archive_Routes.csv")) # append the dataframes
    
    return df_all_obtained_pareto_sets

def group_pareto_fronts_from_model_runs_2(path_to_main_folder, parameters_input, non_dom_set_csv_file_name):
    # NB only all the folders of Runs should me in the main path, otherwise errors will occur
    result_entries = os.listdir(path_to_main_folder) # gets the names of all the results entries
    df_all_obtained_pareto_sets = pd.DataFrame()
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            results_file_path = path_to_main_folder / result_entries[i] # sets the path
            df_all_obtained_pareto_sets = df_all_obtained_pareto_sets.append(pd.read_csv(results_file_path / non_dom_set_csv_file_name)) # append the dataframes
    
    return df_all_obtained_pareto_sets

'''Extracts all the durations of the runs'''
def get_stats_from_model_runs(path_to_main_folder):
    # main folder should only contain folders of the runs
    result_entries = os.listdir(path_to_main_folder) # gets the names of all the results entries
    df_all_obtained_stats = pd.DataFrame(columns=["Run_number","Duration"])
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            results_file_path = path_to_main_folder / result_entries[i] # sets the path
            with open(results_file_path / "stats.pickle",'rb') as read_file:
                stats_dict =  pickle.load(read_file) # load the stats dictionary
            df_all_obtained_stats.loc[len(df_all_obtained_stats)] = [result_entries[i], stats_dict['duration']]

    return df_all_obtained_stats

def get_stats_from_model_runs2(path_to_main_folder, nr_of_runs):
    # NB: folders should be named "Run_1" for example
    df_all_obtained_stats = pd.DataFrame(columns=["Run_number","Duration"])
    for i in range(nr_of_runs):
        results_file_path = path_to_main_folder / ("Run_"+str(i+1)) # sets the path
        with open(results_file_path / "stats.pickle",'rb') as read_file:
            stats_dict =  pickle.load(read_file) # load the stats dictionary
        df_all_obtained_stats.loc[len(df_all_obtained_stats)] = ["Run_"+str(i+1), stats_dict['duration']]

    return df_all_obtained_stats

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
    
def count_Run_folders(path_to_folder):
    # NB only all the folders of Runs should me in the main path, otherwise errors will occur
    result_entries = os.listdir(path_to_folder) # gets the names of all the results entries
    counter = 0
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            counter = counter + 1
    
    return counter

def summarise_completed_results_avgs(path_to_results_folder):
    nr_runs = count_Run_folders(path_to_results_folder)
    get_sens_tests_stats_from_model_runs(path_to_results_folder, nr_runs)
    
def summarise_folder_with_completed_results(path_to_main_folder):
    result_entries = os.listdir(path_to_main_folder) # gets the names of all the results entries
    print()
    for result_folder_name in result_entries:
        summarise_completed_results_avgs(path_to_main_folder / result_folder_name)
    
#main_path = Path("C:/Users/17832020/OneDrive - Stellenbosch University/Desktop\FINAL RESULTS GA UTRP - Copy")
#summarise_folder_with_completed_results(main_path)

"""URTP SA Specific file management functions"""

def get_sens_tests_stats_from_UTRP_SA_runs(path_to_main_folder, nr_of_runs):
    # NB: folders should be named "Run_1" for example
    df_all_obtained_HV = pd.DataFrame()
    df_overall_results = pd.DataFrame()
    
    for i in range(nr_of_runs):
        results_file_path = path_to_main_folder / ("Run_"+str(i+1)) # sets the path
        df_generations = pd.read_csv(f"{results_file_path}/SA_Analysis.csv")
        
        df_all_obtained_HV["Run_"+str(i+1)] = df_generations["HV"]
        df_overall_results["Run_"+str(i+1)] = df_generations.iloc[-1, df_generations.columns.get_loc("HV")]
        #df_overall_results.iloc[1, df_overall_results.columns.get_loc("Run_"+str(i+1))] = len(df_generations["HV"])
        
    df_all_obtained_HV['Mean'] = df_all_obtained_HV.mean(axis=1) # calculates the mean of the HVs
    df_overall_results['Mean'] = df_overall_results.mean(axis=1) # calculates the mean of the HVs and iterations
    
    if True:
        df_all_obtained_HV.to_csv(path_to_main_folder / "Results_all_avg_HV.csv")
    
    df_runs_summary = pd.DataFrame()
    df_runs_summary["Mean_HV"] = df_all_obtained_HV["Mean"]
    df_runs_summary.to_csv(path_to_main_folder / "Results_all_runs_summary.csv")
    
    
def summarise_completed_results_UTRP_SA_avgs(path_to_results_folder):
    nr_runs = count_Run_folders(path_to_results_folder)
    get_sens_tests_stats_from_UTRP_SA_runs(path_to_results_folder, nr_runs)
    
def summarise_folder_with_completed_results_UTRP_SA(path_to_main_folder):
    result_entries = os.listdir(path_to_main_folder) # gets the names of all the results entries
    print()
    for result_folder_name in result_entries:
        summarise_completed_results_UTRP_SA_avgs(path_to_main_folder / result_folder_name)    

#main_path = Path("C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS/Results/Results_Mandl_UTRP")
#summarise_folder_with_completed_results_UTRP_SA(main_path)
    
# %% Add rows to dataframes fast
'''Add a row to a dataframe with a dictionary fast'''
def add_row_to_df(dataframe,dict_entry):
    dataframe.append(dict_entry)
    
    
# %% List into a list of lists 
"""List comprehension is an efficient approach as it doesn’t make use of extra space. 
For each element ‘el’ in list, it simply appends [el] to the output list."""
def extractDigits(lst): 
    return [[el] for el in lst] 

# Generate labels for UTRP GA DFs
def generate_data_analysis_labels_routes(num_objectives, num_variables):
    # format 'x' with number
    label_names = []
    for i in range(num_objectives):
        label_names.append("f"+str(i))
    for j in range(num_variables):
        label_names.append("R"+str(j))
    return label_names

# Create DF from Route population for UTRP GA
def create_df_from_pop(pop_generations):
    df_pop = pd.DataFrame()
    df_pop = df_pop.assign(f_1 = pop_generations[:,0],
                           f_2 = pop_generations[:,1],
                           routes = pop_generations[:,2],
                           generation = pop_generations[:,3]
                           )
    
    df_pop['f_1'] = df_pop['f_1'].astype(float)
    df_pop['f_2'] = df_pop['f_2'].astype(float)
    
    return df_pop

# %% Print progress bar
from time import sleep

def progress(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)

# for i in range(101):
#     progress(i)
#     sleep(0.1)