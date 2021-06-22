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
from datetime import timedelta
from pathlib import Path
import copy



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

#%% Timing functions
def print_timedelta_duration(timedelta_obj):
    # prints the hours:minutes:seconds
    totsec = timedelta_obj.total_seconds()
    h = totsec//3600
    m = (totsec%3600) // 60
    sec =(totsec%3600)%60 #just for reference
    return "%d:%d:%d" %(h,m,sec)

def time_projection(seconds_per_iteration, total_iterations, t_now=False, return_objs=False, print_iter_info=False):
    def get_time_objects(totsec):
        h = totsec//3600
        m = (totsec%3600) // 60
        sec =(totsec%3600)%60 #just for reference
        return h, m , sec
    
    total_estimated_seconds = seconds_per_iteration * total_iterations
    t_additional = timedelta(seconds=total_estimated_seconds)
    dur_h, dur_m, dur_sec = get_time_objects(t_additional.seconds)
    
    # Get time now
    if not t_now:
        t_now = datetime.now()
    date_time_start = t_now.strftime("%a, %d %b, %H:%M:%S")
    
    # Determine expected time
    t_expected = t_now + t_additional
    date_time_due = t_expected.strftime("%a, %d %b, %H:%M:%S")
    
    print(f"Start:    {date_time_start}")
    print(f"Due date: {date_time_due}")
    print(f"Duration: {t_additional.days} days, {dur_h} hrs, {dur_m} min, {dur_sec} sec")
    
    if print_iter_info:
        print(f"Total iterations: {total_iterations} at {seconds_per_iteration:.2f} sec/it")
       
    if return_objs:
        return date_time_start, date_time_due

# Deterime total iterations
def determine_total_iterations(main_problem, multiply_factor=1):
    len_pop = main_problem.problem_GA_parameters.population_size
    num_gen = main_problem.problem_GA_parameters.generations
    num_runs = main_problem.problem_GA_parameters.number_of_runs

    total_iterations = (len_pop + len_pop * num_gen * multiply_factor) * num_runs
    return total_iterations
    
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
    df_all_obtained_stats = pd.DataFrame(columns=["Run_number","Duration", "HV Obtained"])
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            results_file_path = path_to_main_folder / result_entries[i] # sets the path
            with open(results_file_path / "stats.pickle",'rb') as read_file:
                stats_dict =  pickle.load(read_file) # load the stats dictionary
            df_all_obtained_stats.loc[len(df_all_obtained_stats)] = [result_entries[i], stats_dict['duration'], stats_dict['HV obtained']]

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

def get_mutation_stats_from_model_runs(path_to_main_folder):
    # main folder should only contain folders of the runs
    result_entries = os.listdir(path_to_main_folder) # gets the names of all the results entries
    run_counter = 0
    min_len = 0

    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            results_file_path = path_to_main_folder / result_entries[i] # sets the path 
            
            len_df = len(pd.read_csv(results_file_path / 'Mut_ratios.csv'))

            if min_len == 0:
               min_len = len_df
            else:
                if len_df < min_len:
                    min_len = len_df


    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            results_file_path = path_to_main_folder / result_entries[i] # sets the path 
            
            df_mut_ratios = pd.read_csv(results_file_path / 'Mut_ratios.csv')
            
            if run_counter == 0:
                mut_ratios = np.array(df_mut_ratios.values)[:min_len,:]
                run_counter += 1
                
            else:
                mut_ratios = mut_ratios + np.array(df_mut_ratios.values)[:min_len,:]
                run_counter += 1
    
    # average each array
    mut_ratios = mut_ratios/ run_counter

    df_all_mut_ratios = pd.DataFrame(columns=df_mut_ratios.columns, data=mut_ratios)
    df_all_mut_ratios_end_smoothed = exp_smooth_df(df_all_mut_ratios, alpha=0.1, beta=0.1, n=100) 
            
    df_list = [df_all_mut_ratios, df_all_mut_ratios_end_smoothed]
    df_names = ["Avg_mut_ratios", "Smoothed_avg_mut_ratios"]
    
    return df_list, df_names
    
     

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

def get_sens_tests_stats_from_UTRP_SA_runs(path_to_main_folder):
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
    
def capture_all_runs_HV_over_iterations_from_UTRP_SA(path_to_main_folder, prefix_for_each_csv_file="Run_HVs_per_iteration"):
    nr_of_runs = count_Run_folders(path_to_main_folder)
    result_entries = os.listdir(path_to_main_folder)
    
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

# %% NSGA II results capture
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
  
    df_all_obtained_HV['Mean'] = df_all_obtained_HV.mean(axis=1) # calculates the mean of the HVs
    
    if False:
        df_all_obtained_HV.to_csv(path_to_main_folder / "Results_all_avg_HV.csv")
    
    df_runs_summary = pd.DataFrame()
    df_runs_summary["Mean_HV"] = df_all_obtained_HV["Mean"]
    # df_runs_summary.to_csv(path_to_main_folder / "Results_all_runs_summary.csv")
    df_HV_description.to_csv(path_to_main_folder / "Results_description_HV.csv")
    df_overall_results.to_csv(path_to_main_folder / "Overall_results.csv")

def get_sens_tests_stats_from_UTFSP_GA_runs(path_to_main_folder):
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
  
    df_all_obtained_HV['Mean'] = df_all_obtained_HV.mean(axis=1) # calculates the mean of the HVs
    
    if False:
        df_all_obtained_HV.to_csv(path_to_main_folder / "Results_all_avg_HV.csv")
    
    df_runs_summary = pd.DataFrame()
    df_runs_summary["Mean_HV"] = df_all_obtained_HV["Mean"]
    # df_runs_summary.to_csv(path_to_main_folder / "Results_all_runs_summary.csv")
    df_HV_description.to_csv(path_to_main_folder / "Results_description_HV.csv")
    df_overall_results.to_csv(path_to_main_folder / "Overall_results.csv")

# %% Add rows to dataframes fast (list of dictionaries)
'''Add a row to a dataframe with a dictionary fast'''
def add_row_to_df(dataframe,dict_entry):
    dataframe.append(dict_entry)
      
def add_UTRFSP_analysis_data(pop_1, UTRFSP_problem_1, data_for_analysis=False):
    """
    

    Parameters
    ----------
    pop_1 : PopulationRoutesFreq object
        Contains all the variables and objective function values.
    UTRFSP_problem_1 : UTRFSP_problem object
        Object containing the main UTRFSP details.
    data_for_analysis : False or pandas.DataFrame, optional
        The dataframe to contain the data analysis. If False, this is the fist 
        time the Dataframe is created and then only add the first pop_size 
        population of solutions. The default is False.

    Returns
    -------
    data_for_analysis : pandas.DataFrame
        The appended DataFrame.

    """
    len_pop = len(pop_1.objectives)
    pop_size = UTRFSP_problem_1.problem_GA_parameters.population_size
    
    if isinstance(data_for_analysis, pd.DataFrame):
        for index_i in range(pop_size, len_pop):
            data_row = [pop_1.variables_routes_str[index_i]]
            data_row.extend(list(pop_1.variables_freq[index_i,:]))
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
                 
    else:
        columns_list = ["R_x", "F_3", "F_4", "Rank"]
        columns_list[1:1] = ["f_"+str(x) for x in range(UTRFSP_problem_1.problem_constraints.con_r)]
        data_for_analysis = pd.DataFrame(columns=columns_list)
        
        for index_i in range(pop_size):
            data_row = [pop_1.variables_routes_str[index_i]]
            data_row.extend(list(pop_1.variables_freq[index_i,:]))
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
                    
    return data_for_analysis

def add_UTRP_analysis_data_with_generation_nr(pop_1, UTRP_problem_1, generation_num=0, data_for_analysis=False):
    """
    

    Parameters
    ----------
    pop_1 : PopulationRoutesFreq object
        Contains all the variables and objective function values.
    UTRFSP_problem_1 : UTRFSP_problem object
        Object containing the main UTRFSP details.
    data_for_analysis : False or pandas.DataFrame, optional
        The dataframe to contain the data analysis. If False, this is the fist 
        time the Dataframe is created and then only add the first pop_size 
        population of solutions. The default is False.

    Returns
    -------
    data_for_analysis : pandas.DataFrame
        The appended DataFrame.

    """
    len_pop = len(pop_1.objectives)
    pop_size = UTRP_problem_1.problem_GA_parameters.population_size
    
    if isinstance(data_for_analysis, pd.DataFrame):
        for index_i in range(pop_size, len_pop):
            data_row = [pop_1.variables_str[index_i]]
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend([generation_num])
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
                 
    else:
        columns_list = ["R_x", "f_1", "f_2", "Generation", "Rank"]
        data_for_analysis = pd.DataFrame(columns=columns_list)
        
        for index_i in range(pop_size):
            data_row = [pop_1.variables_str[index_i]]
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend([generation_num])
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
                    
    return data_for_analysis

def add_UTRP_analysis_data_with_generation_nr_ld(pop_1, UTRP_problem_1, generation_num=0, data_for_analysis=False):
    """
    

    Parameters
    ----------
    pop_1 : PopulationRoutesFreq object
        Contains all the variables and objective function values.
    UTRFSP_problem_1 : UTRFSP_problem object
        Object containing the main UTRFSP details.
    data_for_analysis : False or pandas.DataFrame, optional
        The dataframe to contain the data analysis. If False, this is the fist 
        time the Dataframe is created and then only add the first pop_size 
        population of solutions. The default is False.

    Returns
    -------
    data_for_analysis : list of dictionaries
        The appended DataFrame.

    """
    len_pop = len(pop_1.objectives)
    pop_size = UTRP_problem_1.problem_GA_parameters.population_size                
    
    if isinstance(data_for_analysis, list):
        zipped = zip(pop_1.variables_str[pop_size:len_pop],
                     pop_1.objectives[pop_size:len_pop,0], 
                     pop_1.objectives[pop_size:len_pop,1], 
                     [generation_num]*pop_size, 
                     pop_1.rank[pop_size:len_pop])
        
        data_for_analysis.extend([{"R_x":r, "f_1":f1, "f_2":f2, "Generation":gen, "Rank":rank[0]} for r, f1, f2, gen, rank in zipped])
    
    else:
        zipped = zip(pop_1.variables_str[:pop_size],
                     pop_1.objectives[:pop_size,0], 
                     pop_1.objectives[:pop_size,1], 
                     [generation_num]*pop_size, 
                     pop_1.rank[:pop_size])
        
        data_for_analysis = [{"R_x":r, "f_1":f1, "f_2":f2, "Generation":gen, "Rank":rank[0]} for r, f1, f2, gen, rank in zipped]
                 
    return data_for_analysis

def add_UTRFSP_analysis_data_with_generation_nr(pop_1, UTRFSP_problem_1, generation_num=0, data_for_analysis=False):
    """
    

    Parameters
    ----------
    pop_1 : PopulationRoutesFreq object
        Contains all the variables and objective function values.
    UTRFSP_problem_1 : UTRFSP_problem object
        Object containing the main UTRFSP details.
    data_for_analysis : False or pandas.DataFrame, optional
        The dataframe to contain the data analysis. If False, this is the fist 
        time the Dataframe is created and then only add the first pop_size 
        population of solutions. The default is False.

    Returns
    -------
    data_for_analysis : pandas.DataFrame
        The appended DataFrame.

    """
    len_pop = len(pop_1.objectives)
    pop_size = UTRFSP_problem_1.problem_GA_parameters.population_size
    
    if isinstance(data_for_analysis, pd.DataFrame):
        for index_i in range(pop_size, len_pop):
            data_row = [pop_1.variables_routes_str[index_i]]
            data_row.extend(list(pop_1.variables_freq[index_i,:]))
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend([generation_num])
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
                 
    else:
        columns_list = ["R_x", "F_3", "F_4", "Generation", "Rank"]
        columns_list[1:1] = ["f_"+str(x) for x in range(UTRFSP_problem_1.problem_constraints.con_r)]
        data_for_analysis = pd.DataFrame(columns=columns_list)
        
        for index_i in range(pop_size):
            data_row = [pop_1.variables_routes_str[index_i]]
            data_row.extend(list(pop_1.variables_freq[index_i,:]))
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend([generation_num])
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
                    
    return data_for_analysis

def add_UTRP_pop_generations_data(pop_1, UTRP_problem_1, generation_num, data_for_analysis=False):
    """
    

    Parameters
    ----------
    pop_1 : PopulationRoutesFreq object
        Contains all the variables and objective function values.
    UTRFSP_problem_1 : UTRFSP_problem object
        Object containing the main UTRFSP details.
    data_for_analysis : False or pandas.DataFrame, optional
        The dataframe to contain the data analysis. If False, this is the fist 
        time the Dataframe is created and then only add the first pop_size 
        population of solutions. The default is False.

    Returns
    -------
    data_for_analysis : pandas.DataFrame
        The appended DataFrame.

    """
    pop_size = UTRP_problem_1.problem_GA_parameters.population_size
    
    if isinstance(data_for_analysis, pd.DataFrame):
        for index_i in range(pop_size):
            data_row = [pop_1.variables_str[index_i]]
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend([generation_num])
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
            
    else:
        columns_list = ["R_x", "f_1", "f_2","Generation", "Rank"]
        data_for_analysis = pd.DataFrame(columns=columns_list)
        
        for index_i in range(pop_size):
            data_row = [pop_1.variables_str[index_i]]
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend([generation_num])
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
            
    return data_for_analysis

def add_UTRP_pop_generations_data_ld(pop_1, UTRP_problem_1, generation_num, data_for_analysis=False):
    """
    This allows for faster DataFrame construction

    Parameters
    ----------
    pop_1 : PopulationRoutesFreq object
        Contains all the variables and objective function values.
    UTRFSP_problem_1 : UTRFSP_problem object
        Object containing the main UTRFSP details.
    data_for_analysis : False or pandas.DataFrame, optional
        The dataframe to contain the data analysis. If False, this is the fist 
        time the Dataframe is created and then only add the first pop_size 
        population of solutions. The default is False.

    Returns
    -------
    data_for_analysis : list of dictionaries
        The appended DataFrame.

    """
    pop_size = UTRP_problem_1.problem_GA_parameters.population_size
    zipped = zip(pop_1.variables_str[:pop_size],pop_1.objectives[:pop_size,0], pop_1.objectives[:pop_size,1], [generation_num]*pop_size, pop_1.rank[:pop_size])

    if isinstance(data_for_analysis, list):
        data_for_analysis.extend([{"R_x":r, "f_1":f1, "f_2":f2, "Generation":gen, "Rank":rank[0]} for r, f1, f2, gen, rank in zipped])
    
    else:
        data_for_analysis = [{"R_x":r, "f_1":f1, "f_2":f2, "Generation":gen, "Rank":rank[0]} for r, f1, f2, gen, rank in zipped]
    
    return data_for_analysis
    
def add_UTRFSP_pop_generations_data(pop_1, UTRFSP_problem_1, generation_num, data_for_analysis=False):
    """
    

    Parameters
    ----------
    pop_1 : PopulationRoutesFreq object
        Contains all the variables and objective function values.
    UTRFSP_problem_1 : UTRFSP_problem object
        Object containing the main UTRFSP details.
    data_for_analysis : False or pandas.DataFrame, optional
        The dataframe to contain the data analysis. If False, this is the fist 
        time the Dataframe is created and then only add the first pop_size 
        population of solutions. The default is False.

    Returns
    -------
    data_for_analysis : pandas.DataFrame
        The appended DataFrame.

    """
    pop_size = UTRFSP_problem_1.problem_GA_parameters.population_size
    
    if isinstance(data_for_analysis, pd.DataFrame):
        for index_i in range(pop_size):
            data_row = [pop_1.variables_routes_str[index_i]]
            data_row.extend(list(pop_1.variables_freq[index_i,:]))
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend([generation_num])
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
            
    else:
        columns_list = ["R_x", "F_3", "F_4","Generation", "Rank"]
        columns_list[1:1] = ["f_"+str(x) for x in range(UTRFSP_problem_1.problem_constraints.con_r)]
        data_for_analysis = pd.DataFrame(columns=columns_list)
        
        for index_i in range(pop_size):
            data_row = [pop_1.variables_routes_str[index_i]]
            data_row.extend(list(pop_1.variables_freq[index_i,:]))
            data_row.extend(list(pop_1.objectives[index_i,:]))
            data_row.extend([generation_num])
            data_row.extend(list(pop_1.rank[index_i]))
            data_for_analysis.loc[len(data_for_analysis)] = data_row
            
    return data_for_analysis

def add_UTRP_SA_data_ld(f_new_0, f_new_1, HV, SA_Temp, epoch, iteration_t, accepts, poor_epoch, routes_R_str, attempts, ld_data=False):
    """
    This allows for faster DataFrame construction

    Parameters
    ----------
    f_new_0 : First objective function value
    f_new_0 : Second objective function value
    HV : Hypervolume value
    SA_Temp : Simulated Annealing Temperature
    epoch : Current epoch of search
    iteration_t : the t'th iteration value
    accepts : the number of accepts
    poor_epoch : the number of poor epochs
    routes_R : Routes list
        The list object containing all the routes.
    attempts : the number of attempts made
    ld_data : a list of dictionaries
        Object containing the main SA search analysis data.
        If False, this is the fist time the list is created 
        and then only add the first iteration. The default is False.
                        
    Returns
    -------
    ld_data : list of dictionaries
        The appended DataFrame.

    """
    
    if isinstance(ld_data, list):
        ld_data.extend(
        [{"f1_ATT":f_new_0,
        "f2_TRT": f_new_1,
        "HV": HV,
        "Temperature": SA_Temp,
        "C_epoch_number": epoch,
        "L_iteration_per_epoch": iteration_t,
        "A_num_accepted_moves_per_epoch": accepts,
        "eps_num_epochs_without_accepting_solution": poor_epoch,
        "Route": routes_R_str,
        "Attempts": attempts}])
        
    else:
        ld_data = [{"f1_ATT": f_new_0,
        "f2_TRT": f_new_1,
        "HV": HV,
        "Temperature": SA_Temp,
        "C_epoch_number": epoch,
        "L_iteration_per_epoch": 0,
        "A_num_accepted_moves_per_epoch": 0,
        "eps_num_epochs_without_accepting_solution": 0,
        "Route": routes_R_str,
        "Attempts": 0}]  
        
    return ld_data
    
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

# %% Normalising functions

def normalise_data_UTRP(objectives, UTRP_problem):
    """A function to normalise data"""
    max_f_1 = UTRP_problem.problem_inputs.ref_point_max_f1_ATT
    min_f_1 = UTRP_problem.problem_inputs.ref_point_min_f1_ATT
    max_f_2 = UTRP_problem.problem_inputs.ref_point_max_f2_TRT
    min_f_2 = UTRP_problem.problem_inputs.ref_point_min_f2_TRT

    objs_norm = np.zeros(objectives.shape)
    objs_norm[:,0] = (objectives[:,0] - min_f_1)/(max_f_1 - min_f_1)
    objs_norm[:,1] = (objectives[:,1] - min_f_2)/(max_f_2 - min_f_2)
    
    return objs_norm

def recast_data_UTRP(objs_norm, UTRP_problem):
    """A function to recast normalised data"""
    max_f_1 = UTRP_problem.problem_inputs.ref_point_max_f1_ATT
    min_f_1 = UTRP_problem.problem_inputs.ref_point_min_f1_ATT
    max_f_2 = UTRP_problem.problem_inputs.ref_point_max_f2_TRT
    min_f_2 = UTRP_problem.problem_inputs.ref_point_min_f2_TRT

    objs_rec = np.zeros(objs_norm.shape)
    objs_rec[:,0] = objs_norm[:,0] * (max_f_1 - min_f_1) + min_f_1
    objs_rec[:,1] = objs_norm[:,1] * (max_f_2 - min_f_2) + min_f_2
    
    return objs_rec

# %% Mutation admin functions

def get_mutations_summary(df_mut_temp, nr_mutations, gen_nr):
    df_mut_summary = pd.DataFrame(columns=(["Generation", "Mut_nr","Total", "Mut_successful", "Mut_repaired", "Included_new_gen"]))
    ld_mut_summary =[]
    for mut_nr in range(nr_mutations+1):
        df_mut_only = df_mut_temp[df_mut_temp["Mut_nr"]==mut_nr]
        df_mut_success = df_mut_only[df_mut_only["Mut_successful"]==1]

        tot_mut = len(df_mut_only)
        summary = df_mut_only.to_numpy().sum(axis=0)
        df_mut_summary.loc[mut_nr] = [gen_nr, mut_nr, tot_mut, summary[1], summary[2], df_mut_success["Included_new_gen"].sum()] 
        
        mut_dict = {"Generation":gen_nr,
                    "Mut_nr":mut_nr,
                    "Total":tot_mut, 
                    "Mut_successful":summary[1], 
                    "Mut_repaired":summary[2], 
                    "Included_new_gen":df_mut_success["Included_new_gen"].sum()}
        
        ld_mut_summary.append(mut_dict)
    
    def sum_row_Inc_over_succ(row):
        if row.Mut_successful + row.Mut_repaired == 0:
            return 0
        else:
            return row.Included_new_gen/(row.Mut_successful + row.Mut_repaired)
    
    row_func_1 = lambda row: sum_row_Inc_over_succ(row)
            
    df_mut_summary["Inc_over_succ"] = df_mut_summary.apply(row_func_1, axis=1)
    
    def sum_row_Inc_over_Tot(row):
        if row.Total == 0:
            return 0
        else:
            return row.Included_new_gen/(row.Total)
    
    row_func_2 = lambda row: sum_row_Inc_over_Tot(row)
            
    df_mut_summary["Inc_over_Tot"] = df_mut_summary.apply(row_func_2, axis=1)
    
    return df_mut_summary


# %% Mutation ratio update functions

def update_mutation_ratio_simple(df_mut_summary, UTNDP_problem_1):
    nr_of_mutations = len(UTNDP_problem_1.mutation_functions)
    mutation_threshold = UTNDP_problem_1.problem_GA_parameters.mutation_threshold
    ratio_update_weights = df_mut_summary["Inc_over_succ"].iloc[-nr_of_mutations:].values
    weight_products = ratio_update_weights * UTNDP_problem_1.mutation_ratio
    mutable_ratios = (weight_products / sum(weight_products))*(1-nr_of_mutations*mutation_threshold)
    updated_ratio = mutation_threshold + mutable_ratios
    UTNDP_problem_1.mutation_ratio = updated_ratio
                
def update_mutation_ratio_exp_smooth(df_mut_summary, UTNDP_problem_1, alpha=0.3):
    old_ratio = UTNDP_problem_1.mutation_ratio
    nr_of_mutations = len(UTNDP_problem_1.mutation_functions)
    mutation_threshold = UTNDP_problem_1.problem_GA_parameters.mutation_threshold
    ratio_update_weights = df_mut_summary["Inc_over_succ"].iloc[-nr_of_mutations:].values
    weight_products = ratio_update_weights * UTNDP_problem_1.mutation_ratio
    mutable_ratios = (weight_products / sum(weight_products))*(1-nr_of_mutations*mutation_threshold)
    updated_ratio = mutation_threshold + mutable_ratios
    UTNDP_problem_1.mutation_ratio = alpha*np.array(updated_ratio) + (1-alpha)*np.array(old_ratio)
    
def update_mutation_ratio_exp_double_smooth(df_mut_summary, UTNDP_problem_1, i_gen, s_t_min_1, b_t_min_1, alpha=0.5, beta=0.3):
    old_ratio = np.array(UTNDP_problem_1.mutation_ratio)
    nr_of_mutations = len(UTNDP_problem_1.mutation_functions)
    mutation_threshold = UTNDP_problem_1.problem_GA_parameters.mutation_threshold
    ratio_update_weights = df_mut_summary["Inc_over_succ"].iloc[-nr_of_mutations:].values
    weight_products = ratio_update_weights * UTNDP_problem_1.mutation_ratio
    mutable_ratios = (weight_products / sum(weight_products))*(1-nr_of_mutations*mutation_threshold)
    updated_ratio = np.array(mutation_threshold + mutable_ratios)
    
    if i_gen == 1:
        s_t_min_1 = old_ratio
        b_t_min_1 = updated_ratio - old_ratio
        
    s_t = alpha*updated_ratio + (1-alpha)*(s_t_min_1 + b_t_min_1)
    b_t = beta*(s_t - s_t_min_1) + (1-beta)*b_t_min_1
    
    UTNDP_problem_1.mutation_ratio = s_t
    
    return s_t, b_t

def exp_smooth_df(df, alpha, beta, n=100):
    '''A function for exp smoothing a dataframe (excluding first column'''
    # ref: https://en.wikipedia.org/wiki/Exponential_smoothing

    mut_r = np.array(df)[:,1:]
    s_t = np.zeros(mut_r.shape)
    b_t = np.zeros(mut_r.shape)
    
    for i in range(len(mut_r)):
        if i == 0:
            s_t[i] = mut_r[i]
            b_t[i] = (mut_r[i+1] - mut_r[i])/n
        
        else:
            s_t[i] = alpha*mut_r[i] + (1-alpha)*(s_t[i-1] + b_t[i-1])
            b_t[i] = beta*(s_t[i] - s_t[i-1]) + (1-beta)*b_t[i-1]
            
    df_smooth = pd.DataFrame(data=s_t, columns=df.columns[1:])
    df_smooth.insert(0, df.columns[0], df[df.columns[0]])
    
    return df_smooth

# %% Pickle Save Object Functions

def load_obj_pickle(name, directory):
    '''Function to easily load object from pickle file'''
    with open(directory / (name+".pickle"),'rb') as read_file:
        obj =  pickle.load(read_file) # load the object
        return obj

def save_obj_pickle(obj, name, directory):
    '''Function to easily save object to pickle file'''
    pickle.dump(obj, open(directory / (name+".pickle"), "ab"))
    