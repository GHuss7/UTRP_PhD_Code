# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:48:53 2020

@author: 17832020
"""
import pandas as pd
import os
import time
import datetime
import re
import string
import json
from pathlib import Path
import numpy as np
import csv

import argparse

# %% Import personal functions
path_DSS_Main_folder = "C:/Users/gunth/OneDrive - Stellenbosch University/Academics 2019 MEng/Documents/GitHub/DSS_Main_Laptop"
if not os.path.exists(path_DSS_Main_folder):
    path_DSS_Main_folder = "C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS/DSS Main"
assert os.path.exists(path_DSS_Main_folder)
#os.chdir(path_DSS_Main_folder)
#print(os.getcwd())
import DSS_Admin as ga
import DSS_UTNDP_Functions as gf_p
import DSS_UTNDP_Functions_c as gf
import DSS_UTNDP_Classes as gc
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import EvaluateRouteSet as ev
import DSS_K_Shortest_Paths as ksp

# Input data names:
input_data_names = ['7_1_Mandl6_GA_Pop_size', #29
                    '7_2_Mumford0_GA_Pop_size', #30
                    '7_3_Mumford1_GA_Pop_size', #31
                    '7_4_Mumford2_GA_Pop_size', #32
                    '7_5_Mumford3_GA_Pop_size', #33
                    '8_1_Mandl6_GA_Crossover_prob', #34
                    '8_2_Mumford0_GA_Crossover_prob', #35
                    '8_3_Mumford1_GA_Crossover_prob', #36
                    '8_4_Mumford2_GA_Crossover_prob', #37
                    '8_5_Mumford3_GA_Crossover_prob', #38
                    '9_1_Mandl6_GA_Mut_prob', #39
                    '9_2_Mumford0_GA_Mut_prob', #40
                    '9_3_Mumford1_GA_Mut_prob', #41
                    '9_4_Mumford2_GA_Mut_prob', #42
                    '9_5_Mumford3_GA_Mut_prob', #43
                    '10_1_Mandl6_GA_Long_run', #44
                    '10_2_Mumford0_GA_Long_run', #45
                    '10_3_Mumford1_GA_Long_run', #46
                    '10_4_Mumford2_GA_Long_run', #47
                    '10_5_Mumford3_GA_Long_run', #48
                    ]

# Arguments from command line
parser = argparse.ArgumentParser()

#-dir DIRECTORY
parser.add_argument("-dir", "--directory", dest = "dir", default = os.path.dirname(os.path.realpath(__file__)), help="Directory", type=str)

args = parser.parse_args()
arg_dir = args.dir

print(arg_dir)

# Ensure the directory is set to the file location
dir_path = arg_dir
path_results = Path(arg_dir)
os.chdir(dir_path)

# %% Load the relevant parameters
Decisions = json.load(open(dir_path+"/Parameters/Decisions.json"))
parameters_constraints = json.load(open(dir_path+"/Parameters/parameters_constraints.json"))
parameters_input = json.load(open(dir_path+"/Parameters/parameters_input.json"))
parameters_GA = json.load(open(dir_path+"/Parameters/parameters_GA.json"))

# Load validation data:
stats_overall = {} # declare stats overall dict
for str_name in input_data_names:
    if str_name in arg_dir:
        name_input_data = str_name
        break
else:
    print("Input data name not found in input_data_names list")
    name_input_data = False
    validation_data = False

'''Load validation data'''
os.chdir(path_DSS_Main_folder)
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)
if os.path.exists("./Input_Data/"+name_input_data+"/Validation_Data/Results_data_headers_all.csv"):
    validation_data = pd.read_csv("./Input_Data/"+name_input_data+"/Validation_Data/Results_data_headers_all.csv")
os.chdir(dir_path)

'''Load the initial set'''
print("Loading the initial route set")
df_pop_generations = pd.read_csv(dir_path+"/Run_1/Pop_generations.csv")
initial_set = df_pop_generations.iloc[0:parameters_GA['population_size'],:] # load initial set

# Get details of runs
def count_Run_folders(path_to_folder):
    # NB only all the folders of Runs should me in the main path, otherwise errors will occur
    result_entries = os.listdir(path_to_folder) # gets the names of all the results entries
    counter = 0
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            counter = counter + 1
    return counter

nr_of_runs = count_Run_folders(path_results)

'''Set the reference point for the Hypervolume calculations'''
max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])

# Define the main problem
UTNDP_problem_1 = gc.UTNDP_problem()
UTNDP_problem_1.problem_data = gc.Problem_data(mx_dist, mx_demand, mx_coords)
UTNDP_problem_1.problem_constraints = gc.Problem_constraints(parameters_constraints)
UTNDP_problem_1.problem_inputs = gc.Problem_inputs(parameters_input)
UTNDP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA)
UTNDP_problem_1.Decisions = Decisions

# Set the labels for graphs
labels = ["f_1", "f_2", "f1_ATT", "f2_TRT"]

# %% Save results after all runs
'''Save the summarised results'''

for _ in range(5):
    try:
        df_overall_pareto_set = ga.group_pareto_fronts_from_model_runs_2(path_results, parameters_input, "Non_dominated_set.csv").iloc[:,1:]
        df_overall_pareto_set = df_overall_pareto_set[gf.is_pareto_efficient(df_overall_pareto_set[["f_1","f_2"]].values, True)] # reduce the pareto front from the total archive
        df_overall_pareto_set = df_overall_pareto_set.sort_values(by='f_1', ascending=True) # sort
        df_overall_pareto_set.to_csv(path_results / "Overall_Pareto_set.csv")   # save the csv file
        break
    except PermissionError:
        time.sleep(5)
        continue

'''Save the stats for all the runs'''
# df_routes_R_initial_set.to_csv(path_results / "Routes_initial_set.csv")
for _ in range(5):
    try:
        df_durations = ga.get_stats_from_model_runs(path_results)
        break
    except:
        time.sleep(5)
        continue

stats_overall['execution_start_time'] = datetime.datetime.now()
stats_overall['execution_end_time'] =  datetime.datetime.now()

stats_overall['total_model_runs'] = nr_of_runs
stats_overall['average_run_time'] = str(df_durations["Duration"].mean())
stats_overall['total_duration'] = stats_overall['execution_end_time']-stats_overall['execution_start_time']
stats_overall['start_time_formatted'] = stats_overall['execution_start_time'].strftime("%m/%d/%Y, %H:%M:%S")
stats_overall['end_time_formatted'] = stats_overall['execution_end_time'].strftime("%m/%d/%Y, %H:%M:%S")
stats_overall['HV obtained'] = gf_p.norm_and_calc_2d_hv(df_overall_pareto_set[["f_1","f_2"]], max_objs, min_objs)
if isinstance(validation_data, pd.DataFrame):
    stats_overall['HV Benchmark'] = gf_p.norm_and_calc_2d_hv(validation_data[validation_data["Approach"]=="John (2016)"].iloc[:,0:2], 
                                                        max_objs, min_objs)
else:
    stats_overall['HV Benchmark'] = 0

df_durations.loc[len(df_durations)] = ["Average", df_durations["Duration"].mean(), df_durations["HV Obtained"].mean()]
df_durations.to_csv(path_results / "Run_durations.csv")
del df_durations

try:
    # Writes all the stats in a csv file
    with open(path_results / "Stats_overall.csv", "w") as archive_file:
        w = csv.writer(archive_file)
        for key, val in {**stats_overall,
                            **parameters_input, 
                            **parameters_constraints, 
                            **parameters_GA}.items():
            w.writerow([key, val])
        del key, val

except PermissionError: pass

for _ in range(5):
    try:
        ga.get_sens_tests_stats_from_model_runs(path_results, nr_of_runs) # prints the runs summary
        gv.save_all_mutation_stats_and_plots(path_results) # gets and prints the mutation stats
        gv.save_all_obj_stats_and_plots(path_results) # gets and prints the objective performance stats
        #gv.save_final_avgd_results_analysis(initial_set, df_overall_pareto_set, validation_data, 
        #                                    pd.read_csv((path_results/'Performance/Avg_obj_performances.csv')), 
        #                                    pd.read_csv((path_results/'Mutations/Avg_mut_ratios.csv')), # can use 'Smoothed_avg_mut_ratios.csv' for a double smooth visualisation
        #                                    name_input_data, 
        #                                    path_results, labels,
        #                                    stats_overall['HV Benchmark'], 'line')
        
        #gv.save_final_avgd_results_analysis(initial_set, df_overall_pareto_set, validation_data, 
        #                                    pd.read_csv((path_results/'Performance/Avg_obj_performances.csv')), 
        #                                    pd.read_csv((path_results/'Mutations/Avg_mut_ratios.csv')), # can use 'Smoothed_avg_mut_ratios.csv' for a double smooth visualisation
        #                                    name_input_data, 
        #                                    path_results, labels,
        #                                    stats_overall['HV Benchmark'], 'stacked')
        
        gv.save_final_avgd_results_analysis(initial_set, df_overall_pareto_set, validation_data, 
                                            pd.read_csv((path_results/'Performance/Avg_obj_performances.csv')), 
                                            pd.read_csv((path_results/'Mutations/Avg_mut_ratios.csv')), # can use 'Smoothed_avg_mut_ratios.csv' for a double smooth visualisation
                                            name_input_data, 
                                            path_results, labels,
                                            stats_overall['HV Benchmark'], 'stacked_smooth')
        
        print("Printing extreme solutions")
        gf.print_extreme_solutions(df_overall_pareto_set, stats_overall['HV obtained'], stats_overall['HV Benchmark'], name_input_data, UTNDP_problem_1, path_results)
        ga.get_sens_tests_stats_from_UTRP_GA_runs(path_results) 
        
        break
    except PermissionError:
        time.sleep(5)
        continue
    
#del archive_file, path_results_per_run, w           

# %% Plot analysis graph
'''Plot the analysis graph'''
try:
    gv.save_results_combined_fig(initial_set, df_overall_pareto_set, validation_data, name_input_data, Decisions, path_results, labels)
except PermissionError: pass
#del run_nr
print("Done!")