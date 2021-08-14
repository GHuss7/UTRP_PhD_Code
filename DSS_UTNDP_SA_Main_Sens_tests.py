# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:02:00 2019

@author: 17832020
"""

# %% Import Libraries
import os
from pathlib import Path
import re
import csv
import json
import pickle
import pandas as pd 
import numpy as np
from math import inf
#import pygmo as pg
import random
import copy
import datetime
import time
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt
import igraph as ig
import networkx as nx


# %% Import personal functions
import DSS_Admin as ga
import DSS_UTNDP_Classes as gc
#import DSS_UTNDP_Functions as gf
import DSS_UTNDP_Functions as gf_p
import DSS_UTNDP_Functions_c as gf
import DSS_Visualisation as gv
import EvaluateRouteSet as ev
import DSS_K_Shortest_Paths as ksp


# %% Load the respective instances
name_input_data = ["Mandl_UTRP", #0
                   "Mumford0_UTRP", #1
                   "Mumford1_UTRP", #2
                   "Mumford2_UTRP", #3
                   "Mumford3_UTRP", #4
                   "Mandl_UTRP_testing", #5
                   "Mandl_UTRP_dis", #6
                   "Mandl4_UTRP", #7
                   "Mandl6_UTRP", #8
                   "Mandl7_UTRP", #9
                   "Mandl8_UTRP", #10
                   
                   '21_1_Mandl6_SA_Initial_solutions', #11
                    '21_2_Mumford0_SA_Initial_solutions', #12
                    '21_3_Mumford1_SA_Initial_solutions', #13
                    '22_1_Mandl6_SA_Mut_update', #14
                    '22_2_Mumford0_SA_Mut_update', #15
                    '22_3_Mumford1_SA_Mut_update', #16

                   '0_21_1_Mandl6_SA_Init_sol_test',
                   '0_22_1_Mandl6_SA_Mut_tests',
                   
                   
                   ][11]   # set the name of the input data

# Set test paramaters
sens_from = 0 # sets the entire list that should be used as input. Lists by be broken down in smaller pieces for convenience
sens_to = (sens_from + 1) if False else -1
test_counters = [] # empty list means all, filled in values means only those tests

# %% Set input parameters
if True:
    Decisions = json.load(open("./Input_Data/"+name_input_data+"/Decisions.json"))

else:
    Decisions = {
    "Choice_generate_initial_set" : True, # the alternative loads a set that is prespecified, False is default for MANDL NB
    "Choice_first_init_set_only" : False, # if true, it selects only the initial route set found in the first location to help standardise
    "Choice_print_results" : True, 
    "Choice_conduct_sensitivity_analysis" : False,
    "Choice_import_dictionaries" : True,
    "Choice_relative_results_referencing" : False,
    "Choice_init_temp_with_trial_runs" : False, # runs M trial runs for the initial temperature
    "Choice_normal_run" : True, # choose this for a normal run without Sensitivity Analysis
    "Choice_import_saved_set" : True, # import the prespecified set
    "Set_name" : "Overall_Pareto_set_for_case_study_GA.csv", # the name of the set in the main working folder
    "Additional_text" : "Tests",
    "Pop_size_to_create" : 2000,
    "Log_prints_every" : 20, # every n generations a log should be printed
    "CSV_prints_every" : 100, # every n generations a csv should be printed
    "PDF_prints_every" : 20, # every n generations a csv should be printed
    "Load_supplementing_pop" : False,
    "Obj_func_disruption" : False,
    "Update_mutation_ratios" : True,
    
    "route_gen_func" : "KSP_unseen_robust_prob",    
    "mutation_funcs" : ["Intertwine_two",
                         "Add_vertex",
                         "Delete_vertex",
                         "Invert_path_vertices",
                         "Insert_inside_vertex",
                         "Relocate_inside_vertex",
                         "Replace_inside_vertex",
                         "Donate_between_routes",
                         "Swap_between_routes",
                         "Trim_one_terminal_cb",
                         "Grow_one_terminal_cb",
                         "Grow_one_path_random_cb",
                         "Grow_routes_random_cb"],
    "mut_update_func" : ["AMALGAM", "AMALGAM_every_n", "Counts_normal"][0]
    }


# %% Set functions to use
route_gen_funcs = {"KSP_unseen_robust" : gc.Routes.return_feasible_route_robust_k_shortest,
                   "KSP_unseen_robust_prob_10000" : gc.Routes.return_feasible_route_robust_k_shortest_probabilistic,
                   "KSP_unseen_robust_prob" : gc.Routes.return_feasible_route_robust_k_shortest_probabilistic,
                    "Greedy_demand" : gc.Routes.return_feasible_route_set_greedy_demand,
                    "Unseen_robust" : gc.Routes.return_feasible_route_robust}
assert Decisions["route_gen_func"] in route_gen_funcs.keys()
route_gen_func_name = Decisions["route_gen_func"]

mutations_all = {#"No_mutation" : gf.no_mutation,
                "Intertwine_two" : gf.mutate_routes_two_intertwine, 
                "Add_vertex" : gf.add_vertex_to_terminal,
                "Delete_vertex" : gf.remove_vertex_from_terminal,
                
                "Invert_path_vertices" : gf.mut_invert_route_vertices,
                "Insert_inside_vertex" : gf.mut_add_vertex_inside_route,
                "Delete_inside_vertex" : gf.mut_delete_vertex_inside_route,
                
                #"Relocate_inside_vertex" : gf.mut_relocate_vertex_inside_route,
                "Replace_inside_vertex" : gf.mut_replace_vertex_inside_route,
                "Donate_between_routes" : gf.mut_donate_vertex_between_routes,
                "Swap_between_routes" : gf.mut_swap_vertices_between_routes,
    
                "Merge_terminals" : gf.mutate_merge_routes_at_common_terminal, 
                "Repl_low_dem_route" : gf.mut_replace_lowest_demand,
                "Rem_low_dem_terminal" : gf.mut_remove_lowest_demand_terminal,
                "Rem_lrg_cost_terminal" : gf.mut_remove_largest_cost_terminal,
                "Repl_high_sim_route":gf.mut_replace_high_sim_routes, # bad mutation
                "Repl_subsets" : gf.mut_replace_path_subsets,
                      
                "Trim_one_terminal_cb" : gf.mut_trim_one_terminal_cb,
                "Trim_one_path_random_cb" : gf.mut_trim_one_path_random_cb,
                "Trim_routes_random_cb" : gf.mut_trim_routes_random_cb,
                "Trim_all_paths_random_cb" : gf.mut_trim_all_paths_random_cb,
                "Trim_full_overall_cb" : gf.mut_trim_full_overall_cb,
                
                "Grow_one_terminal_cb" : gf.mut_grow_one_terminal_cb,
                "Grow_one_path_random_cb" : gf.mut_grow_one_path_random_cb,
                "Grow_routes_random_cb" : gf.mut_grow_routes_random_cb,
                "Grow_all_paths_random_cb" : gf.mut_grow_all_paths_random_cb,
                "Grow_full_overall_cb" : gf.mut_grow_full_overall_cb,
                
                "MSC_add_terminal" : gf.perturb_make_small_add_terminal,
                "MSC_del_terminal" : gf.perturb_make_small_del_terminal,
                }

mutations = {k : v for (k,v) in mutations_all.items() if k in Decisions["mutation_funcs"]}

all_functions_dict = {"Mut_"+k : v.__name__ for (k,v) in mutations.items()}

mutations_dict = {i+1:{"name":k, "func":v} for i,(k,v) in zip(range(len(mutations)),mutations.items())}
mut_functions = [v['func'] for (k,v) in mutations_dict.items()]
mut_names = [v['name'] for (k,v) in mutations_dict.items()]

# %% Load the respective files
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)

#%% Load and set problem parameters #######################################
if Decisions["Choice_import_dictionaries"]:
    parameters_constraints = json.load(open("./Input_Data/"+name_input_data+"/parameters_constraints.json"))
    parameters_input = json.load(open("./Input_Data/"+name_input_data+"/parameters_input.json"))
    parameters_SA_routes = json.load(open("./Input_Data/"+name_input_data+"/parameters_SA_routes.json"))

    file_name_ksp = "K_shortest_paths_50_shortened_5_demand"
    if not os.path.exists("./Input_Data/"+name_input_data+"/K_Shortest_Paths/Saved/"+file_name_ksp+".csv"): 
        file_to_find = "./Input_Data/"+name_input_data+"/K_Shortest_Paths/"+file_name_ksp+".csv"
        if os.path.exists(file_to_find):
            print(f"Move file {file_name_ksp} to Saved folder")
        print("Creating k_shortest paths and saving csv file...")
        df_k_shortest_paths = ksp.create_polished_ksp(mx_dist, mx_demand, k_cutoff=50, save_csv=True)
        df_k_shortest_paths = pd.read_csv(file_to_find)

    else:
        df_k_shortest_paths = pd.read_csv("./Input_Data/"+name_input_data+"/K_Shortest_Paths/Saved/"+file_name_ksp+".csv")
   
    # parameters_SA_routes={
    # "method" : "SA",
    # # ALSO: t_max > A_min (max_iterations_t > min_accepts)
    # "max_iterations_t" : 100, # maximum allowable number length of iterations per epoch; Danie PhD (pg. 98): Dreo et al. chose 100
    # "max_total_iterations" : 8000, # the total number of accepts that are allowed
    # "max_epochs" : 4000, # the maximum number of epochs that are allowed
    # "min_accepts" : 25, # minimum number of accepted moves per epoch; Danie PhD (pg. 98): Dreo et al. chose 12N (N being some d.o.f.)
    # "max_attempts" : 50, # maximum number of attempted moves per epoch
    # "max_reheating_times" : 5, # the maximum number of times that reheating can take place
    # "max_poor_epochs" : 400, # maximum number of epochs which may pass without the acceptance of any new solution
    # "Temp" : 0.1,  # starting temperature and a geometric cooling schedule is used on it # M = 1000 gives 93.249866 from 20 runs
    # "M_iterations_for_temp" : 1000, # the number of initial iterations to establish initial starting temperature
    # "Cooling_rate" : 0.95, # the geometric cooling rate 0.97 has been doing good, but M =1000 gives 0.996168
    # "Reheating_rate" : 1.10, # the geometric reheating rate
    # "number_of_initial_solutions" : 4, # sets the number of initial solutions to generate as starting position
    # "Feasibility_repair_attempts" : 10, # the max number of edges that will be added and/or removed to try and repair the route feasibility
    # "number_of_runs" : 4, # number of runs to complete John 2016 set 20
    # "iter_compare_HV" : 8000, # Compare iterations for improvement in HV
    # "HV_improvement_th": 0.0001, # Treshold that terminates the search
    # "mutation_threshold" : 0.03, # Minimum threshold that mutation probabilities can reach,
    # "mutation_update_interval" : 200, # The interval in which the mutation ratio is updated
    # } 
   

else:
    '''Enter the number of allowed routes''' 
    parameters_constraints = {
    'con_r' : 12,               # (aim for > [numNodes N ]/[maxNodes in route])
    'con_minNodes' : 2,                        # minimum nodes in a route
    'con_maxNodes' : 15,                       # maximum nodes in a route
    'con_N_nodes' : len(mx_dist)              # number of nodes in the network
    }
    
    parameters_input = {
    'total_demand' : sum(sum(mx_demand))/2, # total demand from demand matrix
    'n' : len(mx_dist), # total number of nodes
    'wt' : 0, # waiting time [min]
    'tp' : 5, # transfer penalty [min]
    'ref_point_max_f1_ATT' : 32, # max f1_ATT for the Hypervolume calculations
    'ref_point_min_f1_ATT' : 13, # min f1_ATT for the Hypervolume calculations
    'ref_point_max_f2_TRT' : 700, # max f2_TRT for the Hypervolume calculations
    'ref_point_min_f2_TRT' : 94, # min f2_TRT for the Hypervolume calculations
    }
    
    parameters_SA_routes={
    "method" : "SA",
    # ALSO: t_max > A_min (max_iterations_t > min_accepts)
    "max_iterations_t" : 250, # maximum allowable number length of iterations per epoch; Danie PhD (pg. 98): Dreo et al. chose 100
    "max_total_iterations" : 8000, # the total number of accepts that are allowed
    "max_epochs" : 2000, # the maximum number of epochs that are allowed
    "min_accepts" : 25, # minimum number of accepted moves per epoch; Danie PhD (pg. 98): Dreo et al. chose 12N (N being some d.o.f.)
    "max_attempts" : 50, # maximum number of attempted moves per epoch
    "max_reheating_times" : 5, # the maximum number of times that reheating can take place
    "max_poor_epochs" : 400, # maximum number of epochs which may pass without the acceptance of any new solution
    "Temp" : 10,  # starting temperature and a geometric cooling schedule is used on it # M = 1000 gives 93.249866 from 20 runs
    "M_iterations_for_temp" : 1000, # the number of initial iterations to establish initial starting temperature
    "Cooling_rate" : 0.97, # the geometric cooling rate 0.97 has been doing good, but M =1000 gives 0.996168
    "Reheating_rate" : 1.05, # the geometric reheating rate
    "number_of_initial_solutions" : 1, # sets the number of initial solutions to generate as starting position
    "Feasibility_repair_attempts" : 3, # the max number of edges that will be added and/or removed to try and repair the route feasibility
    "number_of_runs" : 20, # number of runs to complete John 2016 set 20
    "iter_compare_HV" : 4000, # Compare generations for improvement in HV
    "HV_improvement_th": 0.0001, # Treshold that terminates the search
    "mutation_threshold" : 0.03, # Minimum threshold that mutation probabilities can reach,
    "mutation_update_interval" : 200, # The interval in which the mutation ratio is updated
    }
    
    """Full sensitivity analysis"""
    # Set up the list of parameters to test
    # sensitivity_list = [[parameters_SA_routes, "max_iterations_t", 10, 50, 100, 250, 500, 1000, 1500], 
    #                     [parameters_SA_routes, "min_accepts",  1, 3, 5, 10, 25, 50, 100, 200, 400], # takes longer at first... bottleneck
    #                     [parameters_SA_routes, "max_attempts", 1, 3, 5, 10, 25, 50, 100, 200, 400],
    #                     [parameters_SA_routes, "max_reheating_times", 1, 3, 5, 10, 25],
    #                     [parameters_SA_routes, "max_poor_epochs", 1, 3, 5, 10, 25, 50, 100, 200, 400],
    #                     [parameters_SA_routes, "Temp", 1, 5, 10, 25, 50, 100, 150, 200],
    #                     [parameters_SA_routes, "Cooling_rate", 0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 0.9961682402927605],
    #                     [parameters_SA_routes, "Reheating_rate", 1.5, 1.3, 1.1, 1.05, 1.02],
    #                     [parameters_SA_routes, "Feasibility_repair_attempts", 1, 2, 3, 4, 5, 6],
    #                     ]
    
    """Quick tests with main parameters only"""
    sensitivity_list = [[parameters_SA_routes, "Cooling_rate", 0.5, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.99, 0.9961682402927605],
                        [parameters_SA_routes, "Temp", 0.001, 0.01, 0.1, 1, 10, 100],
                        [parameters_SA_routes, "max_poor_epochs", 1, 3, 5, 10, 25, 50, 100, 200, 400],
                        [parameters_SA_routes, "max_attempts", 1, 3, 5, 10, 25, 50, 100, 200, 400],
                        [parameters_SA_routes, "max_reheating_times", 1, 3, 5, 10, 25],
                        [parameters_SA_routes, "Reheating_rate", 2, 1.5, 1.3, 1.1, 1.05, 1.02],

                        [parameters_SA_routes, "max_iterations_t", 10, 50, 100, 250, 500, 1000], 
                        
                        [parameters_SA_routes, "min_accepts",  1, 3, 5, 10, 25, 50, 100, 200, 400], # takes longer at first... bottleneck

                        [parameters_SA_routes, "Feasibility_repair_attempts", 2, 5, 7, 10, 15, 20],
                        ]
    
    """Sensitivity analysis with highs and lows"""
    #sensitivity_list = [#[parameters_SA_routes, "max_iterations_t", 10, 1000], 
                        
                        #[parameters_SA_routes, "min_accepts", 5, 200], # takes longer at first... bottleneck
                        #[parameters_SA_routes, "max_attempts", 3, 200],
                        
                        #[parameters_SA_routes, "max_reheating_times", 1, 25],
                        
                        #[parameters_SA_routes, "max_poor_epochs", 10, 600],
                        #[parameters_SA_routes, "Cooling_rate", 0.97],
                        
                        #[parameters_SA_routes, "Temp", 10],
                        #[parameters_SA_routes, "Temp", 0.001, 1000],
                        
                        #[parameters_SA_routes, "Temp", 10],
                        #[parameters_SA_routes, "Temp", 50],
                        
                        #[parameters_SA_routes, "Cooling_rate", 0.7, 0.9961682402927605],
                        
                        #[parameters_SA_routes, "Reheating_rate", 1.3, 1.01],
                        #]

    

'''Set the reference point for the Hypervolume calculations'''
max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])

labels = ["f_1", "f_2", "f1_ATT", "f2_TRT"] # names labels for the visualisations
  
# %% Define the adjacent mapping of each node
mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes

#%% Define the UTNDP Problem      
UTNDP_problem_1 = gc.UTNDP_problem()
UTNDP_problem_1.problem_data = gc.Problem_data(mx_dist, mx_demand, mx_coords)
UTNDP_problem_1.problem_constraints = gc.Problem_constraints(parameters_constraints)
UTNDP_problem_1.problem_inputs = gc.Problem_inputs(parameters_input)
UTNDP_problem_1.problem_SA_parameters = gc.Problem_metaheuristic_inputs(parameters_SA_routes)
UTNDP_problem_1.Decisions = Decisions
UTNDP_problem_1.k_short_paths = gc.K_shortest_paths(df_k_shortest_paths)
UTNDP_problem_1.mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes
UTNDP_problem_1.max_objs = max_objs
UTNDP_problem_1.min_objs = min_objs
UTNDP_problem_1.add_text = f"T{parameters_SA_routes['max_total_iterations']}_E{parameters_SA_routes['max_epochs']}_R{parameters_SA_routes['number_of_runs']}" # define the additional text for the file name
UTNDP_problem_1.mutation_functions = mut_functions
UTNDP_problem_1.mutation_names = mut_names
UTNDP_problem_1.mutation_ratio = [1/len(mut_functions) for _ in mut_functions]
UTNDP_problem_1.mutation_ratio_counts = [round(x*100,0) for  x in UTNDP_problem_1.mutation_ratio]
UTNDP_problem_1.problem_GA_parameters = {'mutation_probability' : 1} # setting for mutations

# UTNDP_problem_1.R_routes = R_routes

# Add route compare component
if os.path.exists("./Input_Data/"+name_input_data+"/Route_compare.txt"):
    route_file = open("./Input_Data/"+name_input_data+"/Route_compare.txt","r")
    route_compare = route_file.read()
    UTNDP_problem_1.route_compare = gf.convert_routes_str2list(route_compare)
else:
    route_compare = gc.Routes.return_feasible_route_robust(UTNDP_problem_1)
    route_file = open("./Input_Data/"+name_input_data+"/Route_compare.txt","w")
    route_file.write(gf.convert_routes_list2str(route_compare))
    route_file.close()
    UTNDP_problem_1.route_compare = route_compare
del route_file, route_compare

if Decisions["Choice_init_temp_with_trial_runs"]:
    UTNDP_problem_1.problem_SA_parameters.Temp, UTNDP_problem_1.problem_SA_parameters.Cooling_rate = gf.init_temp_trial_searches(UTNDP_problem_1, number_of_runs=1)
    parameters_SA_routes["Temp"], parameters_SA_routes["Cooling_rate"] = UTNDP_problem_1.problem_SA_parameters.Temp, UTNDP_problem_1.problem_SA_parameters.Cooling_rate

# if True:
def main(UTNDP_problem_1):

    # Reload the decisions and adjust appropriately
    Decisions = UTNDP_problem_1.Decisions # Load the decisions
    
    # Mutations reload
    mutations = {k : v for (k,v) in mutations_all.items() if k in Decisions["mutation_funcs"]}

    all_functions_dict = {"Mut_"+k : v.__name__ for (k,v) in mutations.items()}
        
    mutations_dict = {i+1:{"name":k, "func":v} for i,(k,v) in zip(range(len(mutations)),mutations.items())}
    mut_functions = [v['func'] for (k,v) in mutations_dict.items()]
    mut_names = [v['name'] for (k,v) in mutations_dict.items()]

    UTNDP_problem_1.mutation_functions = mut_functions
    UTNDP_problem_1.mutation_names = mut_names
    UTNDP_problem_1.mutation_ratio = [1/len(mut_functions) for _ in mut_functions]
    
    """ Keep track of the stats """
    stats_overall = {
        'execution_start_time' : datetime.datetime.now() # enter the begin time
        } 
    
    stats = {} # define the stats dictionary
    
    #%% Define the Objective UTNDP functions
    def fn_obj_2(routes, UTNDP_problem_input):
        return (ev.evalObjs(routes, 
                UTNDP_problem_input.problem_data.mx_dist, 
                UTNDP_problem_input.problem_data.mx_demand, 
                UTNDP_problem_input.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)
    
    if Decisions["Obj_func_disruption"]:
        def fn_obj_ATT(routes, UTNDP_problem_input):           
            ATT = ev.evalATT(routes, 
                        UTNDP_problem_input.problem_data.mx_dist, 
                        UTNDP_problem_input.problem_data.mx_demand, 
                        UTNDP_problem_input.problem_inputs.__dict__)
            return (ATT, 0) # returns (f1_ATT, 0)
        
        def fn_obj_TRT(routes, UTNDP_problem_input):
            travelTimes = UTNDP_problem_input.problem_data.mx_dist
            RL = ev.evaluateTotalRouteLength(routes,travelTimes)
            return (0, RL) # returns (0, f2_TRT)
        
        #fn_obj_2 = fn_obj_ATT
        # fn_obj_2 = fn_obj_TRT
        fn_obj_2 = gf_p.fn_obj_3 # returns (f1_ATT, RD)
        
    # %% Load files and set folder paths    
    '''Load validation data'''
    if os.path.exists("./Input_Data/"+name_input_data+"/Validation_Data/Results_data_headers_all.csv"):
        validation_data = pd.read_csv("./Input_Data/"+name_input_data+"/Validation_Data/Results_data_headers_all.csv")
        stats_overall['HV Benchmark'] = gf_p.norm_and_calc_2d_hv(validation_data[validation_data["Approach"]=="John (2016)"].iloc[:,0:2], 
                                                               UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
    else:
        validation_data = False
        stats_overall['HV Benchmark'] = 0

    
    '''Main folder path'''
    if Decisions["Choice_relative_results_referencing"]:
        path_parent_folder = Path(os.path.dirname(os.getcwd()))
    else:
        path_parent_folder = Path("C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS")
        if not os.path.isdir(path_parent_folder):
            path_parent_folder = Path("C:/Users/gunth/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS")
        if not os.path.isdir(path_parent_folder):
            path_parent_folder = Path(os.path.dirname(os.getcwd()))
    
    path_results = path_parent_folder / ("Results/Results_"+
                                         name_input_data+
                                         "/"+name_input_data+
                                         "_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S")+
                                         " "+parameters_SA_routes['method']+
                                         f"_{UTNDP_problem_1.add_text}")
    
    '''Save the parameters used in the runs'''
    if not (path_results / "Parameters").exists():
        os.makedirs(path_results / "Parameters")

    # json.dump(UTNDP_problem_1.problem_inputs.__dict__, open(path_results / "Parameters" / "parameters_input.json", "w"), indent=4) # saves the parameters in a json file
    # json.dump(UTNDP_problem_1.problem_constraints.__dict__, open(path_results / "Parameters" / "parameters_constraints.json", "w"), indent=4)
    # json.dump(UTNDP_problem_1.problem_SA_parameters.__dict__, open(path_results / "Parameters" / "parameters_SA_routes.json", "w"), indent=4)
    # json.dump(Decisions, open(path_results / "Parameters" / "Decisions.json", "w"), indent=4)
    with open(path_results / "Parameters" / "parameters_input.json", "w") as f:
        json.dump(UTNDP_problem_1.problem_inputs.__dict__, f, indent=4) # saves the parameters in a json file
    with open(path_results / "Parameters" / "parameters_constraints.json", "w") as f:
        json.dump(UTNDP_problem_1.problem_constraints.__dict__, f, indent=4)
    with open(path_results / "Parameters" / "parameters_SA_routes.json", "w") as f:
        json.dump(UTNDP_problem_1.problem_SA_parameters.__dict__, f, indent=4)
    with open(path_results / "Parameters" / "Decisions.json", "w") as f:
        json.dump(UTNDP_problem_1.Decisions, f, indent=4)
    
    if Decisions["Choice_conduct_sensitivity_analysis"]:
        with open(path_results / "Parameters" / 'Sensitivity_list.txt', 'w') as f:
            try:
                json.dump(sensitivity_list, f, indent=4)
            except:
                print("Sensitivity_list is not loaded")
    
    # %% Generate an initial feasible solution
    #routes_R = gf.generate_initial_feasible_route_set(mx_dist, UTNDP_problem_1.problem_constraints.__dict__)
    routes_R = gc.Routes.return_feasible_route_robust(UTNDP_problem_1)
    
    if UTNDP_problem_1.problem_constraints.con_r != len(routes_R): # if the constraint was leveraged, update constraints
        UTNDP_problem_1.problem_constraints.con_r = len(routes_R)
        print("Number of allowed routes constraint updated to", UTNDP_problem_1.problem_constraints.con_r)
    
    # %% Simulated Annealing: Initial solutions
    '''Initial solutions'''
    
    if Decisions["Choice_import_saved_set"] or Decisions["Load_supplementing_pop"]: # Make true to import a set that is saved
        # df_routes_R_initial_set = pd.read_csv(Decisions["Set_name"]) 
        # df_routes_R_initial_set = df_routes_R_initial_set.drop(df_routes_R_initial_set.columns[0], axis=1)
    
        # routes_R_initial_set = list()
        # for i in range(len(df_routes_R_initial_set)):
        #     routes_R_initial_set.append(gf.convert_routes_str2list(df_routes_R_initial_set.iloc[i,2]))
    
        # Set the correct path for the input data to be loaded  
        pop_to_load_name = "Pop_init_"+route_gen_func_name+"_"+str(Decisions["Pop_size_to_create"])+".pickle"              
        path_input_to_pop = "./Input_Data/"+name_input_data+"/Populations/"+pop_to_load_name
        if os.path.exists(path_input_to_pop):
            path_input_data = Path("./Input_Data/"+name_input_data)
        else: #TODO The else case below is not neccesary, the above is sufficient
            pop_to_load_name = "Pop_init_"+route_gen_func_name+"_"+str(Decisions["Pop_size_to_create"])+".pickle"              
            path_input_to_pop = "Input_Data/"+name_input_data+"/Populations/"+pop_to_load_name
            github_path = "C:/Users/17832020/OneDrive - Stellenbosch University/Documents/GitHub"
            if os.path.exists(github_path+"/DSS-Main/"+path_input_to_pop):
                path_input_data = Path(github_path+"/DSS-Main/Input_Data/"+name_input_data)
            elif os.path.exists(github_path+"/DSS_Main/"+path_input_to_pop):
                path_input_data = Path(github_path+"/DSS_Main/Input_Data/"+name_input_data)
            elif os.path.exists("C:/Users/gunth/OneDrive/Documents/GitHub/DSS-Main/"+path_input_to_pop):
                github_path = "C:/Users/gunth/OneDrive/Documents/GitHub"
                path_input_data = Path(github_path+"/DSS-Main/Input_Data/"+name_input_data)
            elif os.path.exists("C:/Users/17832020/Documents/GitHub/DSS_Main/"+path_input_to_pop):
                 github_path = "C:/Users/17832020/Documents/GitHub"
                 path_input_data = Path(github_path+"/DSS_Main/Input_Data/"+name_input_data)
            else:    
                path_input_data = path_parent_folder / ("DSS Main/Input_Data/"+name_input_data)
                print("No specified path found. Try copying the instance data into a folder listed above.")
        
        '''Load initial solutions from populations'''

        # Load and save initial population
        path_populations = path_input_data/"Populations"
        pop_loaded = gf_p.load_UTRP_pop_or_create("Pop_init_"+route_gen_func_name+"_"+str(Decisions["Pop_size_to_create"]), path_populations, UTNDP_problem_1, route_gen_funcs[route_gen_func_name], fn_obj_2, pop_size_to_create=Decisions["Pop_size_to_create"])
        
        if Decisions["Load_supplementing_pop"]:
            pop_sup_loaded = gf_p.load_UTRP_supplemented_pop_or_create("Pop_sup_"+route_gen_func_name+"_"+str(Decisions["Pop_size_to_create"]), path_populations, UTNDP_problem_1,route_gen_funcs[route_gen_func_name], fn_obj_2, pop_loaded)
            pop_1 = pop_sup_loaded
        
        else:
            pop_1 = pop_loaded
                    
        if Decisions["Obj_func_disruption"]:
            print("Recalculating objectives for initial solutions")
            f_compare_route = fn_obj_2(UTNDP_problem_1.route_compare, UTNDP_problem_1)
            for sol_i in range(len(pop_1.variables)):
                pop_1.variables[sol_i] = UTNDP_problem_1.route_compare
                pop_1.objectives[sol_i] = f_compare_route
                
        #pop_1 = gc.PopulationRoutes(UTNDP_problem_1)  
        #pop_1.generate_or_load_initial_population(UTNDP_problem_1, fn_obj_2, route_gen_func=route_gen_funcs[route_gen_func_name], pop_choices=pop_sup_loaded)
        
        # Get non-dominated initial population
        pop_size = UTNDP_problem_1.problem_SA_parameters.number_of_initial_solutions
        pop_1.objs_norm = ga.normalise_data_UTRP(pop_1.objectives, UTNDP_problem_1)
        survivor_indices = gf.get_survivors_norm(pop_1, pop_size)
        gf.keep_individuals(pop_1, survivor_indices)
        pop_1.population_size = pop_size
        
        routes_R_initial_set = pop_1.variables # create the list
        
        # create the dataframe
        df_routes_R_initial_set =  pd.DataFrame(columns=["f_1","f_2","R_x"])   
        for i in range(len(routes_R_initial_set)):
            #f_new = ev.evalObjs(routes_R_initial_set[i], UTNDP_problem_1.problem_data.mx_dist, UTNDP_problem_1.problem_data.mx_demand, UTNDP_problem_1.problem_inputs.__dict__)
            f_new = pop_1.objectives[i]
            df_routes_R_initial_set.loc[i] = [f_new[0], f_new[1], gf.convert_routes_list2str(routes_R_initial_set[i])]
    
        print("Initial route set imported with size: "+str(len(routes_R_initial_set)))
        
    else:        
        if Decisions["Choice_generate_initial_set"]:
            '''Generate initial route sets for input as initial solutions'''
            routes_R_initial_set, df_routes_R_initial_set = gf_p.generate_initial_route_sets(UTNDP_problem_1,pareto_efficient=False)          

        else: # use this alternative if you want to use another set as input
            """Standard route to begin with"""
            routes_R_initial_set = list()
            routes_R_initial_set.append(gf.convert_routes_str2list("5-7-9-12*9-7-5-3-4*0-1-2-5-14-6*13-9-6-14-8*1-2-5-14*9-10-11-3*"))
          
            
            df_routes_R_initial_set =  pd.DataFrame(columns=["f_1","f_2","Routes"])   
            for i in range(len(routes_R_initial_set)):
                #f_new = ev.evalObjs(routes_R_initial_set[i], UTNDP_problem_1.problem_data.mx_dist, UTNDP_problem_1.problem_data.mx_demand, UTNDP_problem_1.problem_inputs.__dict__)
                f_new = fn_obj_2(routes_R_initial_set[i], UTNDP_problem_1)
                df_routes_R_initial_set.loc[i] = [f_new[0], f_new[1], gf.convert_routes_list2str(routes_R_initial_set[i])]
            
            
        
    # %% Simulated Annealing Algorithm for each of the initial route sets #########################
    '''Simulated Annealing Algorithm for each of the initial route sets'''
    run_nr_counter = range(UTNDP_problem_1.problem_SA_parameters.number_of_runs) # default values
    
    if Decisions["Choice_normal_run"]:
        run_nr_counter = range(len(routes_R_initial_set)) # sets the tests ito for loops to run
        UTNDP_problem_1.add_text = f"Normal_run_{UTNDP_problem_1.problem_SA_parameters.number_of_runs}_routes_{len(df_routes_R_initial_set)}"
    if Decisions["Choice_conduct_sensitivity_analysis"]:
        run_nr_counter = range(UTNDP_problem_1.problem_SA_parameters.number_of_runs) # sets the tests ito for loops to run
        
    for run_nr in run_nr_counter:
        route_set_nr_counter = [run_nr]
        if Decisions["Choice_first_init_set_only"]: route_set_nr_counter = [0] # standardizes the sensitivity analysis to only the first route set
        
        for route_set_nr in route_set_nr_counter:
            print("Started route set number "+str(route_set_nr + 1)+" ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
            stats['begin_time'] = datetime.datetime.now() # enter the begin time
            stats['run_number'] = f"{run_nr + 1}.{route_set_nr}"
        
            if route_set_nr < len(routes_R_initial_set): 
                routes_R = routes_R_initial_set[route_set_nr] # Choose the initial route set to begin with
            else: # if route_ser_number is more than entries in the route set
                routes_R = random.choice(routes_R_initial_set)
            
            '''Initiate algorithm'''
            # Reset the mutation ratios
            UTNDP_problem_1.mutation_ratio = [1/len(mut_functions) for _ in mut_functions]

            
            epoch = 1 # Initialise the epoch counter
            total_iterations = 0 
            poor_epoch = 0 # Initialise the number of epochs without an accepted solution
            attempts = 0 # Initialise the number of attempts made without accepting a solution
            accepts = 0 # Initialise the number of accepts made within an epoch
            reheated = 0 # Initialise the number of times reheated
            SA_Temp = UTNDP_problem_1.problem_SA_parameters.Temp # initialise the starting temperature
            
            df_archive = pd.DataFrame(columns=["f_1","f_2","R_x"]) # create an archive in the correct format
            counter_archive = 1
            # df_SA_analysis = pd.DataFrame(columns = ["f_1",\
            #                                          "f_2",\
            #                                          "HV",\
            #                                          "Temperature",\
            #                                          "C_epoch_number",\
            #                                          "L_iteration_per_epoch",\
            #                                          "A_num_accepted_moves_per_epoch",\
            #                                          "eps_num_epochs_without_accepting_solution",\
            #                                          "Route",\
            #                                          "Attempts"]) # create a df to keep data for SA Analysis
                            
            f_cur = fn_obj_2(routes_R, UTNDP_problem_1)
            df_archive.loc[0] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)]
            HV = gf_p.norm_and_calc_2d_hv(df_archive.iloc[:,0:2], max_objs, min_objs)
            # df_SA_analysis.loc[0] = [f_cur[0], f_cur[1], HV,\
            #                    SA_Temp, epoch, 0, 0, 0, gf.convert_routes_list2str(routes_R), 0]
            
            # Initial list of dictionaries to contain the SA analysis   
            ld_SA_analysis = ga.add_UTRP_SA_data_ld(f_cur[0], f_cur[1], HV, SA_Temp, epoch, 
                                                            0, 0, 0, gf.convert_routes_list2str(routes_R), 0)
              
            # Create dataframe for mutations      
            ld_mut = [{"Mut_nr":0, "Mut_successful":0, "Mut_repaired":0, "Route":gf.convert_routes_list2str(routes_R), "Acceptance":-1}]
            
            ld_mut_summary = []
            ld_mut_ratios = [{k:v for (k,v) in zip(["Total_iterations"]+UTNDP_problem_1.mutation_names,[0]+list(UTNDP_problem_1.mutation_ratio))}]
            ld_mut_summary_temp_init = False # initialisation boolean for the list of dictionaries
            #df_mut_ratios = pd.DataFrame(columns=(["Total_iterations"]+UTNDP_problem_1.mutation_names)) # dataframe to print mutation ratios #TODO Delete after ld implementation
            #df_mut_ratios.loc[0] = [0]+list(UTNDP_problem_1.mutation_ratio) #TODO: Convert to list of dictionaries

            
            print(f'Epoch:{epoch-1} \tHV:{round(HV, 4)}')
             
            
            while poor_epoch <= UTNDP_problem_1.problem_SA_parameters.max_poor_epochs and total_iterations <= UTNDP_problem_1.problem_SA_parameters.max_total_iterations and epoch <= UTNDP_problem_1.problem_SA_parameters.max_epochs:
                accepts = 0 # Initialise the accepts
                iteration_t = 1 # Initialise the number of iterations 
                poor_epoch_flag = True # sets the poor epoch flag, and lowered when solution added to the archive
                prob_acceptance_list = []
                while (iteration_t <= UTNDP_problem_1.problem_SA_parameters.max_iterations_t) and (accepts < UTNDP_problem_1.problem_SA_parameters.min_accepts):
                    '''Generate neighbouring solution'''
                    #routes_R_new = gf.perturb_make_small_change(routes_R, UTNDP_problem_1.problem_constraints.con_r, UTNDP_problem_1.mapping_adjacent)
                    mut_output = gf_p.mutate_overall_routes_all_smart_SA(routes_R, UTNDP_problem_1)
                    routes_R_new = mut_output['Route']
                    
                    while not gf.test_all_four_constraints(routes_R_new, UTNDP_problem_1): # tests whether the new route is feasible
                        for i in range(UTNDP_problem_1.problem_SA_parameters.Feasibility_repair_attempts): # this tries to fix the feasibility, but can be time consuming, 
                                                # could also include a "connectivity" characteristic to help repair graph
                            #routes_R_new = gf.perturb_make_small_change(routes_R_new, UTNDP_problem_1.problem_constraints.con_r, UTNDP_problem_1.mapping_adjacent)
                            # mut_output = gf_p.mutate_overall_routes_all_smart_SA(routes_R_new, UTNDP_problem_1)
                            # mut_output['Mut_repaired'] = 1
                            # routes_R_new = mut_output['Route']
                            
                            routes_R_new = gf.repair_add_missing_from_terminal_multiple(routes_R_new, UTNDP_problem_1)
                            mut_output['Route'] = routes_R_new
                            mut_output['Mut_repaired'] = 1
                            
                            if gf.test_all_four_constraints(routes_R_new, UTNDP_problem_1):
                                break
                        #routes_R_new = gf.perturb_make_small_change(routes_R, UTNDP_problem_1.problem_constraints.con_r, UTNDP_problem_1.mapping_adjacent) # if unsuccesful, start over
                        mut_output = gf_p.mutate_overall_routes_all_smart_SA(routes_R, UTNDP_problem_1)
                        routes_R_new = mut_output['Route']
                    

                    

                    # Evaluate objective function of new solution
                    f_new = fn_obj_2(routes_R_new, UTNDP_problem_1)
                    HV = gf_p.norm_and_calc_2d_hv(df_archive.iloc[:,0:2], max_objs, min_objs)
                    
                    ld_SA_analysis = ga.add_UTRP_SA_data_ld(f_new[0], f_new[1], HV, SA_Temp, epoch, 
                                                            iteration_t, accepts, poor_epoch, gf.convert_routes_list2str(routes_R), 
                                                            attempts, ld_data=ld_SA_analysis)
                        
                    total_iterations = total_iterations + 1 # increments the total iterations for stopping criteria
                
                    '''Test solution acceptance and add to archive if accepted and non-dominated'''
                    mut_output['Acceptance'] = -1 # set default acceptance value
                    prob_to_accept = gf_p.prob_accept_neighbour(df_archive, f_cur, f_new, SA_Temp)
                    prob_acceptance_list.append(prob_to_accept)
                    if random.uniform(0,1) < prob_to_accept: # probability to accept neighbour solution as current solution
                        routes_R = routes_R_new
                        f_cur = f_new
                        accepts = accepts + 1 
                        mut_output['Acceptance']=0
                        
                        if gf.test_min_min_non_dominated(df_archive, f_cur[0], f_cur[1]): # means solution is undominated
                            df_archive.loc[counter_archive] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)] # adds the new solution
                            counter_archive = counter_archive + 1 # this helps with speed 
                            df_archive = df_archive[gf.is_pareto_efficient(df_archive.iloc[:,0:2].values, True)] # remove dominated solutions from archive
                            accepts = accepts - 1 # updates the number of solutions accepted in the epoch
                            poor_epoch_flag = False # lowers flag when a solution is added to the archive
                            poor_epoch = 0 # resets the number of epochs without acceptance
                            mut_output['Acceptance']=1
                    
                    else:
                        if reheated < UTNDP_problem_1.problem_SA_parameters.max_reheating_times: #determines if the max number of reheats have occured
                            attempts = attempts + 1
                            
                            if attempts > UTNDP_problem_1.problem_SA_parameters.max_attempts:
                                SA_Temp = UTNDP_problem_1.problem_SA_parameters.Reheating_rate*SA_Temp # update temperature based on cooling schedule
                                reheated = reheated + 1
                                print(f"Reheated:{reheated}/{UTNDP_problem_1.problem_SA_parameters.max_reheating_times}")
                                break # gets out of the inner while loop
                    
                    iteration_t = iteration_t + 1
                    
                    '''Mutation evaluation'''
                    ld_mut.append(mut_output)
                    mut_update_interval = UTNDP_problem_1.problem_SA_parameters.mutation_update_interval
                    if total_iterations>=mut_update_interval:
                        ld_mut_temp = ld_mut[-mut_update_interval:]
                        df_mut_temp = pd.DataFrame.from_dict(ld_mut_temp)
                        df_mut_temp.drop(['Route'], axis='columns', inplace=True)
                        
                        if not ld_mut_summary_temp_init:
                            ld_mut_summary_temp = ga.get_mutations_summary_ld_for_SA(df_mut_temp, len(UTNDP_problem_1.mutation_functions), total_iterations)
                            ld_mut_summary_temp_init = True
                        else:
                            ga.update_mutations_summary_ld_for_SA(df_mut_temp, ld_mut_summary_temp) # fast update

                        ld_mut_summary.extend(ld_mut_summary_temp)

                        # Update mutation ratios
                        if Decisions["Update_mutation_ratios"]:
                            
                            if Decisions["mut_update_func"] == "AMALGAM":
                                df_mut_summary = pd.DataFrame.from_dict(ld_mut_summary[-len(UTNDP_problem_1.mutation_functions):]) # get only last mutations
                                gf_p.update_mutation_ratio_amalgam_for_SA(df_mut_summary, UTNDP_problem_1)
                            
                            elif Decisions["mut_update_func"] == "AMALGAM_every_n":
                                if total_iterations % mut_update_interval == 0:
                                    df_mut_summary = pd.DataFrame.from_dict(ld_mut_summary[-len(UTNDP_problem_1.mutation_functions):]) # get only last mutations
                                    gf_p.update_mutation_ratio_amalgam_for_SA(df_mut_summary, UTNDP_problem_1)
                            
                            elif Decisions["mut_update_func"] == "Counts_normal":
                                if mut_output['Acceptance'] == -1:
                                    if UTNDP_problem_1.mutation_ratio_counts[mut_output['Mut_nr']-1] > 1:
                                        UTNDP_problem_1.mutation_ratio_counts[mut_output['Mut_nr']-1] -= 1
                                elif mut_output['Acceptance'] == 1:
                                    if UTNDP_problem_1.mutation_ratio_counts[mut_output['Mut_nr']-1] < 100:
                                        UTNDP_problem_1.mutation_ratio_counts[mut_output['Mut_nr']-1] += 1
                                        UTNDP_problem_1.mutation_ratio = UTNDP_problem_1.mutation_ratio_counts / np.sum(UTNDP_problem_1.mutation_ratio_counts)    
                                    
                    ld_mut_ratios.extend([{k:v for (k,v) in zip(["Total_iterations"]+UTNDP_problem_1.mutation_names,[total_iterations]+list(UTNDP_problem_1.mutation_ratio))}])
                    #df_mut_ratios.loc[total_iterations] = [total_iterations]+list(UTNDP_problem_1.mutation_ratio)

                
                
                '''Max accepts reached and continue''' # end of inner while loop
                if poor_epoch_flag:
                    poor_epoch = poor_epoch + 1 # update number of epochs without an accepted solution
                
                print(f'Epoch:{epoch} \tTemp:{round(SA_Temp,4)} \tHV:{round(HV, 4)} [{stats_overall["HV Benchmark"]:.4f}] \tAccepts:{accepts} \tAttempts:{attempts} \tPoor_epoch:{poor_epoch}/{UTNDP_problem_1.problem_SA_parameters.max_poor_epochs} \tTotal_i:{total_iterations}[{iteration_t}] \t P_accept:{round(sum(prob_acceptance_list)/len(prob_acceptance_list),4)}')
    
                '''Update parameters'''
                SA_Temp = UTNDP_problem_1.problem_SA_parameters.Cooling_rate*SA_Temp # update temperature based on cooling schedule
                epoch = epoch + 1 # Increase Epoch counter
                attempts = 0 # resets the attempts
                
                # Test whether HV is still improving
                iter_compare = UTNDP_problem_1.problem_SA_parameters.iter_compare_HV
                HV_improvement_th = UTNDP_problem_1.problem_SA_parameters.HV_improvement_th
                if total_iterations > iter_compare:
                    HV_diff = ld_SA_analysis[-1]['HV'] - ld_SA_analysis[-iter_compare-1]['HV']
                    if HV_diff < HV_improvement_th:
                        stats['Termination'] = 'Non-improving_HV'
                        print(f'Run terminated by non-improving HV after Iter {total_iterations} [Iter comp:{iter_compare} | HV diff: {HV_diff:.6f}')
                        break
                    
                if epoch == 4 or epoch == 30:
                    # Calculate time projections for runs
                    start_time_run = stats['begin_time'] # TIMING FUNCTION # stats_overall['execution_start_time']
                    end_time = datetime.datetime.now() # TIMING FUNCTION
            
                    # Determine and print projections
                    diff = end_time - start_time_run
                    diff_sec = float(str(diff.seconds)+"."+str(diff.microseconds))
                    tot_iter = UTNDP_problem_1.problem_SA_parameters.max_total_iterations*UTNDP_problem_1.problem_SA_parameters.number_of_runs
                    completed_iter =  (run_nr)*UTNDP_problem_1.problem_SA_parameters.max_total_iterations + total_iterations
                    avg_time = diff_sec/(total_iterations)
                    ga.time_projection_intermediate(avg_time, tot_iter, completed_iter,
                                                    t_now=datetime.datetime.now(),
                                                    print_iter_info=True) # prints the time projection of the algorithm
                
            del f_cur, f_new, accepts, attempts, SA_Temp, epoch, poor_epoch, iteration_t, counter_archive
        
            
         # %% Saving Results per run
            if Decisions["Choice_print_results"]:
                stats['end_time'] = datetime.datetime.now() # save the end time of the run
                
                print("Run number "+str(run_nr+1)+" duration: "+ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']))
            
                stats['duration'] = stats['end_time'] - stats['begin_time'] # calculate and save the duration of the run
                stats['begin_time'] = stats['begin_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
                stats['end_time'] =  stats['end_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
                stats['HV obtained'] = HV
                
                
                '''Write all results and parameters to files'''
                
                '''Sub folder path'''
                path_results_per_run = path_results / (f"Run_{run_nr + 1}")
                if not path_results_per_run.exists():
                    os.makedirs(path_results_per_run)
                
                df_SA_analysis = pd.DataFrame.from_dict(ld_SA_analysis)
                df_SA_analysis.to_csv(path_results_per_run / "SA_Analysis.csv")
                df_archive.to_csv(path_results_per_run / "Archive_Routes.csv")
                
                pickle.dump(stats, open(path_results_per_run / "stats.pickle", "ab"))
                
                with open(path_results_per_run / "Run_summary_stats.csv", "w") as archive_file:
                    w = csv.writer(archive_file)
                    for key, val in {**UTNDP_problem_1.problem_inputs.__dict__, **UTNDP_problem_1.problem_constraints.__dict__, **UTNDP_problem_1.problem_SA_parameters.__dict__, **stats}.items():
                        w.writerow([key, val])
                    del key, val
            
                # %% Display and save results per run'''
                try:
                    df_mut_ratios = pd.DataFrame.from_dict(ld_mut_ratios)
                    df_mut_ratios.to_csv(path_results_per_run / "Mut_ratios.csv")
                    
                    gv.save_results_analysis_mut_fig_UTRP_SA(df_routes_R_initial_set, df_archive, validation_data, 
                     df_SA_analysis, df_mut_ratios, name_input_data, 
                     path_results_per_run, 
                     ["f_1", "f_2", "f1_ATT", "f2_TRT"],
                     stats_overall['HV Benchmark'], type_mut='stacked_smooth')
                                             
                except PermissionError: pass
            
    # %% Save results after all runs
    if Decisions["Choice_print_results"]:
        '''Save the summarised results'''
        df_overall_pareto_set = ga.group_pareto_fronts_from_model_runs(path_results, UTNDP_problem_1.problem_inputs.__dict__).iloc[:,1:]
        df_overall_pareto_set = df_overall_pareto_set[gf.is_pareto_efficient(df_overall_pareto_set.iloc[:,0:2].values, True)] # reduce the pareto front from the total archive
        df_overall_pareto_set = df_overall_pareto_set.sort_values(by='f_1', ascending=True) # sort
        df_overall_pareto_set.to_csv(path_results / "Overall_Pareto_set.csv")   # save the csv file
        
        '''Save the stats for all the runs'''
        df_routes_R_initial_set.to_csv(path_results / "Routes_initial_set.csv")
        df_durations = ga.get_stats_from_model_runs(path_results)
        
        stats_overall['execution_end_time'] =  datetime.datetime.now()
        
        stats_overall['total_model_runs'] = run_nr + 1
        stats_overall['average_run_time'] = str(df_durations["Duration"].mean())
        stats_overall['total_duration'] = stats_overall['execution_end_time']-stats_overall['execution_start_time']
        stats_overall['execution_start_time'] = stats_overall['execution_start_time'].strftime("%m/%d/%Y, %H:%M:%S")
        stats_overall['execution_end_time'] = stats_overall['execution_end_time'].strftime("%m/%d/%Y, %H:%M:%S")
        stats_overall['HV initial set'] = gf_p.norm_and_calc_2d_hv(df_routes_R_initial_set.iloc[:,0:2], max_objs, min_objs)
        stats_overall['HV obtained'] = gf_p.norm_and_calc_2d_hv(df_overall_pareto_set.iloc[:,0:2], max_objs, min_objs)
  
        # df_durations.loc[len(df_durations)] = ["Average", df_durations["Duration"].mean()]
        df_durations.to_csv(path_results / "Run_durations.csv")
        del df_durations
        
        with open(path_results / "Stats_overall.csv", "w") as archive_file:
            w = csv.writer(archive_file)
            for key, val in {**stats_overall, **UTNDP_problem_1.problem_inputs.__dict__, **UTNDP_problem_1.problem_constraints.__dict__, **UTNDP_problem_1.problem_SA_parameters.__dict__}.items():
                w.writerow([key, val])
            del key, val
        
        for _ in range(5):
            try:    
                ga.get_sens_tests_stats_from_UTRP_SA_runs(path_results)
                gv.save_all_mutation_stats_and_plots(path_results) # gets and prints the mutation stats
                gv.save_all_obj_stats_and_plots(path_results) # gets and prints the objective performance stats
                # ga.capture_all_runs_HV_over_iterations_from_UTRP_SA(path_results)
                
                gv.save_final_avgd_results_analysis_SA(df_routes_R_initial_set, df_overall_pareto_set, validation_data, 
                                    pd.read_csv((path_results/'Performance/Avg_obj_performances.csv')), 
                                    pd.read_csv((path_results/'Mutations/Avg_mut_ratios.csv')), # can use 'Smoothed_avg_mut_ratios.csv' for a double smooth visualisation
                                    name_input_data, 
                                    path_results, labels,
                                    stats_overall['HV Benchmark'], 'stacked_smooth')
                
                break
            except PermissionError:
                time.sleep(5)
                continue
        
        gf.print_extreme_solutions(df_overall_pareto_set, stats_overall['HV obtained'], stats_overall['HV Benchmark'], name_input_data, UTNDP_problem_1, path_results)
        
#    return df_archive

# %% Sensitivity analysis
''' Sensitivity analysis tests'''

# if False:
if __name__ == "__main__":
    
    if Decisions["Choice_conduct_sensitivity_analysis"]:
        start = time.perf_counter()
 
         # define empty list
        sensitivity_list = []

        # open file and read the content in a list
        with open(("./Input_Data/"+name_input_data+"/Sensitivity_list.txt"), 'r') as filehandle:
            sensitivity_list = json.load(filehandle)  
            print(f"LOADED: Sensitivity list:\n{sensitivity_list}\n")
        if sens_to == -1:
            sensitivity_list = sensitivity_list[sens_from:]  
        else:
            sensitivity_list = sensitivity_list[sens_from:sens_to]
        print(f"TRUNCATED: Sensitivity list:\n{sensitivity_list}")
        
        # Print specific test counters
        if test_counters == []:
            print("Testing all combinations")
        else:
            print(f"Testing only variables: {test_counters}")
 
        for parameter_index in range(len(sensitivity_list)):
            if sensitivity_list[parameter_index][0] == "parameters_SA_routes": 
                sensitivity_list[parameter_index][0] = parameters_SA_routes
            elif sensitivity_list[parameter_index][0] == "Decisions":
                sensitivity_list[parameter_index][0] = Decisions
            else:
                print(f"Sensitivity list not loaded correctly:\n{sensitivity_list[parameter_index]}")
        
        for sensitivity_test in sensitivity_list:
            parameter_dict = sensitivity_test[0]
            dict_entry = sensitivity_test[1]
            for test_counter in range(2,len(sensitivity_test)):
                if test_counters==[] or test_counter in [x_i + 2 for x_i in test_counters]:
                
                    print("\nTest: {0} = {1}".format(sensitivity_test[1], sensitivity_test[test_counter]))
                    
                    if sensitivity_test[1] in ["crossover_func", "mutation_funcs", "mut_update_func"]:
                        UTNDP_problem_1.add_text = f"{sensitivity_test[1]}_{test_counter-2}"
                    else:
                        UTNDP_problem_1.add_text = f"{sensitivity_test[1]}_{round(sensitivity_test[test_counter],4)}"
                    
                    temp_storage = parameter_dict[dict_entry]
                    
                    # Set new parameters
                    parameter_dict[dict_entry] = sensitivity_test[test_counter]
        
                    # Update problem instance
                    UTNDP_problem_1.problem_constraints = gc.Problem_inputs(parameters_constraints)
                    UTNDP_problem_1.problem_SA_parameters = gc.Problem_metaheuristic_inputs(parameters_SA_routes)
                    UTNDP_problem_1.Decisions = Decisions
                    
                    # Run model
                    main(UTNDP_problem_1)
                    
                    # Reset the original parameters
                    parameter_dict[dict_entry] = temp_storage
        
        finish = time.perf_counter()
        
        print(f'Finished in {round(finish-start, 6)} second(s)')
        
    else:
        #pass
        print("Normal run initiated")
        UTNDP_problem_1.add_text = "Long_run"
        main(UTNDP_problem_1) 