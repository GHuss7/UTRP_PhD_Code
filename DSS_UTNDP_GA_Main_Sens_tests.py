# -*- coding: utf-8 -*-
"""
Created on Tue Dec 03 10:54:00 2019

@author: 17832020
"""

# %% Import Libraries
import os
import re
import string
from pathlib import Path
import csv
import json
import pickle
import pandas as pd 
import numpy as np
import math
from math import inf
#import pygmo as pg
import random
import copy
import datetime
import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import igraph as ig
import networkx as nx
import concurrent.futures

# %% Import personal functions
import DSS_Admin as ga
import DSS_UTNDP_Functions as gf_p
import DSS_UTNDP_Functions_c as gf
import DSS_UTNDP_Classes as gc
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import EvaluateRouteSet as ev
import DSS_K_Shortest_Paths as ksp

    
#%% Pymoo functions
from pymoo.util.function_loader import load_function
from pymoo.util.dominator import Dominator
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.population import Population

from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import Dominator
from pymoo.util.misc import find_duplicates, has_feasible
#from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning) # find VisibleDeprecationWarning
    
# %% Load the respective files
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

                   '1_1_Mandl6_GA_Initial_solutions', #11
                    '1_2_Mumford0_GA_Initial_solutions', #12
                    '1_3_Mumford1_GA_Initial_solutions', #13
                    '2_1_Mandl6_GA_Crossover', #14
                    '2_2_Mumford0_GA_Crossover', #15
                    '2_3_Mumford1_GA_Crossover', #16
                    '3_1_Mandl6_GA_Mutations', #17
                    '3_2_Mumford0_GA_Mutations', #18
                    '3_3_Mumford1_GA_Mutations', #19
                    '4_1_Mandl6_GA_Update_mut_ratio', #20
                    '4_2_Mumford0_GA_Update_mut_ratio', #21
                    '4_3_Mumford1_GA_Update_mut_ratio', #22
                    '5_1_Mandl6_GA_repair_func', #23
                    '5_2_Mumford0_GA_repair_func', #24
                    '5_3_Mumford1_GA_repair_func', #25
                    '6_1_Mandl6_GA_Mut_threshold', #26
                    '6_2_Mumford0_GA_Mut_threshold', #27
                    '6_3_Mumford1_GA_Mut_threshold', #28
                    '7_1_Mandl6_GA_Pop_size', #29
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
                    
                    '11_1_Mandl6_GA_Very_Long_run', #49
                    '11_2_Mumford0_GA_Very_Long_run', #50
                    '11_3_Mumford1_GA_Very_Long_run', #51
                    '11_4_Mumford2_GA_Very_Long_run', #52
                    '11_5_Mumford3_GA_Very_Long_run', #53
                    '11_6_Mandl4_GA_Very_Long_run', #54
                    '11_7_Mandl7_GA_Very_Long_run', #55
                    '11_8_Mandl8_GA_Very_Long_run', #56

                    '12_1_Mandl6_GA_Obj_dis', #57
                    '12_2_Mumford0_GA_Obj_dis', #58
                    '12_3_Mumford1_GA_Obj_dis', #59
                    '12_4_Mumford2_GA_Obj_dis', #60
                    '12_5_Mumford3_GA_Obj_dis', #61

                   '0_0_Mandl6_GA_Tester',
                   '0_5_Mumford3_GA_Time_tester'
                   
                   ][59]   # set the name of the input data

# Set test paramaters
sens_from = 0 # sets the entire list that should be used as input. Lists by be broken down in smaller pieces for convenience
sens_to = (sens_from + 1) if False else -1
test_counters = [] # empty list means all, filled in values means only those tests

# %% Set input parameters
if True:
    Decisions = json.load(open("./Input_Data/"+name_input_data+"/Decisions.json"))

else:
    Decisions = {
    "Choice_print_results" : True, 
    "Choice_import_dictionaries" : True,
    "Choice_relative_results_referencing" : False,
    "Choice_print_full_data_for_analysis" : True,
    "Pop_size_to_create" : 2000,
    "Log_prints_every" : 20, # every n generations a log should be printed
    "CSV_prints_every" : 100, # every n generations a csv should be printed
    "PDF_prints_every" : 20, # every n generations a csv should be printed
    "Measure_APD" : False, # measure Average Population Diversity
    
    "Additional_text" : "Tests",
    "Choice_conduct_sensitivity_analysis" : False,
    "Load_supplementing_pop" : True,
    "Obj_func_disruption" : False,
    "Update_mutation_ratios" : False,
    "Repair_multiple" : False,
    
    "route_gen_func" : "KSP_unseen_robust_prob",
    "crossover_func" : "Unseen_probabilistic_replace_subsets",
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
    }
    
#%% Set functions to use
route_gen_funcs = {"KSP_unseen_robust" : gc.Routes.return_feasible_route_robust_k_shortest,
                   "KSP_unseen_robust_prob_10000" : gc.Routes.return_feasible_route_robust_k_shortest_probabilistic,
                   "KSP_unseen_robust_prob" : gc.Routes.return_feasible_route_robust_k_shortest_probabilistic,
                    "Greedy_demand" : gc.Routes.return_feasible_route_set_greedy_demand,
                    "Unseen_robust" : gc.Routes.return_feasible_route_robust}
assert Decisions["route_gen_func"] in route_gen_funcs.keys()
route_gen_func_name = Decisions["route_gen_func"]

crossover_funcs = {"Mumford" : gf.crossover_mumford,
                   "Unseen_probabilistic" : gf.crossover_unseen_probabilistic,
                   "Mumford_replace_subsets_ksp" : gf.crossover_mumford_rem_subsets_ksp,
                   "Unseen_probabilistic_replace_subsets_ksp" : gf.crossover_unseen_probabilistic_rem_subsets_ksp,
                   "Mumford_replace_subsets" : gf.crossover_mumford_rem_subsets,
                   "Unseen_probabilistic_replace_subsets" : gf.crossover_unseen_probabilistic_rem_subsets}
assert Decisions["crossover_func"] in crossover_funcs.keys()
crossover_func_name = Decisions["crossover_func"]

mutations_all = {#"No_mutation" : gf.no_mutation,
                "Intertwine_two" : gf.mutate_routes_two_intertwine, 
                "Add_vertex" : gf.add_vertex_to_terminal,
                "Delete_vertex" : gf.remove_vertex_from_terminal,
                
                "Invert_path_vertices" : gf.mut_invert_route_vertices,
                "Insert_inside_vertex" : gf.mut_add_vertex_inside_route,
                "Delete_inside_vertex" : gf.mut_delete_vertex_inside_route,
                
                "Relocate_inside_vertex" : gf.mut_relocate_vertex_inside_route,
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
                }

mutations = {k : v for (k,v) in mutations_all.items() if k in Decisions["mutation_funcs"]}

all_functions_dict = {"Mut_"+k : v.__name__ for (k,v) in mutations.items()}
all_functions_dict = {'Route_gen':route_gen_func_name,**all_functions_dict, 'Crossover':crossover_func_name}
    
mutations_dict = {i+1:{"name":k, "func":v} for i,(k,v) in zip(range(len(mutations)),mutations.items())}
mut_functions = [v['func'] for (k,v) in mutations_dict.items()]
mut_names = [v['name'] for (k,v) in mutations_dict.items()]

# %% Load the respective files
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)

#%% Load and set problem parameters #######################################
if Decisions["Choice_import_dictionaries"]:
    parameters_constraints = json.load(open("./Input_Data/"+name_input_data+"/parameters_constraints.json"))
    parameters_input = json.load(open("./Input_Data/"+name_input_data+"/parameters_input.json"))
    if True:
        parameters_GA = json.load(open("./Input_Data/"+name_input_data+"/parameters_GA.json"))
    else:
        '''State the various GA input parameters for frequency setting''' 
        parameters_GA={
        "method" : "GA",
        "termination_criterion" : ["StoppingByEvaluations", "StoppingByNonImprovement", "FirstToBreach"][0],
        "population_size" : 200, #should be an even number STANDARD: 200 (John 2016)
        "generations" : 200, # STANDARD: 200 (John 2016)
        "number_of_runs" : 20, # STANDARD: 20 (John 2016)
        "crossover_probability" : 0.6, 
        "mutation_probability" : 1, # John: 1/|Route set| -> set later
        "mutation_threshold" : 0.01, # Minimum threshold that mutation probabilities can reach
        "mutation_ratio" : 0.5, # Ratio used for the probabilites of mutations applied
        "gen_compare_HV" : 40, # Compare generations for improvement in HV
        "HV_improvement_th": 0.0001, # Treshold that terminates the search
        "crossover_distribution_index" : 5,
        "mutation_distribution_index" : 10,
        "tournament_size" : 2,
        "max_evaluations" : 25000,
        "number_of_variables" : parameters_constraints["con_r"],
        "number_of_objectives" : 2, # this could still be automated in the future
        "Number_of_initial_solutions" : 10000 # number of initial solutions to be generated and chosen from
        }

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
   

 
else:    
    '''State the various parameter constraints''' 
    parameters_constraints = {
    'con_r' : 6,               # number of allowed routes (aim for > [numNodes N ]/[maxNodes in route])
    'con_minNodes' : 2,                        # minimum nodes in a route
    'con_maxNodes' : 8,                       # maximum nodes in a route
    'con_N_nodes' : len(mx_dist),              # number of nodes in the network
    'con_fleet_size' : 40,                     # number of vehicles that are allowed
    'con_vehicle_capacity' : 20,               # the max carrying capacity of a vehicle
    'con_lower_bound' : 0,                 # the lower bound for the problem
    'con_upper_bound' : 1                 # the upper bound for the problem
    }
    
    '''State the various input parameter''' 
    parameters_input = {
    'total_demand' : sum(sum(mx_demand))/2, # total demand from demand matrix
    'n' : len(mx_dist), # total number of nodes
    'wt' : 0, # waiting time [min]
    'tp' : 5, # transfer penalty [min]
    'ref_point_max_f1_ATT' : 32, # max f1_ATT for the Hypervolume calculations
    'ref_point_min_f1_ATT' : 13, # min f1_ATT for the Hypervolume calculations
    'ref_point_max_f2_TRT' : 700, # max f2_TRT for the Hypervolume calculations
    'ref_point_min_f2_TRT' : 94, # min f2_TRT for the Hypervolume calculations
    'walkFactor' : 3, # factor it takes longer to walk than to drive
    'boardingTime' : 0.1, # assume boarding and alighting time = 6 seconds
    'alightingTime' : 0.1, # problem when alighting time = 0 (good test 0.5)(0.1 also works)
    'large_dist' : int(mx_dist.max()), # the large number from the distance matrix
    'alpha_const_inter' : 0.5 # constant for interarrival times relationship 0.5 (Spiess 1989)
    }
    
    '''State the various GA input parameters for frequency setting''' 
    parameters_GA={
    "method" : "GA",
    "population_size" : 400, #should be an even number STANDARD: 200 (John 2016)
    "generations" : 2000, # STANDARD: 200 (John 2016)
    "number_of_runs" : 20, # STANDARD: 20 (John 2016)
    "crossover_probability" : 0.6, 
    "mutation_probability" : 0.95, # John: 1/|Route set| -> set later
    "mutation_ratio" : 0.1, # Ratio used for the probabilites of mutations applied
    "termination_criterion" : "StoppingByEvaluations",
    "crossover_distribution_index" : 5,
    "mutation_distribution_index" : 10,
    "tournament_size" : 2,
    "max_evaluations" : 25000,
    "number_of_variables" : parameters_constraints["con_r"],
    "number_of_objectives" : 2, # this could still be automated in the future
    "Number_of_initial_solutions" : 10000 # number of initial solutions to be generated and chosen from
    }
    
    # Sensitivity analysis lists    
    sensitivity_list = [#[parameters_constraints, "con_r", 4, 6, 7, 8],
                        #[parameters_constraints, "con_minNodes", 2, 3, 4, 5],
                        ["population_size", 10, 20, 50, 100, 150, 200, 300, 400],
                        ["generations", 10, 20, 50, 100, 150, 200, 300, 400],
                        ["crossover_probability", 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                        ["mutation_probability", 0.05, 0.1, 1/parameters_constraints["con_r"], 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                        ["mutation_ratio", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        ["population_size", 10]
                        ]
    
    sensitivity_list = sensitivity_list[sens_from:sens_to] # truncates the sensitivity list


'''Set the reference point for the Hypervolume calculations'''
max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])

        
# %% Define the adjacent mapping of each node
mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes
    
#%% Input parameter tests

# '''Test the inputs for feasibility'''
# Test feasibility if there are enough buses to cover each route once
# if parameters_constraints["con_r"] > parameters_constraints["con_fleet_size"]:
#     print("Warning: Number of available vehicles are less than the number of routes.\n"\
#           "Number of routes allowed set to "+ str(parameters_constraints["con_r"]))
#     parameters_constraints["con_r"] = parameters_constraints["con_fleet_size"]


#%% Define the UTNDP Problem      
UTNDP_problem_1 = gc.UTNDP_problem()
UTNDP_problem_1.problem_data = gc.Problem_data(mx_dist, mx_demand, mx_coords)
UTNDP_problem_1.problem_constraints = gc.Problem_constraints(parameters_constraints)
UTNDP_problem_1.problem_inputs = gc.Problem_inputs(parameters_input)
UTNDP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA)
UTNDP_problem_1.Decisions = Decisions
UTNDP_problem_1.k_short_paths = gc.K_shortest_paths(df_k_shortest_paths)
UTNDP_problem_1.mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes
UTNDP_problem_1.max_objs = max_objs
UTNDP_problem_1.min_objs = min_objs
UTNDP_problem_1.add_text = f"G{parameters_GA['generations']}_P{parameters_GA['population_size']}_R{parameters_GA['number_of_runs']}_{crossover_func_name}" # define the additional text for the file name
UTNDP_problem_1.mutation_functions = mut_functions
UTNDP_problem_1.mutation_names = mut_names
UTNDP_problem_1.mutation_ratio = [1/len(mut_functions) for _ in mut_functions]

# Add route compare component
if os.path.exists("./Input_Data/"+name_input_data+"/Route_compare.txt"):
    route_file = open("./Input_Data/"+name_input_data+"/Route_compare.txt","r")
    route_compare_str = route_file.read()
    route_compare = gf.convert_routes_str2list(route_compare_str)
    UTNDP_problem_1.route_compare = copy.deepcopy(route_compare)
else:
    route_compare = gc.Routes.return_feasible_route_robust(UTNDP_problem_1)
    route_file = open("./Input_Data/"+name_input_data+"/Route_compare.txt","w")
    route_file.write(gf.convert_routes_list2str(route_compare))
    route_file.close()
    UTNDP_problem_1.route_compare = copy.deepcopy(route_compare)
del route_file

#if True:
def main(UTNDP_problem_1):
    
    # Reload the decisions and adjust appropriately
    Decisions = UTNDP_problem_1.Decisions # Load the decisions
    
    # Mutations reload
    mutations = {k : v for (k,v) in mutations_all.items() if k in Decisions["mutation_funcs"]}

    all_functions_dict = {"Mut_"+k : v.__name__ for (k,v) in mutations.items()}
    all_functions_dict = {'Route_gen':route_gen_func_name,**all_functions_dict, 'Crossover':crossover_func_name}
        
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
        
        # fn_obj_2 = fn_obj_ATT
        # fn_obj_2 = fn_obj_TRT
        fn_obj_2 = gf_p.fn_obj_3 # returns (f1_ATT, RD)
    
    # Add/Delete individuals to/from population
    def combine_offspring_with_pop_3(pop, offspring_variables, UTNDP_problem_input):
        """Function to combine the offspring with the population for the UTNDP routes
        NB: avoid casting lists to numpy arrays, keep it lists"""
        
        len_pop = len(pop.objectives)
        pop.variables = pop.variables + offspring_variables # adds two lists to each other
        
        # TODO: Filter out duplicates
        #is_unique = np.where(np.logical_not(find_duplicates(pop.variables, epsilon=1e-24)))[0]
        #pop.variables = pop.variables[is_unique]
        
        # Only evaluate the offspring
        offspring_variables = pop.variables[len_pop:] # this is done so that if duplicates are removed, no redundant calculations are done
        # offspring_variables_str = list(np.apply_along_axis(gf.convert_routes_list2str, 1, offspring_variables)) # potentially gave errors
        
        offspring_variables_str = [None] * len(offspring_variables)
        offspring_objectives = np.empty([len_pop, pop.objectives.shape[1]])
        
        for index_i in range(len(offspring_variables)):
            offspring_variables_str[index_i] = gf.convert_routes_list2str(offspring_variables[index_i])
            
            offspring_objectives[index_i,] = fn_obj_2(offspring_variables[index_i], UTNDP_problem_input)
    
        # Add evaluated offspring to population
        # pop.variables = np.vstack([pop.variables, offspring_variables])
        pop.variables_str = pop.variables_str + offspring_variables_str # adds two lists to each other
        pop.objectives = np.vstack([pop.objectives, offspring_objectives])  
        
        #pop_1.variables_str = np.vstack([pop_1.variables_str, offspring_variables_str])
        # This continues as normal
        pop.rank = np.empty([len(pop.variables), 1])
        pop.crowding_dist = np.empty([len(pop.variables), 1])
        
        # get the objective space values and objects
        F = pop.objectives
    
        # do the non-dominated sorting until splitting front
        fronts = gc.NonDominated_Sorting().do(F)
    
        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = gf.calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop.rank[i] = k
                pop.crowding_dist[i] = crowding_of_front[j]
    
    
    #%% GA Implementation UTNDP ############################################
    '''Load validation data'''
    if os.path.exists("./Input_Data/"+name_input_data+"/Validation_Data/Results_data_headers_all.csv"):
        validation_data = pd.read_csv("./Input_Data/"+name_input_data+"/Validation_Data/Results_data_headers_all.csv")
        if len(validation_data[validation_data["Approach"]=="John (2016)"]) != 0:
            stats_overall['HV Benchmark'] = gf_p.norm_and_calc_2d_hv(validation_data[validation_data["Approach"]=="John (2016)"].iloc[:,0:2], 
                                                                 UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        else:
            stats_overall['HV Benchmark'] = 0    
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
                                         " "+parameters_GA['method']+
                                         f"_{UTNDP_problem_1.add_text}")
    
    '''Save the parameters used in the runs'''
    if not (path_results / "Parameters").exists():
        os.makedirs(path_results / "Parameters")
    
    json.dump(UTNDP_problem_1.problem_inputs.__dict__, open(path_results / "Parameters" / "parameters_input.json", "w"), indent=4) # saves the parameters in a json file
    json.dump(UTNDP_problem_1.problem_constraints.__dict__, open(path_results / "Parameters" / "parameters_constraints.json", "w"), indent=4)
    json.dump(UTNDP_problem_1.problem_GA_parameters.__dict__, open(path_results / "Parameters" / "parameters_GA.json", "w"), indent=4)
    json.dump(UTNDP_problem_1.Decisions, open(path_results / "Parameters" / "Decisions.json", "w"), indent=4)
    if Decisions["Choice_conduct_sensitivity_analysis"]:
        json.dump(sensitivity_list, open(path_results / "Parameters" / 'Sensitivity_list.txt', 'w'), indent=4)
    
    
    '''################## Initiate the runs ##################'''
    for run_nr in range(1, parameters_GA["number_of_runs"]+1):

        if Decisions["Choice_print_results"]:           
            '''Sub folder path'''
            path_results_per_run = path_results / ("Run_"+str(run_nr))
            if not path_results_per_run.exists():
                os.makedirs(path_results_per_run)  
                
        # Create the initial population
        # TODO: Remove duplicate functions! (compare set similarity and obj function values)
        
        stats['begin_time'] = datetime.datetime.now() # enter the begin time
        stats['Termination'] ="By_user"
        print("######################### RUN {0} #########################".format(run_nr))
        print("Generation 0 initiated" + " ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
        
        # Reset the mutation ratios
        UTNDP_problem_1.mutation_ratio = [1/len(mut_functions) for _ in mut_functions]

        
        # Frequently used variables
        pop_size = UTNDP_problem_1.problem_GA_parameters.population_size

        # %% Generate intitial population
           
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
            for sol_i in range(len(pop_1.variables)):
                pop_1.objectives[sol_i] = fn_obj_2(pop_1.variables[sol_i], UTNDP_problem_1)
            
        #pop_1 = gc.PopulationRoutes(UTNDP_problem_1)  
        #pop_1.generate_or_load_initial_population(UTNDP_problem_1, fn_obj_2, route_gen_func=route_gen_funcs[route_gen_func_name], pop_choices=pop_sup_loaded)
        
        # Get non-dominated initial population
        pop_size = UTNDP_problem_1.problem_GA_parameters.population_size
        pop_1.objs_norm = ga.normalise_data_UTRP(pop_1.objectives, UTNDP_problem_1)
        survivor_indices = gf.get_survivors_norm(pop_1, pop_size)
        gf.keep_individuals(pop_1, survivor_indices)
        pop_1.population_size = pop_size
        
        # Save initial population
        ga.save_obj_pickle(pop_1, "Pop_init", path_results_per_run)
        
        # If disruption obj function employed, seed the solution
        if Decisions["Obj_func_disruption"]:
            pop_1.insert_solution_into_pop([copy.deepcopy(route_compare)], 
                                           UTNDP_problem_1, fn_obj=fn_obj_2, obj_values=False)
        

        pop_1.objs_norm = ga.normalise_data_UTRP(pop_1.objectives, UTNDP_problem_1)        
        
        # Create generational dataframe
        ld_pop_generations = ga.add_UTRP_pop_generations_data_ld(pop_1, UTNDP_problem_1, generation_num=0)
                                  

        # Create data for analysis dataframe
        ld_data_for_analysis = ga.add_UTRP_analysis_data_with_generation_nr_ld(pop_1, UTNDP_problem_1, generation_num=0)
        df_data_for_analysis = pd.DataFrame.from_dict(ld_data_for_analysis)

        # Create dataframe for mutations      
        ld_mut = [{"Mut_nr":0, "Mut_successful":0, "Mut_repaired":0, "Included_new_gen":1} for _ in range(pop_size)]
        
        #df_mut_summary = pd.DataFrame()
        ld_mut_summary = []
        df_mut_ratios = pd.DataFrame(columns=(["Generation"]+UTNDP_problem_1.mutation_names))
        df_mut_ratios.loc[0] = [0]+list(UTNDP_problem_1.mutation_ratio) #TODO: Convert to list of dictionaries
        
        # Determine non-dominated set
        df_non_dominated_set = gf.create_non_dom_set_from_dataframe_fast(df_data_for_analysis, obj_1_name='f_1', obj_2_name='f_2')
        HV = gf_p.norm_and_calc_2d_hv_np(df_non_dominated_set[["f_1","f_2"]].values, UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs) # Calculate HV
        
        if Decisions['Measure_APD']: 
            APD = gf_p.calc_avg_route_set_diversity(pop_1.variables) # average population similarity  
        else: 
            APD = 0
            
        df_data_generations = pd.DataFrame(columns = ["Generation","HV","APD"]) # create a df to keep data
        df_data_generations.loc[0] = [0, HV, APD]
        
        
        stats['end_time'] = datetime.datetime.now() # enter the begin time
        
        # Load the initial set
        df_pop_generations = pd.DataFrame.from_dict(ld_pop_generations)
        initial_set = df_pop_generations.iloc[0:UTNDP_problem_1.problem_GA_parameters.population_size,:] # load initial set
        initial_set.to_csv(path_results_per_run / "Pop_initial_set.csv")

        print("Generation {0} duration: {1} [HV:{2} | BM:{3}] APD:{4}".format(str(0),
                                                        ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']),
                                                        round(HV, 4),
                                                        round(stats_overall['HV Benchmark'],4),
                                                        round(APD, 4)))
        
        # %% Run generations
        """ ######## Run each generation ################################################################ """
        if UTNDP_problem_1.problem_GA_parameters.termination_criterion == "StoppingByNonImprovement":
            UTNDP_problem_1.problem_GA_parameters.generations = 100000
            
        for i_gen in range(1, UTNDP_problem_1.problem_GA_parameters.generations + 1):    
            # Some stats
            stats['begin_time_gen'] = datetime.datetime.now() # enter the begin time
            stats['generation'] = i_gen
            
            if i_gen % Decisions["Log_prints_every"] == 0 or i_gen == UTNDP_problem_1.problem_GA_parameters.generations:
                print("Generation " + str(int(i_gen)) + " Init: ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")",end=" ")
            
            # Crossover
            # offspring_variables = gf.crossover_pop_routes_individuals_debug(pop_1, UTNDP_problem_1)
            offspring_variables = gf.crossover_pop_routes_individuals_smart(pop_1, UTNDP_problem_1, crossover_func=crossover_funcs[crossover_func_name])
            
            # Mutation
            ld_mut_temp = gf.mutate_route_population_detailed_ld(offspring_variables, UTNDP_problem_1)
            mutated_variables = [v['Route'] for v in ld_mut_temp]
            
            # Combine offspring with population
            combine_offspring_with_pop_3(pop_1, mutated_variables, UTNDP_problem_1)
            
            # Append data for analysis
            ld_data_for_analysis = ga.add_UTRP_analysis_data_with_generation_nr_ld(pop_1, UTNDP_problem_1, i_gen, ld_data_for_analysis)
            df_data_for_analysis = pd.DataFrame.from_dict(ld_data_for_analysis)

            # Determine non-dominated set
            df_non_dominated_set = pd.concat([df_data_for_analysis[df_data_for_analysis['Generation']==i_gen],
                                             df_non_dominated_set]) # adds the new gen to non-dom set and sifts it
            df_non_dominated_set = gf.create_non_dom_set_from_dataframe_fast(df_non_dominated_set, obj_1_name='f_1', obj_2_name='f_2')
                
            # Get new generation
            pop_size = UTNDP_problem_1.problem_GA_parameters.population_size
            pop_1.objs_norm = ga.normalise_data_UTRP(pop_1.objectives, UTNDP_problem_1)
            survivor_indices = gf.get_survivors_norm(pop_1, pop_size)
            
            # Add additional details for mutations
            new_pop_details = [x - pop_size for x in survivor_indices if x >= pop_size]
            new_pop_columns = np.zeros((pop_size))
            new_pop_columns[new_pop_details] = 1
                
            for (v,k) in zip(range(len(ld_mut_temp)), new_pop_columns):
                ld_mut_temp[v]["Included_new_gen"]=k
            
            ld_mut.extend(ld_mut_temp)
            df_mut_temp = pd.DataFrame.from_dict(ld_mut_temp)
            df_mut_temp.drop(['Route'], axis='columns', inplace=True)
                        
            # Update mutation summary and overall analysis
            #df_mut_summary = df_mut_summary.append(ga.get_mutations_summary(df_mut_temp, len(UTNDP_problem_1.mutation_functions), i_gen))
            ld_mut_summary_temp = ga.get_mutations_summary_ld(df_mut_temp, len(UTNDP_problem_1.mutation_functions), i_gen)
            ld_mut_summary.extend(ld_mut_summary_temp)
            df_mut_summary = pd.DataFrame.from_dict(ld_mut_summary)

            df_mut = pd.DataFrame.from_dict(ld_mut)
            df_mut.reset_index(drop=True, inplace=True)
            
            # Update the mutation ratios
            if Decisions["Update_mutation_ratios"]:
                gf_p.update_mutation_ratio_amalgam(df_mut_summary, UTNDP_problem_1)
            df_mut_ratios.loc[i_gen] = [i_gen]+list(UTNDP_problem_1.mutation_ratio)
            
            # Remove old generation
            gf.keep_individuals(pop_1, survivor_indices)
        
            # Adds the population to the dataframe
            ld_pop_generations = ga.add_UTRP_pop_generations_data_ld(pop_1, UTNDP_problem_1, i_gen, ld_pop_generations)

            # Calculate the HV and APD Quality Measure
            HV = gf_p.norm_and_calc_2d_hv_np(df_non_dominated_set[["f_1","f_2"]].values, UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs) # Calculate HV
            if Decisions['Measure_APD']: 
                APD = gf_p.calc_avg_route_set_diversity(pop_1.variables) # average population similarity  
            else: 
                APD = 0
            df_data_generations.loc[i_gen] = [i_gen, HV, APD]
            
            # Intermediate print-outs for observance 
            if i_gen % Decisions["CSV_prints_every"] == 0 or i_gen == UTNDP_problem_1.problem_GA_parameters.generations:
                try: 
                    df_data_for_analysis = pd.DataFrame.from_dict(ld_data_for_analysis)
                    df_data_for_analysis.to_csv(path_results_per_run / "Data_for_analysis.csv")
                    df_pop_generations = pd.DataFrame.from_dict(ld_pop_generations)
                    df_pop_generations.to_csv(path_results_per_run / "Pop_generations.csv")
                    df_non_dominated_set.to_csv(path_results_per_run / "Non_dominated_set.csv")
                    df_data_generations.to_csv(path_results_per_run / "Data_generations.csv")

                except PermissionError: pass

                    
            if i_gen % Decisions["PDF_prints_every"] == 0 or i_gen == UTNDP_problem_1.problem_GA_parameters.generations:
                try: 
                    
                    df_pop_generations = pd.DataFrame.from_dict(ld_pop_generations)
                    # Compute means for generations
                    df_gen_temp = copy.deepcopy(df_data_generations)
                    df_gen_temp = df_gen_temp.assign(mean_f_1=df_pop_generations.groupby('Generation', as_index=False)['f_1'].mean().iloc[:,1],
                                       mean_f_2=df_pop_generations.groupby('Generation', as_index=False)['f_2'].mean().iloc[:,1])
                    labels = ["f_1", "f_2", "f1_ATT", "f2_TRT"] # names labels for the visualisations
                    #gv.save_results_analysis_mut_fig(initial_set, df_non_dominated_set, validation_data, 
                    #                     df_gen_temp, df_mut_ratios, name_input_data, 
                    #                     path_results_per_run, labels,
                    #                     stats_overall['HV Benchmark'], type_mut='line')
                    #gv.save_results_analysis_mut_fig(initial_set, df_non_dominated_set, validation_data, 
                    #                     df_gen_temp, df_mut_ratios, name_input_data, 
                    #                     path_results_per_run, labels,
                    #                     stats_overall['HV Benchmark'], type_mut='stacked')
                    gv.save_results_analysis_mut_fig(initial_set, df_non_dominated_set, validation_data, 
                                         df_gen_temp, df_mut_ratios, name_input_data, 
                                         path_results_per_run, labels,
                                         stats_overall['HV Benchmark'], type_mut='stacked_smooth')
                    
                    #gv.plot_generations_objectives_UTRP(df_pop_generations, every_n_gen=10, path=path_results_per_run)

                except PermissionError: pass
            
            stats['end_time_gen'] = datetime.datetime.now() # save the end time of the run
            
            if i_gen % Decisions["Log_prints_every"] == 0 or i_gen == UTNDP_problem_1.problem_GA_parameters.generations:
                print("Dur: {0} [HV:{1} | BM:{2}] APD:{3}".format(ga.print_timedelta_duration(stats['end_time_gen'] - stats['begin_time_gen']),
                                                                round(HV, 4),
                                                                round(stats_overall['HV Benchmark'],4),
                                                                round(APD, 4)))
                
            # Test whether HV is still improving
            if UTNDP_problem_1.problem_GA_parameters.termination_criterion == "StoppingByNonImprovement" or \
                UTNDP_problem_1.problem_GA_parameters.termination_criterion == "FirstToBreach":
                gen_compare = UTNDP_problem_1.problem_GA_parameters.gen_compare_HV
                HV_improvement_th = UTNDP_problem_1.problem_GA_parameters.HV_improvement_th
                if len(df_data_generations) > gen_compare:
                    HV_diff = df_data_generations['HV'].iloc[-1] - df_data_generations['HV'].iloc[-gen_compare-1]
                    if HV_diff < HV_improvement_th:
                        stats['Termination'] = 'Non-improving_HV'
                        print(f'Run terminated by non-improving HV after Gen {i_gen} HV:{HV:.4f} [Gen comp:{gen_compare} | HV diff: {HV_diff:.6f}]')
                        break
            
            if i_gen ==2 or i_gen == 100:
                # Calculate time projections for runs
                start_time_run = stats['begin_time'] # TIMING FUNCTION # stats_overall['execution_start_time']
                end_time = datetime.datetime.now() # TIMING FUNCTION
        
                # Determine and print projections
                diff = end_time - start_time_run
                diff_sec = float(str(diff.seconds)+"."+str(diff.microseconds))
                tot_iter = ga.determine_total_iterations(UTNDP_problem_1, 1)
                completed_iter =  (run_nr-1)*UTNDP_problem_1.problem_GA_parameters.generations*(pop_size) + i_gen*pop_size
                avg_time = diff_sec/(i_gen*pop_size)
                ga.time_projection_intermediate(avg_time, tot_iter, completed_iter,
                                                t_now=datetime.datetime.now(),
                                                print_iter_info=True) # prints the time projection of the algorithm


                
        #%% Stats updates
        
        if (stats['Termination'] != 'Non-improving_HV') and (__name__ == "__main__"):
            stats['Termination'] = 'By_evaluations'
        stats['end_time'] = datetime.datetime.now() # save the end time of the run
        stats['duration'] = stats['end_time'] - stats['begin_time'] # calculate and save the duration of the run
        stats['begin_time'] = stats['begin_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
        stats['end_time'] =  stats['end_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
        stats['HV obtained'] = HV
        stats['APD end'] = APD
            
        #%% Save the results #####################################################
        if Decisions["Choice_print_results"]:
            '''Write all results to files'''
            
            # Create and save the dataframe 
            df_data_for_analysis = pd.DataFrame.from_dict(ld_data_for_analysis)
            df_non_dominated_set = gf.create_non_dom_set_from_dataframe_fast(df_data_for_analysis, obj_1_name='f_1', obj_2_name='f_2')
            
            # Compute means for generations
            df_pop_generations = pd.DataFrame.from_dict(ld_pop_generations)
            df_data_generations = df_data_generations.assign(mean_f_1=df_pop_generations.groupby('Generation', as_index=False)['f_1'].mean().iloc[:,1],
                                       mean_f_2=df_pop_generations.groupby('Generation', as_index=False)['f_2'].mean().iloc[:,1])
            
            try:
                """Print-outs for observations"""
                df_pop_generations.to_csv(path_results_per_run / "Pop_generations.csv")
                df_non_dominated_set.to_csv(path_results_per_run / "Non_dominated_set.csv")
                df_data_for_analysis.to_csv(path_results_per_run / "Data_for_analysis.csv")
                df_data_generations.to_csv(path_results_per_run / "Data_generations.csv")
                df_mut_ratios.to_csv(path_results_per_run / "Mut_ratios.csv")
                df_mut_summary.to_csv(path_results_per_run / "Mut_summary.csv")
                
                # Print and save result summary figures:
                labels = ["f_1", "f_2", "f1_ATT", "f2_TRT"] # names labels for the visualisations
                #gv.save_results_analysis_mut_fig(initial_set, df_non_dominated_set, validation_data, 
                #                             df_data_generations, df_mut_ratios, name_input_data, 
                #                             path_results_per_run, labels,
                #                             stats_overall['HV Benchmark'], type_mut='line')
                #gv.save_results_analysis_mut_fig(initial_set, df_non_dominated_set, validation_data, 
                #                             df_data_generations, df_mut_ratios, name_input_data, 
                #                             path_results_per_run, labels,
                #                             stats_overall['HV Benchmark'], type_mut='stacked')
                gv.save_results_analysis_mut_fig(initial_set, df_non_dominated_set, validation_data, 
                                             df_data_generations, df_mut_ratios, name_input_data, 
                                             path_results_per_run, labels,
                                             stats_overall['HV Benchmark'], type_mut='stacked_smooth')
            except PermissionError: pass
            
            #%% Post analysis 
            try:
                pickle.dump(stats, open(path_results_per_run / "stats.pickle", "ab"))
                
                # Writes all the stats in a csv file
                with open(path_results_per_run / "Run_summary_stats.csv", "w") as archive_file:
                    w = csv.writer(archive_file)
                    for key, val in {**parameters_input, **parameters_constraints, **parameters_GA, **stats}.items():
                        w.writerow([key, val])
                    del key, val
                
                # Writes all the functions used in a csv file
                with open(path_results / "Functions_used.csv", "w") as archive_file:
                    w = csv.writer(archive_file)
                    for key, val in {**all_functions_dict}.items():
                        w.writerow([key, val])
                    del key, val  
            
            except PermissionError: pass
            
            print("End of generations: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            
            # Visualise the generations
            #try:
            #    gv.plot_generations_objectives_UTRP(df_pop_generations, every_n_gen=10, path=path_results_per_run)
            #except PermissionError: pass
            
    #del i_gen, mutated_variables, offspring_variables, pop_size, survivor_indices
    
        # %% Save results after all runs
        if Decisions["Choice_print_results"]:
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
            
            stats_overall['execution_end_time'] =  datetime.datetime.now()
            
            stats_overall['total_model_runs'] = run_nr
            stats_overall['average_run_time'] = str(df_durations["Duration"].mean())
            stats_overall['total_duration'] = stats_overall['execution_end_time']-stats_overall['execution_start_time']
            stats_overall['start_time_formatted'] = stats_overall['execution_start_time'].strftime("%m/%d/%Y, %H:%M:%S")
            stats_overall['end_time_formatted'] = stats_overall['execution_end_time'].strftime("%m/%d/%Y, %H:%M:%S")
            stats_overall['HV obtained'] = gf_p.norm_and_calc_2d_hv(df_overall_pareto_set[["f_1","f_2"]], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
            
            df_durations.loc[len(df_durations)] = ["Average", df_durations["Duration"].mean(), df_durations["HV Obtained"].mean()]
            df_durations.to_csv(path_results / "Run_durations.csv")
            del df_durations
            
            try:
                # Writes all the stats in a csv file
                with open(path_results / "Stats_overall.csv", "w") as archive_file:
                    w = csv.writer(archive_file)
                    for key, val in {**stats_overall,
                                     **UTNDP_problem_1.problem_inputs.__dict__, 
                                     **UTNDP_problem_1.problem_constraints.__dict__, 
                                     **UTNDP_problem_1.problem_GA_parameters.__dict__}.items():
                        w.writerow([key, val])
                    del key, val
                
                # Writes all the functions used in a csv file
                with open(path_results / "Functions_used.csv", "w") as archive_file:
                    w = csv.writer(archive_file)
                    for key, val in {**all_functions_dict}.items():
                        w.writerow([key, val])
                    del key, val
            
            except PermissionError: pass
            
            for _ in range(5):
                try:
                    ga.get_sens_tests_stats_from_model_runs(path_results, run_nr) # prints the runs summary
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

# %% Sensitivity analysis
''' Sensitivity analysis tests'''
#if False:
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
            if sensitivity_list[parameter_index][0] == "parameters_GA": 
                sensitivity_list[parameter_index][0] = parameters_GA
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
                    
                    if (sensitivity_test[1] == "crossover_func") or  (sensitivity_test[1] == "mutation_funcs"):
                        UTNDP_problem_1.add_text = f"{sensitivity_test[1]}_{test_counter-2}"
                    else:
                        UTNDP_problem_1.add_text = f"{sensitivity_test[1]}_{round(sensitivity_test[test_counter],4)}"
                    
                    temp_storage = parameter_dict[dict_entry]
                    
                    # Set new parameters
                    parameter_dict[dict_entry] = sensitivity_test[test_counter]
        
                    # Update problem instance
                    UTNDP_problem_1.problem_constraints = gc.Problem_inputs(parameters_constraints)
                    UTNDP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA)
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
