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
                   "Mandl_UTRP_dis"][4]   # set the name of the input data

# %% Set input parameters
sens_from = 0
sens_to = (sens_from + 1) if False else -1
dis_obj = False
load_sup = False #TODO Remove later

if False:
    Decisions = json.load(open("./Input_Data/"+name_input_data+"/Decisions.json"))

else:
    Decisions = {
    "Choice_print_results" : True, 
    "Choice_conduct_sensitivity_analysis" : False,
    "Choice_import_dictionaries" : True,
    "Choice_print_full_data_for_analysis" : True,
    "Choice_relative_results_referencing" : False,
    "Additional_text" : "Tests",
    "Pop_size_to_create" : 2000,
    } 
    
#%% Set functions to use
    route_gen_funcs = {"KSP_unseen_robust" : gc.Routes.return_feasible_route_robust_k_shortest,
                       "KSP_unseen_robust_prob_10000" : gc.Routes.return_feasible_route_robust_k_shortest_probabilistic,
                       "KSP_unseen_robust_prob" : gc.Routes.return_feasible_route_robust_k_shortest_probabilistic,
                        "Greedy_demand" : gc.Routes.return_feasible_route_set_greedy_demand,
                        "Unseen_robust" : gc.Routes.return_feasible_route_robust}
    route_gen_func_name = list(route_gen_funcs.keys())[2]
    
    crossover_funcs = {"Mumford" : gf.crossover_mumford,
                       "Unseen_probabilistic" : gf.crossover_unseen_probabilistic,
                       "Mumford_replace_subsets_ksp" : gf.crossover_mumford_rem_subsets_ksp,
                       "Unseen_probabilistic_replace_subsets_ksp" : gf.crossover_unseen_probabilistic_rem_subsets_ksp,
                       "Mumford_replace_subsets" : gf.crossover_mumford_rem_subsets,
                       "Unseen_probabilistic_replace_subsets" : gf.crossover_unseen_probabilistic_rem_subsets}
    crossover_func_name = list(crossover_funcs.keys())[5]
    
    mutations = {#"No_mutation" : gf.no_mutation,
                    "Intertwine_two" : gf.mutate_routes_two_intertwine, 
                    "Add_vertex" : gf.add_vertex_to_terminal,
                    "Delete_vertex" : gf.remove_vertex_from_terminal,
                    #"Merge_terminals" : gf.mutate_merge_routes_at_common_terminal, 
                    #"Repl_low_dem_route" : gf.mut_replace_lowest_demand,
                    #"Rem_low_dem_terminal" : gf.mut_remove_lowest_demand_terminal,
                    #"Rem_lrg_cost_terminal" : gf.mut_remove_largest_cost_terminal,
                    #"Repl_high_sim_route":gf.mut_replace_high_sim_routes, # bad mutation
                    #"Repl_subsets" : gf.mut_replace_path_subsets,
                    #"Invert_path_vertices" : gf.mut_invert_route_vertices,
                    
                    #"Rem_largest_cost_per_dem" : gf.mut_remove_largest_cost_per_dem_terminal,
                    #"Trim_one_path_random_cb" : gf.mut_trim_one_path_random_cb,
                    #"Trim_routes_random_cb" : gf.mut_trim_routes_random_cb,
                    #"Trim_all_paths_random_cb" : gf.mut_trim_all_paths_random_cb,
                    #"Trim_full_overall_cb" : gf.mut_trim_full_overall_cb,
                    
                    #"Add_largest_dem_per_cost" : gf.mut_add_terminal_highest_demand_per_cost,
                    #"Grow_one_path_random_cb" : gf.mut_grow_one_path_random_cb,
                    #"Grow_routes_random_cb" : gf.mut_grow_routes_random_cb,
                    #"Grow_all_paths_random_cb" : gf.mut_grow_all_paths_random_cb,
                    #"Grow_full_overall_cb" : gf.mut_grow_full_overall_cb,
                    }
    
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
    #parameters_GA = json.load(open("./Input_Data/"+name_input_data+"/parameters_GA.json"))
    
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
   
    '''State the various GA input parameters for frequency setting''' 
    parameters_GA={
    "method" : "GA",
    "population_size" : 400, #should be an even number STANDARD: 200 (John 2016)
    "generations" : 1000, # STANDARD: 200 (John 2016)
    "number_of_runs" : 1, # STANDARD: 20 (John 2016)
    "crossover_probability" : 0.6, 
    "crossover_distribution_index" : 5,
    "mutation_probability" : 1, # John: 1/|Route set| -> set later
    "mutation_distribution_index" : 10,
    "mutation_ratio" : 0.5, # Ratio used for the probabilites of mutations applied
    "mutation_threshold" : 0.05, # Minimum threshold that mutation probabilities can reach
    "tournament_size" : 2,
    "termination_criterion" : "StoppingByEvaluations",
    "max_evaluations" : 25000,
    "gen_compare_HV" : 20, # Compare generations for improvement in HV
    "HV_improvement_th": 0.00005, # Treshold that terminates the search
     "number_of_variables" : parameters_constraints["con_r"],
    "number_of_objectives" : 2, # this could still be automated in the future
    "Number_of_initial_solutions" : 10000 # number of initial solutions to be generated and chosen from
    }
 
else:    
    '''State the various parameter constraints''' 
    parameters_constraints = {
    'con_r' : 6,               # number of allowed routes (aim for > [numNodes N ]/[maxNodes in route])
    'con_minNodes' : 2,                        # minimum nodes in a route
    'con_maxNodes' : 10,                       # maximum nodes in a route
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
    "generations" : 400, # STANDARD: 200 (John 2016)
    "number_of_runs" : 20, # STANDARD: 20 (John 2016)
    "crossover_probability" : 0.6, 
    "crossover_distribution_index" : 5,
    "mutation_probability" : 0.95, # John: 1/|Route set| -> set later
    "mutation_distribution_index" : 10,
    "mutation_ratio" : 0.1, # Ratio used for the probabilites of mutations applied
    "tournament_size" : 2,
    "termination_criterion" : "StoppingByEvaluations",
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
    
    #sensitivity_list = [
                        #["population_size", 10, 20, 50, 100, 150, 200, 300], 
                        #["generations", 10, 20, 50, 100, 150, 200, 300],
                        #["crossover_probability", 0.5, 0.6, 0.7, 0.8],
                        #["crossover_probability", 0.9, 0.95, 1],
                        #["mutation_probability", 0.05, 0.1, 1/parameters_constraints["con_r"], 0.2, 0.3],                   
                        #["mutation_probability", 0.5, 0.7, 0.9, 1],
                        #["mutation_ratio", 0.1, 0.2]
                        #]
    
    #sensitivity_list = [
                        #["population_size", 20, 400], 
                        #["generations", 20, 400],
                        #["crossover_probability", 0.1],
                        #["crossover_probability", 1],
                        #["mutation_probability", 0.05], 
                        #["mutation_probability", 0.9],
                        #["mutation_probability", 1],  
                        
                        # ["mutation_ratio", 0.01],
                        # ["mutation_ratio", 0.05],
                        # ["mutation_ratio", 0.1],
                        # ["mutation_ratio", 0.2],
                        # ["mutation_ratio", 0.3],
                        # ["mutation_ratio", 0.4],
                        # ["mutation_ratio", 0.5],
                        # ["mutation_ratio", 0.6],
                        # ["mutation_ratio", 0.7],
                        # ["mutation_ratio", 0.8],
                        # ["mutation_ratio", 0.9],
                        # ["mutation_ratio", 0.95],
                        # ]
    
    sensitivity_list = sensitivity_list[sens_from:sens_to] # truncates the sensitivity list


'''Set the reference point for the Hypervolume calculations'''
max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])

            
#%% Input parameter tests

'''Test the inputs for feasibility'''
# Test feasibility if there are enough buses to cover each route once
if parameters_constraints["con_r"] > parameters_constraints["con_fleet_size"]:
    print("Warning: Number of available vehicles are less than the number of routes.\n"\
          "Number of routes allowed set to "+ str(parameters_constraints["con_r"]))
    parameters_constraints["con_r"] = parameters_constraints["con_fleet_size"]


#%% Define the UTNDP Problem      
UTNDP_problem_1 = gc.UTNDP_problem()
UTNDP_problem_1.problem_data = gc.Problem_data(mx_dist, mx_demand, mx_coords)
UTNDP_problem_1.problem_constraints = gc.Problem_constraints(parameters_constraints)
UTNDP_problem_1.problem_inputs = gc.Problem_inputs(parameters_input)
UTNDP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA)
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
    route_compare = route_file.read()
    UTNDP_problem_1.route_compare = gf.convert_routes_str2list(route_compare)
else:
    route_compare = gc.Routes.return_feasible_route_robust(UTNDP_problem_1)
    route_file = open("./Input_Data/"+name_input_data+"/Route_compare.txt","w")
    route_file.write(gf.convert_routes_list2str(route_compare))
    route_file.close()
    UTNDP_problem_1.route_compare = route_compare
del route_file, route_compare

if True:
#def main(UTNDP_problem_1):
    
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
    
    if dis_obj:
        fn_obj_2 = gf.fn_obj_3 # returns (ATT, RD)
    
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
                                         " "+parameters_GA['method']+
                                         f"_{UTNDP_problem_1.add_text}")
    
    '''Save the parameters used in the runs'''
    if not (path_results / "Parameters").exists():
        os.makedirs(path_results / "Parameters")
    
    json.dump(parameters_input, open(path_results / "Parameters" / "parameters_input.json", "w")) # saves the parameters in a json file
    json.dump(parameters_constraints, open(path_results / "Parameters" / "parameters_constraints.json", "w"))
    json.dump(parameters_GA, open(path_results / "Parameters" / "parameters_GA.json", "w"))
    json.dump(Decisions, open(path_results / "Parameters" / "Decisions.json", "w"))
    if Decisions["Choice_conduct_sensitivity_analysis"]:
        json.dump(sensitivity_list, open(path_results / "Parameters" / 'Sensitivity_list.txt', 'w'))
    
    
    '''################## Initiate the runs ##################'''
    for run_nr in range(1, parameters_GA["number_of_runs"]+1):

        if Decisions["Choice_print_results"]:           
            '''Sub folder path'''
            path_results_per_run = path_results / ("Run_"+str(run_nr))
            if not path_results_per_run.exists():
                os.makedirs(path_results_per_run)  
                
        # Create the initial population
        # TODO: Insert initial 10000 solutions generatations and NonDom Sort your initial population, ensuring diversity
        # TODO: Remove duplicate functions! (compare set similarity and obj function values)
        
        stats['begin_time'] = datetime.datetime.now() # enter the begin time
        print("######################### RUN {0} #########################".format(run_nr))
        print("Generation 0 initiated" + " ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
        
        # Reset the mutation ratios
        UTNDP_problem_1.mutation_ratio = [1/len(mut_functions) for _ in mut_functions]

        
        # Frequently used variables
        pop_size = UTNDP_problem_1.problem_GA_parameters.population_size

        # %% Generate intitial population
                           
        # Load and save initial population
        directory = Path(path_parent_folder / ("DSS Main/Input_Data/"+name_input_data+"/Populations"))
        pop_loaded = gf_p.load_UTRP_pop_or_create("Pop_init_"+route_gen_func_name+"_"+str(Decisions["Pop_size_to_create"]), directory, UTNDP_problem_1, route_gen_funcs[route_gen_func_name], fn_obj_2, pop_size_to_create=Decisions["Pop_size_to_create"])
        
        if load_sup:
            pop_sup_loaded = gf_p.load_UTRP_supplemented_pop_or_create("Pop_sup_"+route_gen_func_name+"_"+str(Decisions["Pop_size_to_create"]), directory, UTNDP_problem_1,route_gen_funcs[route_gen_func_name], fn_obj_2, pop_loaded)
            pop_1 = pop_sup_loaded
        
        else:
            pop_1 = pop_loaded
            
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
        if dis_obj:
            pop_1.insert_solution_into_pop([UTNDP_problem_1.route_compare], 
                                           UTNDP_problem_1, fn_obj=fn_obj_2, obj_values=False)
        

        pop_1.objs_norm = ga.normalise_data_UTRP(pop_1.objectives, UTNDP_problem_1)        
        
        # Create generational dataframe
        ld_pop_generations = ga.add_UTRP_pop_generations_data_ld(pop_1, UTNDP_problem_1, generation_num=0)
                                  

        # Create data for analysis dataframe
        ld_data_for_analysis = ga.add_UTRP_analysis_data_with_generation_nr_ld(pop_1, UTNDP_problem_1, generation_num=0)
        df_data_for_analysis = pd.DataFrame.from_dict(ld_data_for_analysis)
        df_overall_analysis = pd.DataFrame()

        # Create dataframe for mutations      
        ld_mut = [{"Mut_nr":0, "Mut_successful":0, "Mut_repaired":0, "Included_new_gen":1} for _ in range(pop_size)]
        
        df_mut_summary = pd.DataFrame()
        df_mut_ratios = pd.DataFrame(columns=(["Generation"]+UTNDP_problem_1.mutation_names))
        df_mut_ratios.loc[0] = [0]+list(UTNDP_problem_1.mutation_ratio)
        
        # Determine non-dominated set
        df_non_dominated_set = gf.create_non_dom_set_from_dataframe(df_data_for_analysis, obj_1_name='f_1', obj_2_name='f_2')
        HV = gf_p.norm_and_calc_2d_hv_np(df_non_dominated_set[["f_1","f_2"]].values, UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs) # Calculate HV
        APD = gf.calc_avg_route_set_diversity(pop_1.variables) # average population similarity
        
        df_data_generations = pd.DataFrame(columns = ["Generation","HV","APD"]) # create a df to keep data
        df_data_generations.loc[0] = [0, HV, APD]
        
        stats['end_time'] = datetime.datetime.now() # enter the begin time
        
        # Load the initial set
        df_pop_generations = pd.DataFrame.from_dict(ld_pop_generations)
        initial_set = df_pop_generations.iloc[0:UTNDP_problem_1.problem_GA_parameters.population_size,:] # load initial set

        print("Generation {0} duration: {1} [HV:{2} | BM:{3}] APD:{4}".format(str(0),
                                                        ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']),
                                                        round(HV, 4),
                                                        round(stats_overall['HV Benchmark'],4),
                                                        round(APD, 4)))
        
        # %% Run generations
        """ ######## Run each generation ################################################################ """
        for i_gen in range(1, UTNDP_problem_1.problem_GA_parameters.generations + 1):    
            # Some stats
            stats['begin_time_gen'] = datetime.datetime.now() # enter the begin time
            stats['generation'] = i_gen
            
            if i_gen % 20 == 0 or i_gen == UTNDP_problem_1.problem_GA_parameters.generations:
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
            df_non_dominated_set = gf.create_non_dom_set_from_dataframe(df_data_for_analysis, obj_1_name='f_1', obj_2_name='f_2')
                
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
            df_mut_summary = df_mut_summary.append(ga.get_mutations_summary(df_mut_temp, len(UTNDP_problem_1.mutation_functions), i_gen))
            df_mut = pd.DataFrame.from_dict(ld_mut)
            df_mut.reset_index(drop=True, inplace=True)
            
            # Update the mutation ratios
            def update_mutation_ratio_amalgam(df_mut_summary, UTNDP_problem_1):
                nr_of_mutations = len(UTNDP_problem_1.mutation_functions)
                mutation_threshold = UTNDP_problem_1.problem_GA_parameters.mutation_threshold
                success_ratio = df_mut_summary["Inc_over_Tot"].iloc[-nr_of_mutations:].values
                
                # reset the success ratios if all have falied
                if sum(success_ratio) == 0:
                    success_ratio = np.array([1/len(success_ratio) for _ in success_ratio])
                
                success_proportion = (success_ratio / sum(success_ratio))*(1-nr_of_mutations*mutation_threshold)      
                updated_ratio = mutation_threshold + success_proportion
                UTNDP_problem_1.mutation_ratio = updated_ratio
                
            update_mutation_ratio_amalgam(df_mut_summary, UTNDP_problem_1)
            df_mut_ratios.loc[i_gen] = [i_gen]+list(UTNDP_problem_1.mutation_ratio)
            
            # Remove old generation
            gf.keep_individuals(pop_1, survivor_indices)
        
            # Adds the population to the dataframe
            ld_pop_generations = ga.add_UTRP_pop_generations_data_ld(pop_1, UTNDP_problem_1, i_gen, ld_pop_generations)

            # Calculate the HV and APD Quality Measure
            HV = gf_p.norm_and_calc_2d_hv_np(df_non_dominated_set[["f_1","f_2"]].values, UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs) # Calculate HV
            APD = gf.calc_avg_route_set_diversity(pop_1.variables) # average population similarity
            df_data_generations.loc[i_gen] = [i_gen, HV, APD]
            
            # Intermediate print-outs for observance 
            if i_gen % 20 == 0 or i_gen == UTNDP_problem_1.problem_GA_parameters.generations:
                try: 
                    df_data_for_analysis = pd.DataFrame.from_dict(ld_data_for_analysis)
                    df_data_for_analysis.to_csv(path_results_per_run / "Data_for_analysis.csv")
                    df_pop_generations = pd.DataFrame.from_dict(ld_pop_generations)
                    df_pop_generations.to_csv(path_results_per_run / "Pop_generations.csv")
                    df_non_dominated_set.to_csv(path_results_per_run / "Non_dominated_set.csv")
                    df_data_generations.to_csv(path_results_per_run / "Data_generations.csv")

                    # gv.save_results_analysis_fig_interim_UTRP(initial_set, df_non_dominated_set, 
                    #                                           validation_data, df_data_generations, 
                    #                                           name_input_data, path_results_per_run,
                    #                                           stats_overall['HV Benchmark']) 
                    
                    # Compute means for generations
                    df_gen_temp = copy.deepcopy(df_data_generations)
                    df_gen_temp = df_gen_temp.assign(mean_f_1=df_pop_generations.groupby('Generation', as_index=False)['f_1'].mean().iloc[:,1],
                                       mean_f_2=df_pop_generations.groupby('Generation', as_index=False)['f_2'].mean().iloc[:,1])
                    labels = ["f_1", "f_2", "f1_ATT", "f2_TRT"] # names labels for the visualisations
                    gv.save_results_analysis_mut_fig(initial_set, df_non_dominated_set, validation_data, 
                                         df_gen_temp, df_mut_ratios, name_input_data, 
                                         path_results_per_run, labels,
                                         stats_overall['HV Benchmark'])
                    
                    gv.plot_generations_objectives_UTRP(df_pop_generations, every_n_gen=10, path=path_results_per_run)

                    #gv.save_results_analysis_fig_interim_save_all(initial_set, df_non_dominated_set, validation_data, 
                    #                                              df_data_generations, name_input_data, path_results_per_run, add_text=i_gen, labels)
                except PermissionError: pass
            
            stats['end_time_gen'] = datetime.datetime.now() # save the end time of the run
            
            if i_gen % 20 == 0 or i_gen == UTNDP_problem_1.problem_GA_parameters.generations:
                print("Dur: {0} [HV:{1} | BM:{2}] APD:{3}".format(ga.print_timedelta_duration(stats['end_time_gen'] - stats['begin_time_gen']),
                                                                round(HV, 4),
                                                                round(stats_overall['HV Benchmark'],4),
                                                                round(APD, 4)))
                
            # Test whether HV is still improving
            gen_compare = UTNDP_problem_1.problem_GA_parameters.gen_compare_HV
            HV_improvement_th = UTNDP_problem_1.problem_GA_parameters.HV_improvement_th
            if len(df_data_generations) > gen_compare:
                HV_diff = df_data_generations['HV'].iloc[-1] - df_data_generations['HV'].iloc[-gen_compare-1]
                if HV_diff < HV_improvement_th:
                    stats['Termination'] = 'Non-improving_HV'
                    print(f'Run terminated by non-improving HV after Gen {i_gen} HV:{HV:.4f} [Gen comp:{gen_compare} | HV diff: {HV_diff:.6f}]')
                    break
            
                
        #%% Stats updates
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
            df_non_dominated_set = gf.create_non_dom_set_from_dataframe(df_data_for_analysis, obj_1_name='f_1', obj_2_name='f_2')
            
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
                gv.save_results_analysis_mut_fig(initial_set, df_non_dominated_set, validation_data, 
                                             df_data_generations, df_mut_ratios, name_input_data, 
                                             path_results_per_run, labels,
                                             stats_overall['HV Benchmark'])
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
            try:
                gv.plot_generations_objectives_UTRP(df_pop_generations, every_n_gen=10, path=path_results_per_run)
            except PermissionError: pass
            
    #del i_gen, mutated_variables, offspring_variables, pop_size, survivor_indices
    
        # %% Save results after all runs
        if Decisions["Choice_print_results"]:
            '''Save the summarised results'''
            df_overall_pareto_set = ga.group_pareto_fronts_from_model_runs_2(path_results, parameters_input, "Non_dominated_set.csv").iloc[:,1:]
            df_overall_pareto_set = df_overall_pareto_set[gf.is_pareto_efficient(df_overall_pareto_set[["f_1","f_2"]].values, True)] # reduce the pareto front from the total archive
            df_overall_pareto_set = df_overall_pareto_set.sort_values(by='f_1', ascending=True) # sort
            try:
                df_overall_pareto_set.to_csv(path_results / "Overall_Pareto_set.csv")   # save the csv file
            except PermissionError: pass
            
            '''Save the stats for all the runs'''
            # df_routes_R_initial_set.to_csv(path_results / "Routes_initial_set.csv")
            df_durations = ga.get_stats_from_model_runs(path_results)
            
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
            
            try:
                ga.get_sens_tests_stats_from_model_runs(path_results, run_nr) # prints the runs summary
                gv.save_all_mutation_stats_and_plots(path_results) # gets and prints the mutation stats
                gv.save_all_obj_stats_and_plots(path_results) # gets and prints the objective performance stats
                gv.save_final_avgd_results_analysis(initial_set, df_overall_pareto_set, validation_data, 
                                                  pd.read_csv((path_results/'Performance/Avg_obj_performances.csv')), 
                                                  pd.read_csv((path_results/'Mutations/Avg_mut_ratios.csv')), # can use 'Smoothed_avg_mut_ratios.csv' for a double smooth visualisation
                                                  name_input_data, 
                                                  path_results, labels,
                                                  stats_overall['HV Benchmark'])
            
                gf.print_extreme_solutions(df_overall_pareto_set, HV, stats_overall['HV Benchmark'], name_input_data, UTNDP_problem_1, path_results)
                # ga.get_sens_tests_stats_from_UTRP_GA_runs(path_results) 
            
            except PermissionError: pass
    
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
 
        sensitivity_list = sensitivity_list[sens_from:sens_to]   
 
        for parameter_index in range(len(sensitivity_list)):
            sensitivity_list[parameter_index].insert(0, parameters_GA)
        
        for sensitivity_test in sensitivity_list:
            parameter_dict = sensitivity_test[0]
            dict_entry = sensitivity_test[1]
            for test_counter in range(2,len(sensitivity_test)):
                
                print("Test: {0} = {1}".format(sensitivity_test[1], sensitivity_test[test_counter]))
                
                UTNDP_problem_1.add_text = f"{sensitivity_test[1]}_{round(sensitivity_test[test_counter],2)}"
                
                temp_storage = parameter_dict[dict_entry]
                
                # Set new parameters
                parameter_dict[dict_entry] = sensitivity_test[test_counter]
    
                # Update problem instance
                UTNDP_problem_1.problem_constraints = gc.Problem_inputs(parameters_constraints)
                UTNDP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA)
                
                # Run model
                #main(UTNDP_problem_1)
                
                # Reset the original parameters
                parameter_dict[dict_entry] = temp_storage
        
        finish = time.perf_counter()
        
        print(f'Finished in {round(finish-start, 6)} second(s)')
        
    else:
        pass
        #main(UTNDP_problem_1) 
