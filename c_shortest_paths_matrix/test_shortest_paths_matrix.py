"""
To run these tests, assuming you have py.test installed, just run:
    py.test -vs

('-v' is verbose and '-s' switch does not suppress prints)
"""

import time
from numpy import array, inf, asarray, diagonal, minimum, newaxis, fill_diagonal
#from numpy.random import random

import pyximport; #pyximport.install()
pyximport.install(reload_support=True)
#import shortest_paths_matrix

#%% Import all the packages as usual to test with
import os
main_dir = os.path.abspath(os.curdir)
print (os.path.abspath(os.curdir))
os.chdir("..")
print (os.path.abspath(os.curdir))
###############################################################################################################
# Special imports
from floyd_warshall_cython_master import floyd_warshall



if True:
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
    import DSS_UTNDP_Functions as gf
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
                       "Mandl_UTRP_dis"][2]   # set the name of the input data
    
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
        "Additional_text" : "Tests"
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
                        "Repl_subsets" : gf.mut_replace_path_subsets,
                        #"Invert_path_vertices" : gf.mut_invert_route_vertices,
                        
                        "Rem_largest_cost_per_dem" : gf.mut_remove_largest_cost_per_dem_terminal,
                        "Trim_one_path_random_cb" : gf.mut_trim_one_path_random_cb,
                        #"Trim_routes_random_cb" : gf.mut_trim_routes_random_cb,
                        #"Trim_all_paths_random_cb" : gf.mut_trim_all_paths_random_cb,
                        #"Trim_full_overall_cb" : gf.mut_trim_full_overall_cb,
                        
                        "Add_largest_dem_per_cost" : gf.mut_add_terminal_highest_demand_per_cost,
                        "Grow_one_path_random_cb" : gf.mut_grow_one_path_random_cb,
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
        "population_size" : 100, #should be an even number STANDARD: 200 (John 2016)
        "generations" : 10, # STANDARD: 200 (John 2016)
        "number_of_runs" : 2, # STANDARD: 20 (John 2016)
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


    #%% Define the Objective UTNDP functions
    def fn_obj_2(routes, UTNDP_problem_input):
        return (ev.evalObjs(routes, 
                UTNDP_problem_input.problem_data.mx_dist, 
                UTNDP_problem_input.problem_data.mx_demand, 
                UTNDP_problem_input.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)

    #%% GA Implementation UTNDP ############################################
    '''Load validation data'''
    if os.path.exists("./Input_Data/"+name_input_data+"/Validation_Data/Results_data_headers_all.csv"):
        validation_data = pd.read_csv("./Input_Data/"+name_input_data+"/Validation_Data/Results_data_headers_all.csv")
    else:
        validation_data = False

    '''Main folder path'''
    if Decisions["Choice_relative_results_referencing"]:
        path_parent_folder = Path(os.path.dirname(os.getcwd()))
    else:
        path_parent_folder = Path("C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS")
        if not os.path.isdir(path_parent_folder):
            path_parent_folder = Path("C:/Users/gunth/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS")
        if not os.path.isdir(path_parent_folder):
            path_parent_folder = Path(os.path.dirname(os.getcwd()))
    

    # Load and save initial population
    directory = Path(path_parent_folder / ("DSS Main/Input_Data/"+name_input_data+"/Populations"))
    pop_loaded = gf.load_UTRP_pop_or_create("Pop_init_"+route_gen_func_name, directory, UTNDP_problem_1, route_gen_funcs[route_gen_func_name], fn_obj_2, pop_size_to_create=10000)
    
    if load_sup:
        pop_sup_loaded = gf.load_UTRP_supplemented_pop_or_create("Pop_sup_"+route_gen_func_name, directory, UTNDP_problem_1, route_gen_funcs[route_gen_func_name], fn_obj_2, pop_loaded)
        pop_1 = pop_sup_loaded
    
    else:
        pop_1 = pop_loaded


###############################################################################################################
os.chdir(main_dir)
print (os.path.abspath(os.curdir))

# %% Function experiments

if True:
    routeset=pop_1.variables[2]
    travelTimes=mx_dist
    DemandMat=mx_demand

#def evalObjs(routeset=R_x,travelTimes=mx_dist,DemandMat=mx_demand,parameters_input=parameters_input):
    total_demand = parameters_input['total_demand']
    n = parameters_input['n'] # number of nodes
    wt = parameters_input['wt'] # waiting time
    tp = parameters_input['tp'] # transfer penalty
    
    RL = ev.evaluateTotalRouteLength(routeset,travelTimes)
    routeadj,inv_map,t,shortest,longest = ev.expandTravelMatrix(routeset, travelTimes,n,tp,wt)

    # Numpy floyd_warshall:
    #D = floyd_warshall_fastest(routeadj,t)
    
    # Cython floyd_warshall:
    np.fill_diagonal(routeadj, 0)
    D = floyd_warshall.floyd_warshall_single_core(routeadj)    

    SPMatrix = ev.shortest_paths_matrix(D, inv_map, t, n)
    ATT = ev.EvaluateATT(SPMatrix, DemandMat, total_demand, wt)
#    return ATT, RL

#evalObjs(routeset=offspring_variables[0],travelTimes=mx_dist,DemandMat=mx_demand,parameters_input=parameters_input)


def shortest_paths_matrix_naive(D, inv_map, t, n):

    SPMatrix = np.inf*np.ones((n,n), dtype=float)
    #count = 0
    for i in range(t):
        p1 = inv_map[i]
        for j in range(t):
            p2 = inv_map[j]
            if (D[i][j]<SPMatrix[p1][p2]):
                SPMatrix[p1][p2] = D[i][j]
                #count = count + 1
    return(SPMatrix)

#shortest_paths_matrix(D, inv_map, t, n)

if False:
#    %timeit floyd_warshall.floyd_warshall_single_core(routeadj)
#    %timeit ev.floyd_warshall_fastest(routeadj,len(routeadj))

    np.fill_diagonal(routeadj, 0)
    c_M = floyd_warshall.floyd_warshall_single_core(routeadj)
    p_M = ev.floyd_warshall_fastest(routeadj,len(routeadj))
    assert (c_M == p_M).all() 

if True:
    from shortest_paths_matrix_master import shortest_paths_matrix
    
    p_SPM = shortest_paths_matrix_naive(D, inv_map, t, n)
    c_SPM = shortest_paths_matrix(D, inv_map, t, n)
    assert (c_SPM == p_SPM).all() 


# %% Defs
def check_and_convert_adjacency_matrix(adjacency_matrix):
    mat = asarray(adjacency_matrix)

    (nrows, ncols) = mat.shape
    assert nrows == ncols
    n = nrows

    assert (diagonal(mat) == 0.0).all()

    return (mat, n)

# def test_floyd_warshall_algorithms_on_small_matrix():
#     INPUT = array([
#         [  0.,  inf,  -2.,  inf],
#         [  4.,   0.,   3.,  inf],
#         [ inf,  inf,   0.,   2.],
#         [ inf,  -1.,  inf,   0.]
#     ])

#     OUTPUT = array([
#         [ 0., -1., -2.,  0.],
#         [ 4.,  0.,  2.,  4.],
#         [ 5.,  1.,  0.,  2.],
#         [ 3., -1.,  1.,  0.]])

#     assert (floyd_warshall_naive(INPUT) == OUTPUT).all()
#     assert (floyd_warshall_numpy(INPUT) == OUTPUT).all()

class Timer(object):
    def __init__(self, text = None):
        self.start_clock = time.process_time() # time.process_time() #alternative
        self.start_time = time.time()
        if text != None:
            print ('%s:' % text), 
    def stop(self):
        wall_time = time.time() - self.start_time
        cpu_time = time.process_time() - self.start_clock
        print(f'Wall time: {wall_time:.3f} seconds.  CPU time: {cpu_time:.3f} seconds.')
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.stop()

def test_funcs_speed():
    M = np.random.random((100, 100))
    np.fill_diagonal(M, 0)

    prev_res = None
    print('')
    for (name, func) in [('Python+NumPy', shortest_paths_matrix_naive), ('Cython', shortest_paths_matrix)]:
        print ('%20s: ' % name),
        with Timer():
            result = func(D, inv_map, t, n)
        if not (prev_res == result).all():
        #if prev_res == None:
            prev_res = result
        else:
            assert (prev_res == result).all()

test_funcs_speed()
