# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:21:33 2020

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
import DSS_UTNDP_Functions as gf
import DSS_Visualisation as gv
import EvaluateRouteSet as ev

# todo def main_dss(): # create a main function to encapsulate the main body
# def main():
# %% Load the respective files
#name_input_data = "SSML_STB_DAY_SUM_0700_1700"      # set the name of the input data
name_input_data = "Mandl_UTRP"      # set the name of the input data
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)
# del name_input_data

if name_input_data == "SSML_STB_DAY_SUM_0700_1700":
    # %% Set variables
    Decisions = {
    "Choice_generate_initial_set" : True, # the alternative loads a set that is prespecified, False is default for MANDL NB
    "Choice_print_results" : True, 
    "Choice_conduct_sensitivity_analysis" : True,
    "Choice_init_temp_with_trial_runs" : False, # runs M trial runs for the initial temperature
    "Choice_normal_run" : False, # choose this for a normal run without Sensitivity Analysis
    "Choice_import_saved_set" : False, # import the prespecified set
    #"Set_name" : "Overall_Pareto_test_set_for_GA.csv" # the name of the set in the main working folder
    "Set_name" : "Overall_Pareto_set_for_case_study_GA.csv" # the name of the set in the main working folder
    }
    
    '''Enter the number of allowed routes''' 
    parameters_constraints = {
    'con_r' : 8,               # (aim for > [numNodes N ]/[maxNodes in route])
    'con_minNodes' : 2,                        # minimum nodes in a route
    'con_maxNodes' : 6,                       # maximum nodes in a route
    'con_N_nodes' : len(mx_dist)              # number of nodes in the network
    }
    
    parameters_input = {
    'total_demand' : sum(sum(mx_demand))/2, # total demand from demand matrix
    'n' : len(mx_dist), # total number of nodes
    'wt' : 0, # waiting time [min]
    'tp' : 5, # transfer penalty [min]
    'Problem_name' : "Case_study_UTRP_DBMOSA", # Specify the name of the problem currently being addresses
    'ref_point_max_f1_ATT' : 15, # max f1_ATT for the Hypervolume calculations
    'ref_point_min_f1_ATT' : 10, # min f1_ATT for the Hypervolume calculations
    'ref_point_max_f2_TRT' : 224, # max f2_TRT for the Hypervolume calculations
    'ref_point_min_f2_TRT' : 63 # min f2_TRT for the Hypervolume calculations
    }
    
    parameters_SA_routes={
    "method" : "SA",
    # ALSO: t_max > A_min (max_iterations_t > min_accepts)
    "max_iterations_t" : 250, # maximum allowable number length of iterations per epoch; Danie PhD (pg. 98): Dreo et al. chose 100
    "max_total_iterations" : 25000, # the total number of accepts that are allowed
    "max_epochs" : 1500, # the maximum number of epochs that are allowed
    "min_accepts" : 25, # minimum number of accepted moves per epoch; Danie PhD (pg. 98): Dreo et al. chose 12N (N being some d.o.f.)
    "max_attempts" : 50, # maximum number of attempted moves per epoch
    "max_reheating_times" : 5, # the maximum number of times that reheating can take place
    "max_poor_epochs" : 400, # maximum number of epochs which may pass without the acceptance of any new solution
    "Temp" : 10,  # starting temperature and a geometric cooling schedule is used on it # M = 1000 gives 93.249866 from 20 runs
    "M_iterations_for_temp" : 1000, # the number of initial iterations to establish initial starting temperature
    "Cooling_rate" : 0.97, # the geometric cooling rate 0.97 has been doing good, but M =1000 gives 0.996168
    "Reheating_rate" : 1.05, # the geometric reheating rate
    "number_of_initial_solutions" : 2, # sets the number of initial solutions to generate as starting position
    "Feasibility_repair_attempts" : 3, # the max number of edges that will be added and/or removed to try and repair the route feasibility
    "number_of_runs" : 20, # number of runs to complete John 2016 set 20
    }
    
    '''Set the reference point for the Hypervolume calculations'''
    max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
    min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])
    
else:
    # %% Set variables
    Decisions = {
    "Choice_generate_initial_set" : True, # the alternative loads a set that is prespecified, False is default for MANDL NB
    "Choice_print_results" : True, 
    "Choice_conduct_sensitivity_analysis" : True,
    "Choice_init_temp_with_trial_runs" : False, # runs M trial runs for the initial temperature
    "Choice_normal_run" : False, # choose this for a normal run without Sensitivity Analysis
    "Choice_import_saved_set" : False, # import the prespecified set
    #"Set_name" : "Overall_Pareto_test_set_for_GA.csv" # the name of the set in the main working folder
    "Set_name" : "Overall_Pareto_set_for_case_study_GA.csv" # the name of the set in the main working folder
    }
    
    '''Enter the number of allowed routes''' 
    parameters_constraints = {
    'con_r' : 2,               # (aim for > [numNodes N ]/[maxNodes in route])
    'con_minNodes' : 2,                        # minimum nodes in a route
    'con_maxNodes' : 15,                       # maximum nodes in a route
    'con_N_nodes' : len(mx_dist)              # number of nodes in the network
    }
    
    parameters_input = {
    'total_demand' : sum(sum(mx_demand))/2, # total demand from demand matrix
    'n' : len(mx_dist), # total number of nodes
    'wt' : 0, # waiting time [min]
    'tp' : 5, # transfer penalty [min]
    'Problem_name' : f"{name_input_data}_UTRP_DBMOSA", # Specify the name of the problem currently being addresses
    'ref_point_max_f1_ATT' : 30, # max f1_ATT for the Hypervolume calculations
    'ref_point_min_f1_ATT' : 10, # min f1_ATT for the Hypervolume calculations
    'ref_point_max_f2_TRT' : 400, # max f2_TRT for the Hypervolume calculations
    'ref_point_min_f2_TRT' : 63 # min f2_TRT for the Hypervolume calculations
    }
    
    parameters_SA_routes={
    "method" : "SA",
    # ALSO: t_max > A_min (max_iterations_t > min_accepts)
    "max_iterations_t" : 250, # maximum allowable number length of iterations per epoch; Danie PhD (pg. 98): Dreo et al. chose 100
    "max_total_iterations" : 70000, # the total number of accepts that are allowed
    "max_epochs" : 2000, # the maximum number of epochs that are allowed
    "min_accepts" : 25, # minimum number of accepted moves per epoch; Danie PhD (pg. 98): Dreo et al. chose 12N (N being some d.o.f.)
    "max_attempts" : 50, # maximum number of attempted moves per epoch
    "max_reheating_times" : 5, # the maximum number of times that reheating can take place
    "max_poor_epochs" : 400, # maximum number of epochs which may pass without the acceptance of any new solution
    "Temp" : 10,  # starting temperature and a geometric cooling schedule is used on it # M = 1000 gives 93.249866 from 20 runs
    "M_iterations_for_temp" : 1000, # the number of initial iterations to establish initial starting temperature
    "Cooling_rate" : 0.97, # the geometric cooling rate 0.97 has been doing good, but M =1000 gives 0.996168
    "Reheating_rate" : 1.05, # the geometric reheating rate
    "number_of_initial_solutions" : 2, # sets the number of initial solutions to generate as starting position
    "Feasibility_repair_attempts" : 3, # the max number of edges that will be added and/or removed to try and repair the route feasibility
    "number_of_runs" : 20, # number of runs to complete John 2016 set 20
    }
    
    '''Set the reference point for the Hypervolume calculations'''
    max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
    min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])
    
  
# %% Define the adjacent mapping of each node
mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes

#%% Define the UTNDP Problem      
UTNDP_problem_1 = gc.UTNDP_problem()
UTNDP_problem_1.problem_data = gc.Problem_data(mx_dist, mx_demand, mx_coords)
UTNDP_problem_1.problem_constraints = gc.Problem_constraints(parameters_constraints)
UTNDP_problem_1.problem_inputs = gc.Problem_inputs(parameters_input)
UTNDP_problem_1.problem_SA_parameters = gc.Problem_metaheuristic_inputs(parameters_SA_routes)
UTNDP_problem_1.mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes
UTNDP_problem_1.max_objs = max_objs
UTNDP_problem_1.min_objs = min_objs
UTNDP_problem_1.add_text = "" # define the additional text for the file name
# UTNDP_problem_1.R_routes = R_routes


#def generate_initial_feasible_route_set_test(mx_dist, parameters_constraints):
con_minNodes = parameters_constraints['con_minNodes']
con_maxNodes = parameters_constraints['con_maxNodes']
con_r = parameters_constraints['con_r']

  
'''Create the transit network graph'''
g_tn = gf.create_igraph_from_dist_mx(mx_dist)

paths_shortest_all = gf.get_all_shortest_paths(g_tn) # Generate all the shortest paths

# set(paths_shortest_all) 
"""Remove duplicate lists in reverse order"""

# Shorten the candidate routes according to the constraints
for i in range(len(paths_shortest_all)-1, -1, -1):
    if len(paths_shortest_all[i]) < con_minNodes or len(paths_shortest_all[i]) > con_maxNodes:  
        del paths_shortest_all[i]

# Generate initial feasible solution
routes_R = gf.generate_feasible_solution(paths_shortest_all, con_r, len(mx_dist), 100000)

    #return routes_R

demand_for_routes_list = gf.determine_demand_per_route(paths_shortest_all, mx_demand)

demand_for_routes_list / sum(demand_for_routes_list)

num_to_draw = 10
draw = list(np.random.choice(np.arange(len(paths_shortest_all)), num_to_draw,
              p=demand_for_routes_list / sum(demand_for_routes_list), replace=False))

chosen_routes = [paths_shortest_all[x] for x in draw]

initial_route_set_test = gf.routes_generation_unseen_prob(paths_shortest_all, paths_shortest_all, UTNDP_problem_1.problem_constraints.con_r)

print(gf.test_route_feasibility(initial_route_set_test, UTNDP_problem_1.problem_constraints.__dict__))
gf.test_all_four_constraints_debug(initial_route_set_test, UTNDP_problem_1.problem_constraints.__dict__)


routes_R = copy.deepcopy(initial_route_set_test)

R_set = gc.Routes(routes_R)
R_set.plot_routes(UTNDP_problem_1)

n_nodes = len(UTNDP_problem_1.mapping_adjacent)
gf.test_all_four_constraints_debug(routes_R, UTNDP_problem_1.problem_constraints.__dict__)


routes_R = gf.repair_add_missing_from_terminal_multiple(routes_R, UTNDP_problem_1)

R_set_3 = gc.Routes(routes_R)
R_set_3.plot_routes(UTNDP_problem_1)

#%% Mutation tests
# route_to_mutate = gf.convert_routes_str2list("13-9-7-5-3-4*1-3-11-10-9-6-14-8*0-1-2-5-7-9-10-12-13*6-14-5-2-1-4*6-14-5-3-1-0*6-14-7*")
paths_shortest_all_unique = gf.remove_half_duplicate_routes(paths_shortest_all)

route_to_mutate = gf.convert_routes_str2list("13-9-7-5-3-4*0-1-2-5-7-9-10-12-13*6-14-5-2-1-4*6-14-5-3-1-0*6-14-7*")
R_to_mutate = gc.Routes(route_to_mutate)
R_to_mutate.plot_routes(UTNDP_problem_1)

demand_for_routes_list = gf.determine_demand_per_route(paths_shortest_all_unique, mx_demand)

all_nodes = [y for x in route_to_mutate for y in x] # flatten all the elements in route
    
# Initial test for all nodes present:
if (len(set(all_nodes)) != n_nodes): # if not true, go on to testing for what nodes are ommited
    missing_nodes = list(set(range(n_nodes)).difference(set(all_nodes))) # find all the missing nodes

indices_of_compatible_routes = []
for path_index in range(len(paths_shortest_all_unique)):
    if set(missing_nodes).issubset(set(paths_shortest_all_unique[path_index])):
        indices_of_compatible_routes.append(path_index)

path_to_add = paths_shortest_all_unique[random.choice(indices_of_compatible_routes)]
new_route = copy.deepcopy(route_to_mutate)
new_route.extend([path_to_add])

R_new = gc.Routes(new_route)
R_new.plot_routes(UTNDP_problem_1)
