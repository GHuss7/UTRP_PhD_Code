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
import pygmo as pg
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
name_input_data = "Mandl_Data"      # set the name of the input data
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
    'con_r' : 4,               # (aim for > [numNodes N ]/[maxNodes in route])
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


#routes_R = generate_initial_feasible_route_set_test(mx_dist, UTNDP_problem_1.problem_constraints.__dict__)

def determine_demand_per_route(list_of_routes, mx_demand):
    # Takes as input a list of routes and a demand matrix and calculates the total demand that a route satisfies
    # Returns a list containing the demand met by the route with the corresponding index
    demand_for_shortest_path_list = [0] * len(list_of_routes) # generates a list with only zeroes

    for list_counter in range(len(list_of_routes)):
        for i_counter in range(len(paths_shortest_all[list_counter])):
            for j_counter in range(len(paths_shortest_all[list_counter])):
                if i_counter != j_counter:  
                    demand_for_shortest_path_list[list_counter] = demand_for_shortest_path_list[list_counter] + mx_demand[list_of_routes[list_counter][i_counter], list_of_routes[list_counter][j_counter]]
    
    return demand_for_shortest_path_list


#%% Crossover type route generation based on unseen vertices probability
def routes_generation_unseen_prob(parent_i, parent_j, solution_len):
    """Crossover function for routes based on Mumford 2013's Crossover function
    for routes based on alternating between parents and including a route from
    each parent that maximises the unseen vertices added to the child route
    Note: only generates one child, needs to be tested for feasibility and repaired if needed"""
    parents = []  
    parents.append(copy.deepcopy(parent_i))
    parents.append(copy.deepcopy(parent_j))
    parent_index = random.randint(0,1) # counter to help alternate between parents  
    
    child_i = [] # define child
    parent_len = len(parent_i)
    
    # Randomly select the first seed solution for the child
    random_index = random.randint(0,parent_len-1)
    child_i.append(parents[parent_index][random_index]) # adds seed solution to parent
    del(parents[parent_index][random_index]) # removes the route from parent so that it is not evaluated again
    
    # Alternates parent solutions
    parent_index = gf.looped_incrementor(parent_index, 1)
    
    
    # Calculate the unseen proportions to select next route for inclusion into child
    while len(child_i) < solution_len:
        # Determines all nodes present in the child
        all_nodes_present = set([y for x in child_i for y in x]) # flatten all the elements in child
    
        parent_curr = parents[parent_index] # sets the current parent
        
        proportions = []
        for i_candidate in range(len(parent_curr)):
            R_i = set(parent_curr[i_candidate])
            if bool(R_i.intersection(all_nodes_present)): # test whether there is a common transfer point
                proportions.append(len(R_i - all_nodes_present) / len(R_i)) # calculate the proportion of unseen vertices
            else:
                proportions.append(0) # set proportion to zero so that it won't be chosen
        
        # Get route that maximises the proportion of unseen nodes included
        max_indices = set([i for i, j in enumerate(proportions) if j == max(proportions)]) # position of max proportion/s
        max_index = random.sample(max_indices, 1)[0] # selects only one index randomly between a possible tie, else the only one
        
        # Add the route to the child
        child_i.append(parent_curr[max_index]) # add max proportion unseen nodes route to the child
        del(parents[parent_index][max_index]) # removes the route from parent so that it is not evaluated again
        
        # Alternates parent solutions
        parent_index = gf.looped_incrementor(parent_index, 1)
    
    return child_i



demand_for_shortest_path_list = determine_demand_per_route(paths_shortest_all, mx_demand)

demand_for_shortest_path_list / sum(demand_for_shortest_path_list)

num_to_draw = 10
draw = list(np.random.choice(np.arange(len(paths_shortest_all)), num_to_draw,
              p=demand_for_shortest_path_list / sum(demand_for_shortest_path_list), replace=False))

chosen_routes = [paths_shortest_all[x] for x in draw]

initial_route_set_test = routes_generation_unseen_prob(paths_shortest_all, paths_shortest_all, UTNDP_problem_1.problem_constraints.con_r)

print(gf.test_route_feasibility(initial_route_set_test, UTNDP_problem_1.problem_constraints.__dict__))
gf.test_all_four_constraints_debug(initial_route_set_test, UTNDP_problem_1.problem_constraints.__dict__)

R_set = gc.Routes(initial_route_set_test)
R_set.plot_routes(UTNDP_problem_1)

repaired_set_1 = gf.repair_add_missing_from_terminal(initial_route_set_test, UTNDP_problem_1.problem_constraints.con_N_nodes, mapping_adjacent)
gf.test_all_four_constraints_debug(repaired_set_1, UTNDP_problem_1.problem_constraints.__dict__)

R_set_2 = gc.Routes(repaired_set_1)
R_set_2.plot_routes(UTNDP_problem_1)




distance_list_vertex_u = gf.get_graph_distance_levels_from_vertex_u(4,5,mapping_adjacent)

def repair_add_missing_from_terminal_multiple_backup(routes_R, n_nodes, mapping_adjacent):
    """ A function that searches for all the missing nodes, and tries to connect 
    them with one route's terminal node by trying to add one or more vertices to terminals"""
    all_nodes = [y for x in routes_R for y in x] # flatten all the elements in route
    
    # Initial test for all nodes present:
    if (len(set(all_nodes)) != n_nodes): # if not true, go on to testing for what nodes are ommited
    
        missing_nodes = list(set(range(n_nodes)).difference(set(all_nodes))) # find all the missing nodes
        random.shuffle(missing_nodes) # shuffles the nodes for randomness
        
        for missing_node in missing_nodes:
            # Find adjacent nodes of the missing nodes
            adj_nodes = mapping_adjacent[missing_node] # get the required nodes that are adjacent to 
            
            # Get terminal nodes
            terminal_nodes_front = [x[0] for x in routes_R] # get all the terminal nodes in the first position
            terminal_nodes_back = [x[-1] for x in routes_R] # get all the terminal nodes in the last position
            
            # Two cases, one from the front and one from the back
            terminal_nodes_front_candidates = set(terminal_nodes_front).intersection(adj_nodes) # Find intersection between first terminal nodes and adj nodes
            terminal_nodes_back_candidates = set(terminal_nodes_back).intersection(adj_nodes) # Find intersection between last terminal nodes and adj nodes
    
            if random.random() < 0.5:   # adds randomness to either front or back, and not just always one direction
                if bool(terminal_nodes_front_candidates):
                    random_adj_node = random.sample(terminal_nodes_front_candidates, 1)[0]
                    terminal_nodes_front.index(random_adj_node)
                    # Insert missing node at the front of adjacent terminal node
                    routes_R[terminal_nodes_front.index(random_adj_node)].insert(0, missing_node) 
                    
                elif bool(terminal_nodes_back_candidates):
                    random_adj_node = random.sample(terminal_nodes_back_candidates, 1)[0]
                    # Insert missing node at the back of adjacent terminal node
                    routes_R[terminal_nodes_back.index(random_adj_node)].append(missing_node) 
                    
            else:
                if bool(terminal_nodes_back_candidates):
                    random_adj_node = random.sample(terminal_nodes_back_candidates, 1)[0]
                    # Insert missing node at the back of adjacent terminal node
                    routes_R[terminal_nodes_back.index(random_adj_node)].append(missing_node) 
                
                elif bool(terminal_nodes_front_candidates):
                    random_adj_node = random.sample(terminal_nodes_front_candidates, 1)[0]
                    terminal_nodes_front.index(random_adj_node)
                    # Insert missing node at the front of adjacent terminal node
                    routes_R[terminal_nodes_front.index(random_adj_node)].insert(0, missing_node) 
                    
    return routes_R

routes_R = copy.deepcopy(initial_route_set_test)
n_nodes = len(mapping_adjacent)
gf.test_all_four_constraints_debug(initial_route_set_test, UTNDP_problem_1.problem_constraints.__dict__)

#def repair_add_missing_from_terminal_multiple_backup(routes_R, n_nodes, mapping_adjacent, UTNDP_problem_1):
""" A function that searches for all the missing nodes, and tries to connect 
them with one route's terminal node by trying to add one or more vertices to terminals"""

#TODO: REPLACE ALL n_nodes and mapping_adjacent with UTNDP_problem_1. ... 

max_depth = UTNDP_problem_1.problem_constraints.con_maxNodes - UTNDP_problem_1.problem_constraints.con_minNodes

all_nodes = [y for x in routes_R for y in x] # flatten all the elements in route

# Initial test for all nodes present:
if (len(set(all_nodes)) != n_nodes): # if not true, go on to testing for what nodes are ommited
    
    missing_nodes = list(set(range(n_nodes)).difference(set(all_nodes))) # find all the missing nodes
    random.shuffle(missing_nodes) # shuffles the nodes for randomness

    for missing_node in missing_nodes:
        # Find adjacent nodes of the missing nodes
        distance_list_vertex_u = gf.get_graph_distance_levels_from_vertex_u(missing_node,max_depth,mapping_adjacent)

        # Get terminal nodes
        terminal_nodes_front = [x[0] for x in routes_R] # get all the terminal nodes in the first position
        terminal_nodes_back = [x[-1] for x in routes_R] # get all the terminal nodes in the last position
        terminal_nodes_all = terminal_nodes_front + terminal_nodes_back

        missing_node_insertions = []

        for distance_dept in range(len(distance_list_vertex_u)-1):
            intersection_terminal_dist_list = set(terminal_nodes_all).intersection(set(distance_list_vertex_u[distance_dept+1]))
            if intersection_terminal_dist_list:
                random_adj_node = random.choice(tuple(intersection_terminal_dist_list))
                missing_node_insertions.extend([random_adj_node])
                if distance_dept == 1:
                    
                    if terminal_nodes_all.index(random_adj_node) < len(routes_R):
                        routes_R[terminal_nodes_all.index(random_adj_node)].insert(0, missing_node) 
                    else:
                        routes_R[terminal_nodes_all.index(random_adj_node) - len(routes_R)].append(missing_node) 
                    break
                        
R_set_3 = gc.Routes(routes_R)
R_set_3.plot_routes(UTNDP_problem_1)