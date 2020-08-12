# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:48:40 2020

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
name_input_data = "Mandl_Data"      # set the name of the input data
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)
del name_input_data

# %% Set variables

Choice_generate_initial_set = True # the alternative loads a set that is prespecified
Choice_print_results = True 
Choice_conduct_sensitivity_analysis = True 

'''Enter the number of allowed routes''' 
parameters_constraints = {
'con_r' : 6,               # (aim for > [numNodes N ]/[maxNodes in route])
'con_minNodes' : 3,                        # minimum nodes in a route
'con_maxNodes' : 10,                       # maximum nodes in a route
'con_N_nodes' : len(mx_dist)              # number of nodes in the network
}

parameters_input = {
'total_demand' : sum(sum(mx_demand))/2, # total demand from demand matrix
'n' : len(mx_dist), # total number of nodes
'wt' : 0, # waiting time [min]
'tp' : 5, # transfer penalty [min]
'Problem_name' : "Mandl_UTRP", # Specify the name of the problem currently being addresses
'ref_point_max_f1_ATT' : 15.1304, # max f1_ATT for the Hypervolume calculations
'ref_point_min_f1_ATT' : 10.3301, # min f1_ATT for the Hypervolume calculations
'ref_point_max_f2_TRT' : 224, # max f2_TRT for the Hypervolume calculations
'ref_point_min_f2_TRT' : 63 # min f2_TRT for the Hypervolume calculations
}

parameters_SA_routes={
"method" : "SA",
"max_iterations_t" : 1000, # maximum allowable number length of iterations per epoch c
"max_accepts" : 1, # maximum number of accepted moves per epoch
"max_attempts" : 1, # maximum number of attempted moves per epoch
"max_reheating_times" : 1, # the maximum number of times that reheating can take place
"max_poor_epochs" : 1, # maximum number of epochs which may pass without the acceptance of any new solution
"Temp" : 1,  # starting temperature and a geometric cooling schedule is used on it
"M_iterations_for_temp" : 1000, # the number of initial iterations to establish initial starting temperature
"Cooling_rate" : 0.97, # the geometric cooling rate
"Reheating_rate" : 1.02, # the geometric reheating rate
"number_of_initial_solutions" : 2, # sets the number of initial solutions to generate as starting position
"Feasibility_repair_attempts" : 2 # the max number of edges that will be added and/or removed to try and repair the route feasibility
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


# %% Route Manipulations

if False:
    # %% Route visualisations
    '''Graph visualisation'''
    # gv.plotRouteSet2(mx_dist, routes_R, mx_coords)
    gv.plotRouteSet2(mx_dist, routes_R_new, mx_coords)
    #gv.plotRouteSet2(mx_dist, gf.convert_routes_str2list(df_routes_R_initial_set.loc[6739,'Routes']), mx_coords)  # gets a route set from initial
    #points = list()
    #for i in range(len(df_overall_pareto_set)):
        #points = points.append([df_overall_pareto_set.iloc[:,1], df_overall_pareto_set.iloc[:,2]])
    
    # %% Better repair strategy --- tested on Mandl
    """Tests to create a better repair strategy"""
    routes_R = "4-3-5-7-9-13-12*7-14-6*9-7-5-2*" # removed 5 nodes for test
    routes_R = "4-3-5-7-9-13-12*0-1-2-5-14-8*12-13-9*13-12-10*7-14-6*9-7-5-2*" # removed the 11 node for test
    routes_R = gf.convert_routes_str2list(routes_R)
    gv.plotRouteSet2(mx_dist, routes_R, mx_coords)
    
    # Repair function
    routes_R = gf.repair_add_missing_from_terminal(routes_R, UTNDP_problem_1.problem_inputs.n, mapping_adjacent)
    
    # Mutate
    candidate_routes_R = gf.mutate_routes_two_intertwine(routes_R, UTNDP_problem_1.problem_constraints.__dict__, mapping_adjacent)        
    #gv.plotRouteSet2(mx_dist, candidate_routes_R, mx_coords)     
           
    # Test all 4 constraints        
    gf.test_all_four_constraints_debug(candidate_routes_R, UTNDP_problem_1.problem_constraints.__dict__)
    
    #  Smart Crossover (Mumford 2013)
    import DSS_UTNDP_Functions as gf
    	
    parent_i = "5-7-9-12*9-7-5-3-4*0-1-2-5-14-6*13-9-6-14-8*1-2-5-14*9-10-11-3*"
    parent_j = "7-9-12-10-11*6-9-7-5-3-1*0-1-2-5-14-6-9*12-13-9-6-14-8*4-1-2-5-14-6*9-10-11-3-4*"
    
    parent_i = "7-9-12*9-7-5-3-4*0-1-2-5-14-6*13-9-6-14-8*1-2-5*9-10-11-3*"
    parent_j = "7-9-12-10*5-3-1*0-1-2-5-14-6-9*12-13-9-6-14-8*4-1-2-5-14-6*9-10-11-3-4*"
    
    parent_i = gf.convert_routes_str2list(parent_i)
    parent_j = gf.convert_routes_str2list(parent_j)
    
            
    child_i = gf.crossover_routes_unseen_prob(parent_i, parent_j) 
    
    gf.test_all_four_constraints_debug(child_i, UTNDP_problem_1.problem_constraints.__dict__)
    
    
    gv.plotRouteSet2(mx_dist, child_i, mx_coords)
    
    # Testers
    routes_R = "11-3-1-0*3-5-7*12-10-9-7-5-2*5-14-6*6-14-5-3-4*7-9-13*"
    routes_R = gf.convert_routes_str2list(routes_R)
    gv.plotRouteSet2(mx_dist, routes_R, mx_coords)
    gf.test_all_four_constraints_debug(routes_R, UTNDP_problem_1.problem_constraints.__dict__)
    
    routes_R = gf.repair_add_missing_from_terminal(routes_R, UTNDP_problem_1.problem_inputs.n, UTNDP_problem_1.mapping_adjacent)
    gv.plotRouteSet2(mx_dist, routes_R, mx_coords)
    
    #%% Repair for the case of disconnected components
    #TODO: add a node to a terminal node from the disconnected components
    G = gf.create_nx_graph_from_routes_list(routes_R)
    
    #routes_R = gf.repair_add_missing_from_terminal(routes_R, UTNDP_problem_1.problem_inputs.n, UTNDP_problem_1.mapping_adjacent)
    #nx.draw_networkx(G)
    
    all_nodes = set([y for x in routes_R for y in x])
    network_components = list(nx.connected_components(G))
    
    disconnected_component = min(network_components, key=len) #gets the component with the minimum elements
    
    for element in disconnected_component:
        # Find a node adjacent and in the other component to connect
        connections = set(UTNDP_problem_1.mapping_adjacent[element]).intersection(all_nodes.difference(disconnected_component))
        print(connections)
    
    
    # while len(network_components) != 1:
    routes_R = [[1, 2, 5], [0, 1, 2, 5, 14, 6, 9], [9, 10, 11, 3], [12, 13, 9, 6, 14, 8], [9, 7, 5, 3, 4], [4, 1, 2, 5, 14, 6]]
