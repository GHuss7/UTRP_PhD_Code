# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:22:17 2020

@author: 17832020
"""

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
import math
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
'con_minNodes' : 2,                        # minimum nodes in a route
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
# ALSO: t_max > A_min (max_iterations_t > min_accepts)
"max_iterations_t" : 1000, # maximum allowable number length of iterations per epoch; Danie PhD (pg. 98): Dreo et al. chose 100
"min_accepts" : 5, # minimum number of accepted moves per epoch; Danie PhD (pg. 98): Dreo et al. chose 12N (N being some d.o.f.)
"max_attempts" : 3, # maximum number of attempted moves per epoch
"max_reheating_times" : 3, # the maximum number of times that reheating can take place
"max_poor_epochs" : 3, # maximum number of epochs which may pass without the acceptance of any new solution
"Temp" : 200,  # starting temperature and a geometric cooling schedule is used on it
"M_iterations_for_temp" : 1000, # the number of initial iterations to establish initial starting temperature
"Cooling_rate" : 0.97, # the geometric cooling rate
"Reheating_rate" : 1.1, # the geometric reheating rate
"number_of_initial_solutions" : 1, # sets the number of initial solutions to generate as starting position
"Feasibility_repair_attempts" : 2, # the max number of edges that will be added and/or removed to try and repair the route feasibility
"number_of_runs" : 2, # number of runs to complete John 2016 set 20
}
  
# %% Define the adjacent mapping of each node
mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes

#%% Define the UTNDP Problem      
UTNDP_problem_1 = gc.UTNDP_problem()
UTNDP_problem_1.problem_data = gc.Problem_data(mx_dist, mx_demand, mx_coords)
UTNDP_problem_1.problem_constraints = gc.Problem_constraints(parameters_constraints)
UTNDP_problem_1.problem_inputs = gc.Problem_inputs(parameters_input)
UTNDP_problem_1.problem_SA_parameters = gc.Problem_metaheuristic_inputs(parameters_SA_routes)
UTNDP_problem_1.mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes
UTNDP_problem_1.add_text = "" # define the additional text for the file name
# UTNDP_problem_1.R_routes = R_routes



""" Keep track of the stats """
stats_overall = {
    'execution_start_time' : datetime.datetime.now() # enter the begin time
    } 

stats = {} # define the stats dictionary

#%% Define the Objective UTNDP functions
def fn_obj(routes, UTNDP_problem_input):
    return (ev.evalObjs(routes, 
            UTNDP_problem_input.problem_data.mx_dist, 
            UTNDP_problem_input.problem_data.mx_demand, 
            UTNDP_problem_input.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)

def fn_obj_row(routes):
    return (ev.evalObjs(routes, 
            UTNDP_problem_1.problem_data.mx_dist, 
            UTNDP_problem_1.problem_data.mx_demand, 
            UTNDP_problem_1.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)

# %% Generate an initial feasible solution
routes_R = gf.generate_initial_feasible_route_set(mx_dist, UTNDP_problem_1.problem_constraints.__dict__)

if UTNDP_problem_1.problem_constraints.con_r != len(routes_R): # if the constraint was leveraged, update constraints
    UTNDP_problem_1.problem_constraints.con_r = len(routes_R)
    print("Number of allowed routes constraint updated to", UTNDP_problem_1.problem_constraints.con_r)

def init_temp_trial_searches(UTNDP_problem, number_of_runs=1, P_0=0.999, P_N=0.001, N_search_epochs = 1000):
    '''Test for starting temperature and cooling schedule'''
    """ number_of_runs: sets more trial runs for better estimates (averages)
        P_0: initial probability of acceptance
        P_N: final probability of acceptance
        N_search_epochs: roughly the number of desired search epochs
        returns: the starting temp and Beta coefficient for geometric cooling
    """
    df_results = pd.DataFrame(columns=["Avg_E","T_0","T_N","Beta"])
    
    for run_nr in range(number_of_runs):
        
        routes_R_initial_set, df_routes_R_initial_set = gf.generate_initial_route_sets(UTNDP_problem, False)
    
        for route_set_nr in range(len(routes_R_initial_set)):
            stats['begin_time'] = datetime.datetime.now() # enter the begin time
            stats['run_number'] = f"{run_nr + 1}.{route_set_nr}"
        
        
            routes_R = routes_R_initial_set[route_set_nr] # Choose the initial route set to begin with
            '''Initiate algorithm'''
            iteration_t = 1 # Initialise the number of iterations 
            counter_archive = 0
            
            df_energy_values = pd.DataFrame(columns=["Delta_E"]) 
            df_archive = pd.DataFrame(columns=["f1_ATT","f2_TRT","Routes"]) # create an archive in the correct format        
            
            f_cur = fn_obj(routes_R, UTNDP_problem)
            df_archive.loc[0] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)]            
    
            while (iteration_t <= UTNDP_problem.problem_SA_parameters.M_iterations_for_temp):
                '''Generate neighbouring solution'''
                routes_R_new = gf.perturb_make_small_change(routes_R, UTNDP_problem.problem_constraints.con_r, mapping_adjacent)
                
                while not gf.test_route_feasibility(routes_R_new, UTNDP_problem.problem_constraints.__dict__):    # tests whether the new route is feasible
                    for i in range(UTNDP_problem.problem_SA_parameters.Feasibility_repair_attempts): # this tries to fix the feasibility, but can be time consuming, 
                                            # could also include a "connectivity" characteristic to help repair graph
                        routes_R_new = gf.perturb_make_small_change(routes_R_new, UTNDP_problem.problem_constraints.con_r, mapping_adjacent)
                        if gf.test_route_feasibility(routes_R_new, UTNDP_problem.problem_constraints.__dict__):
                            break
                    routes_R_new = gf.perturb_make_small_change(routes_R, UTNDP_problem.problem_constraints.con_r, mapping_adjacent) # if unsuccesful, start over
            
                f_new = fn_obj(routes_R_new, UTNDP_problem)
            
                df_energy_values.loc[len(df_energy_values)] = [abs(gf.energy_function_for_initial_temp(df_archive, f_cur[0], f_cur[1], f_new[0], f_new[1]))]
            
                '''Test solution acceptance and add to archive if accepted and non-dominated'''
                routes_R = routes_R_new
                f_cur = f_new
                
                df_archive.loc[counter_archive] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)] # adds the new solution
                counter_archive = counter_archive + 1 # this helps with speed 
                
                iteration_t = iteration_t + 1
        
            avg_E = df_energy_values[['Delta_E']].mean(axis=0)[0]
            T_0 = -(avg_E / math.log10(P_0))
            T_N = -(avg_E / math.log10(P_N))
            
            Beta = math.exp((math.log10(T_N) - math.log10(T_0)) / N_search_epochs)
        
        df_results.loc[run_nr] = [avg_E, T_0, T_N, Beta]

    result_means = df_results.mean(axis=0)
    return result_means[1], result_means[3] # T_0 and Beta cooling ratio
     

init_temp_trial_searches(UTNDP_problem_1, number_of_runs=1)
