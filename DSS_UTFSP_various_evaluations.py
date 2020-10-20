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
import pygmo as pg
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
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import EvaluateRouteSet as ev
import DSS_UTNDP_Classes as gc

# todo def main_dss(): # create a main function to encapsulate the main body
# def main():
    
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

from pymoo.model.selection import Selection
from pymoo.util.misc import random_permuations
    
# %% Load the respective files
#name_input_data = "SSML_STB_1700_UTFSP"      # set the name of the input data
name_input_data = "SSML_STB_1200_UTFSP"      # set the name of the input data
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)
if os.path.exists("./Input_Data/"+name_input_data+"/Walk_Matrix.csv"):
    mx_walk = pd.read_csv("./Input_Data/"+name_input_data+"/Walk_Matrix.csv") 
    mx_walk = gf.format_mx_dist(mx_walk)
    mx_walk = mx_walk.values
else:
    mx_walk = False

# %% Set input parameters
Decisions = {
#"Choice_generate_initial_set" : True, 
"Choice_print_results" : True, 
"Choice_conduct_sensitivity_analysis" : False,
"Choice_consider_walk_links" : True,
"Choice_import_dictionaries" : True,
"Choice_print_full_data_for_analysis" : True,
"Set_name" : "Overall_Pareto_set_for_case_study_GA.csv" # the name of the set in the main working folder
}

# Disables walk links
if not(Decisions["Choice_consider_walk_links"]):
    mx_walk = False

# Load the respective dictionaries for the instance
if Decisions["Choice_import_dictionaries"]:
    parameters_constraints = json.load(open("./Input_Data/"+name_input_data+"/parameters_constraints.json"))
    parameters_input = json.load(open("./Input_Data/"+name_input_data+"/parameters_input.json"))
    parameters_GA_frequencies = json.load(open("./Input_Data/"+name_input_data+"/parameters_GA_frequencies.json"))

else:
    '''State the various parameter constraints''' 
    parameters_constraints = {
    'con_r' : 8,               # number of allowed routes (aim for > [numNodes N ]/[maxNodes in route])
    'con_minNodes' : 2,                        # minimum nodes in a route
    'con_maxNodes' : 6,                       # maximum nodes in a route
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
    'Problem_name' : name_input_data, # Specify the name of the problem currently being addressed
    'walkFactor' : 100, # factor it takes longer to walk than to drive
    'boardingTime' : 0.1, # assume boarding and alighting time = 6 seconds
    'alightingTime' : 0.1, # problem when alighting time = 0 (good test 0.5)(0.1 also works)
    'large_dist' : int(mx_dist.max()), # the large number from the distance matrix
    'alpha_const_inter' : 0.5 # constant for interarrival times relationship 0.5 (Spiess 1989)
    }
    
    '''State the various GA input parameters for frequency setting''' 
    parameters_GA_frequencies={
    "method" : "GA",
    "population_size" : 200, #should be an even number, John: 200
    "generations" : 40, # John: 200
    "number_of_runs" : 1, # John: 20
    "crossover_probability" : 0.7,  # John: 0.9 BEST: 0.8
    "crossover_distribution_index" : 5,
    "mutation_probability" : 1/parameters_constraints["con_r"], # John: 1/|Route set| -> set later BEST: 0.1 
    "mutation_distribution_index" : 10,
    "tournament_size" : 2,
    "termination_criterion" : "StoppingByEvaluations",
    "max_evaluations" : 40000,
    "number_of_variables" : "not_set",
    "number_of_objectives" : 2 # this could still be automated in the future
    }


#%% Input parameter tests

'''Test the inputs for feasibility'''
# Test feasibility if there are enough buses to cover each route once
if parameters_constraints["con_r"] > parameters_constraints["con_fleet_size"]:
    print("Warning: Number of available vehicles are less than the number of routes.\n"\
          "Number of routes allowed set to "+ str(parameters_constraints["con_r"]))
    parameters_constraints["con_r"] = parameters_constraints["con_fleet_size"]

# %% Import the route set to be evaluated
''' Import the route set '''    
with open("./Input_Data/"+name_input_data+"/Route_set.txt", 'r') as file:
    route_str = file.read().replace('\n', '')

R_x = gf.convert_routes_str2list(route_str) # convert route set into list
del route_str

if 0 not in set([y for x in R_x for y in x]): # NB: test whether the route is in the correct format containing a 0
    for i in range(len(R_x)): # get routes in the correct format
        R_x[i] = [x - 1 for x in R_x[i]] # subtract 1 from each element in the list
    del i
R_routes = gf2.Routes(R_x)

parameters_GA_frequencies["mutation_probability"] = 1/(len(R_routes.routes))

# %% Initialise the decision variables
'''Initialise the decision variables'''
F_frequencies = gf2.Frequencies(parameters_constraints['con_r']) 

F_frequencies.set_frequencies(np.full(len(R_x), 1/5)) 
    
F_x = F_frequencies.frequencies

parameters_GA_frequencies["number_of_variables"] = len(F_x)

#%% Define the UTFSP Problem      
UTFSP_problem_1 = gf2.UTFSP_problem()
UTFSP_problem_1.problem_data = gf2.Problem_data(mx_dist, mx_demand, mx_coords, mx_walk)
UTFSP_problem_1.problem_constraints = gf2.Problem_constraints(parameters_constraints)
UTFSP_problem_1.problem_inputs = gf2.Problem_inputs(parameters_input)
UTFSP_problem_1.problem_GA_parameters = gf2.Problem_GA_inputs(parameters_GA_frequencies)
UTFSP_problem_1.R_routes = R_routes
UTFSP_problem_1.add_text = "" # define the additional text for the file name

#%% Define the Transit network
TN = gf2.Transit_network(R_x, F_x, mx_dist, mx_demand, parameters_input, mx_walk) 

# %% TESTS: 

if False:  
    
    def fn_obj(frequencies, UTFSP_problem_input):
        return (gf2.f3_ETT(UTFSP_problem_input.R_routes.routes,
                           frequencies, 
                           UTFSP_problem_input.problem_data.mx_dist, 
                           UTFSP_problem_input.problem_data.mx_demand, 
                           UTFSP_problem_input.problem_inputs.__dict__), #f3_ETT
                gf2.f4_TBR(UTFSP_problem_input.R_routes.routes, 
                           frequencies, 
                           UTFSP_problem_input.problem_data.mx_dist)) #f4_TBR
    
    def fn_obj_row(frequencies):
        
        return (gf2.f3_ETT(UTFSP_problem_1.R_routes.routes,
                           frequencies, 
                           UTFSP_problem_1.problem_data.mx_dist, 
                           UTFSP_problem_1.problem_data.mx_demand, 
                           UTFSP_problem_1.problem_inputs.__dict__), #f3_ETT
                gf2.f4_TBR(UTFSP_problem_1.R_routes.routes, 
                           frequencies, 
                           UTFSP_problem_1.problem_data.mx_dist)) #f4_TBR
       

    offspring_variables = np.array([[1/5, 1/5, 1/7, 1/5, 1/5, 1/5],
                                    [1/30, 1/30, 1/30, 1/30, 1/30, 1/30]])
    
    F_x = offspring_variables[1,]
    
    # offspring_variables = df_res[1:,3:]
       
    
    # start = time.perf_counter()
    # offspring_objectives = np.apply_along_axis(fn_obj_row, 1, offspring_variables)
    # finish = time.perf_counter()
    # print(f'Finished in {round(finish-start, 2)} second(s): apply_along_axis')
    if True:
        start = time.perf_counter()
        offspring_objectives = gf2.calc_fn_obj_for_np_array(fn_obj_row, offspring_variables) # takses about 6 seconds for one evaluation   
        finish = time.perf_counter()
        print(f'Finished in {round(finish-start, 2)} second(s): for loop')
        
        print(f'Freq: 1/5 \t f_1: {round(offspring_objectives[0,0], 3)} | 15.189 \t f_2: {round(offspring_objectives[0,1], 3)} | 24.286')
        print(f'Freq: 1/30 \t f_1: {round(offspring_objectives[1,0], 3)} | 31.814 \t f_2: {round(offspring_objectives[1,1], 3)} | 4.2')

    

A_link_volumes = TN.mx_volumes_links
A_mx_C_a = TN.mx_C_a
# fn_obj(np.full((1,UTFSP_problem_1.problem_constraints.con_r), 1/10)[0],UTFSP_problem_1)
#TN.R_routes
#gv.plotUTFSPAndSavePDF(mx_dist,routes_R, mx_coords, name)

if True:
    # %% Transit network tests
    TN = gf2.Transit_network(R_x, F_x, mx_dist, mx_demand, parameters_input) 
    
    B_df_opt_strat_alg = TN.df_opt_strat_alg
    B_df_opt_strat_alg_named = TN.df_opt_strat_alg_named
    B_mx_A_bar_strategy_lines = TN.mx_A_bar_strategy_lines
    B_df_A_bar_strategy_lines = TN.df_A_bar_strategy_lines
    
    # B_df_opt_strat_alg[0].iloc[:,0]
    
    def fn_obj_test(frequencies, UTFSP_problem_input):
        return (gf2.f3_ETT(UTFSP_problem_input.R_routes.routes,
                           frequencies, 
                           UTFSP_problem_input.problem_data.mx_dist, 
                           UTFSP_problem_input.problem_data.mx_demand, 
                           UTFSP_problem_input.problem_inputs.__dict__,
                           UTFSP_problem_input.problem_data.mx_walk), #f3_ETT
                gf2.f4_TBR(UTFSP_problem_input.R_routes.routes, 
                           frequencies, 
                           UTFSP_problem_input.problem_data.mx_dist)) #f4_TBR
    
    fn_obj_test(np.full((1,UTFSP_problem_1.problem_constraints.con_r), 1/30)[0], UTFSP_problem_1)
    
    
    gf2.f3_ETT(UTFSP_problem_1.R_routes.routes,
                           np.full((1,UTFSP_problem_1.problem_constraints.con_r), 1/30)[0], 
                           UTFSP_problem_1.problem_data.mx_dist, 
                           UTFSP_problem_1.problem_data.mx_demand, 
                           UTFSP_problem_1.problem_inputs.__dict__)
    
    
    gf2.f3_ETT(UTFSP_problem_1.R_routes.routes,
                           np.full((1,UTFSP_problem_1.problem_constraints.con_r), 1/30)[0], 
                           UTFSP_problem_1.problem_data.mx_dist, 
                           UTFSP_problem_1.problem_data.mx_demand, 
                           UTFSP_problem_1.problem_inputs.__dict__,
                           UTFSP_problem_1.problem_data.mx_walk)

# %% Post optimisation analysis:
def objective_function_analysis(UTFSP_problem_1, F_x):
    return(
    f"{round(gf2.f3_ETT(UTFSP_problem_1.R_routes.routes,F_x, UTFSP_problem_1.problem_data.mx_dist, UTFSP_problem_1.problem_data.mx_demand, UTFSP_problem_1.problem_inputs.__dict__),3)}"\
    f" min & "\
    f"{round(gf2.f3_ETT(UTFSP_problem_1.R_routes.routes,F_x, UTFSP_problem_1.problem_data.mx_dist, UTFSP_problem_1.problem_data.mx_demand, UTFSP_problem_1.problem_inputs.__dict__, UTFSP_problem_1.problem_data.mx_walk),3)}"\
    f" min & "\
    f"{gf2.f4_TBR(UTFSP_problem_1.R_routes.routes, F_x, UTFSP_problem_1.problem_data.mx_dist)}"
    f" buses"
    )

F_x = np.array([1/5, 1/5, 1/7, 1/5, 1/5, 1/5, 1/5, 1/5])
F_x = np.full((1,UTFSP_problem_1.problem_constraints.con_r), 1/30)[0]
F_x = np.array([0.2,	0.2,	0.2,	0.2,	0.0333333333333333,	0.0333333333333333,	0.2,	0.2]) # case study

1/F_x

Route_lengths = gf.calc_seperate_route_length(R_x, mx_dist)  
    
buses_required = gf2.f4_TBR(R_x, F_x, mx_dist, False)

for i in range(len(buses_required)):  
    buses_required[i] = math.ceil(buses_required[i])

""" Print the routes with buses required and the frequencies """
print_str = ""
comma = ","
rowcol_boolean = True
for i in range(len(buses_required)): 
    freq_temp = round(buses_required[i]/(Route_lengths[i]*2),3)
    headway_temp = round(1/(buses_required[i]/(Route_lengths[i]*2)),3) 
    
    temp_str = f"{string.ascii_uppercase[i]} & $\\left\\langle {comma.join([str(x) for x in R_x[i]])}\\right\\rangle$ & {Route_lengths[i]} & {math.ceil(buses_required[i])} & {freq_temp} & {headway_temp}\\\\"
    if rowcol_boolean:
        temp_str = "".join([temp_str, "\\rowcol"])
    rowcol_boolean = not rowcol_boolean
    print_str = "\n".join([print_str, temp_str])

print(print_str)

# Initial performance
print(f"Initial performance:\t {objective_function_analysis(UTFSP_problem_1, F_x)}")

# Actual performance 
F_x_achieved = buses_required/Route_lengths
print(f"Actual performance:\t {objective_function_analysis(UTFSP_problem_1, F_x_achieved)}")


TN = gf2.Transit_network(R_x, F_x, mx_dist, mx_demand, parameters_input, mx_walk) 

mx_volumes_links = TN.mx_volumes_links
b = TN.mx_volumes_nodes

c = TN.volumes_links_per_destination

TN.volumes_nodes_per_destination_i[0]

d = TN.df_opt_strat_alg_named[1]


# %% Links Analysis for UTFSP
TN = gf2.Transit_network(R_x, F_x_achieved, mx_dist, mx_demand, parameters_input, mx_walk) 

def print_links_analysis(TN):
    links_analysis_df = pd.DataFrame(columns = ["v_i", "v_j", "c_a","f_a","v_a"])
    
    for v_i in range(len(TN.mx_volumes_links)):
        for v_j in range(len(TN.mx_volumes_links)):
            if TN.mx_volumes_links[v_i,v_j] != 0:
                links_analysis_df.loc[len(links_analysis_df)] = [v_i, v_j, TN.mx_C_a[v_i,v_j], TN.mx_f_a[v_i,v_j], TN.mx_volumes_links[v_i,v_j]]
                
    links_analysis_df_named = copy.deepcopy(links_analysis_df)
    
    for i in range(len(links_analysis_df)):
        links_analysis_df_named.iloc[i,0] = TN.names_all_transit_nodes[int(links_analysis_df.iloc[i,0])]
        links_analysis_df_named.iloc[i,1] = TN.names_all_transit_nodes[int(links_analysis_df.iloc[i,1])] 
    
    
    """ Max usage per link """     
    links_analysis_max_df = pd.DataFrame(columns = ["Route", "Max"])
    
    for index_i in range(len(TN.R_routes_named)):
        temp_storage = 0 
        for link_nr in range(len(links_analysis_df_named)):
            if links_analysis_df_named.iloc[link_nr,0] in TN.R_routes_named[index_i] or links_analysis_df_named.iloc[link_nr,1] in TN.R_routes_named[index_i]:
                if links_analysis_df_named.iloc[link_nr,4] > temp_storage:
                    temp_storage = links_analysis_df_named.iloc[link_nr,4]
        
        links_analysis_max_df.loc[len(links_analysis_max_df)] = [string.ascii_uppercase[index_i], temp_storage]
    
    """ Walk usage for links """     
    walk_analysis_max_df = pd.DataFrame(columns = ["v_i", "v_i", "Usage"])
    
    for link_nr in range(len(links_analysis_df_named)):
        if links_analysis_df_named.iloc[link_nr,0] in TN.names_all_transit_nodes[0:len(TN.mapping_adjacent)] and links_analysis_df_named.iloc[link_nr,1] in TN.names_all_transit_nodes[0:len(TN.mapping_adjacent)]:
            walk_analysis_max_df.loc[len(walk_analysis_max_df)] = [links_analysis_df_named.iloc[link_nr,0], links_analysis_df_named.iloc[link_nr,1], links_analysis_df_named.iloc[link_nr,4]]
    
    def inf_conversion(number_to_test): 
        if number_to_test == inf:
            return "\infty"
        else:   
            return round(number_to_test,3)
    
    """  Print links analysis for latex """
    print_str = ""
    rowcol_boolean = True
    for i in range(len(links_analysis_df_named)):   
        temp_str = f"{links_analysis_df_named.iloc[i,0]} & {links_analysis_df_named.iloc[i,1]} & {links_analysis_df_named.iloc[i,2]} & ${inf_conversion(links_analysis_df_named.iloc[i,3])}$ & {round(links_analysis_df_named.iloc[i,4],3)}\\\\"
        if rowcol_boolean:
            temp_str = "".join([temp_str, "\\rowcol"])
        rowcol_boolean = not rowcol_boolean
        
        print_str = "\n".join([print_str, temp_str])
    
    print("\nLinks analysis:")
    print(print_str)
    
    
    """  Print max route usage for latex """
    print_str = ""
    rowcol_boolean = True
    for i in range(len(links_analysis_max_df)):   
        temp_str = f"{links_analysis_max_df.iloc[i,0]} & $\\left\\langle {comma.join([str(x) for x in TN.R_routes[i]])}\\right\\rangle$ & {round(links_analysis_max_df.iloc[i,1],3)}\\\\"
        if rowcol_boolean:
            temp_str = "".join([temp_str, "\\rowcol"])
        rowcol_boolean = not rowcol_boolean
        
        print_str = "\n".join([print_str, temp_str])
   
    print("\nRoute usage:")
    print(print_str)
    
    """  Print walk links usage for latex """
    print_str = ""
    rowcol_boolean = True
    for i in range(len(walk_analysis_max_df)):   
        temp_str = f"{walk_analysis_max_df.iloc[i,0]} & {walk_analysis_max_df.iloc[i,1]} & {round(walk_analysis_max_df.iloc[i,2],3)}\\\\"
        if rowcol_boolean:
            temp_str = "".join([temp_str, "\\rowcol"])
        rowcol_boolean = not rowcol_boolean
        
        print_str = "\n".join([print_str, temp_str])
        
    print("\nWalk link usage:")
    print(print_str)
    
print_links_analysis(TN)

# %% Final performance
buses_required_final = copy.deepcopy(buses_required)
F_x_achieved = buses_required_final/Route_lengths
F_x_achieved[4] = 1/100000
F_x_achieved[5] = 1/100000
objective_function_analysis(UTFSP_problem_1, F_x_achieved)

# %% Matie Bus Evaluations
UTFSP_problem_MB = copy.deepcopy(UTFSP_problem_1)
R_x_MB = gf.convert_routes_str2list("0-7*8-3*8-9*")
UTFSP_problem_MB.R_routes.routes = R_x_MB
F_x_MB = np.array([1/3, 1/3, 1/3])
print(f"\nMB performance:\n {objective_function_analysis(UTFSP_problem_MB, F_x_MB)}")

parameters_input['wt'] = 0 # waiting time [min] for UTRP evaluation
parameters_input['tp'] = 5 # transfer penalty [min] for UTRP evaluation

# UTRP Evaluations
objs = ev.evalObjs(R_x_MB,mx_dist,mx_demand,parameters_input)
evaluation = ev.fullPassengerEvaluation(R_x_MB, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
print(f'{round(objs[0], 6)} & {round(objs[1], 2)} & {round(evaluation[1], 2)} & {round(evaluation[2], 2)} & {round(evaluation[3], 2)} & {round(evaluation[4], 2)}')

d_0 = 0
d_1 = 0
d_2 = 0
d_u = 0
tot_demand = sum(sum(mx_demand))

for route in R_x_MB:
    for v_i in range(len(route)):
        for v_j in range(len(route)):
            if v_i != v_j:
                d_0 = d_0 + mx_demand[v_i, v_j]

d_1 = d_1 + mx_demand[3, 9]
d_1 = d_1 + mx_demand[9, 3]

d_u = tot_demand - d_0 - d_1
print(f"d_0: {d_0} \t d_1: {d_1} \t d_2: {d_2} d_u: {d_u}")

print(f"d_0: {round(d_0/tot_demand,4)} \t d_1: {round(d_1/tot_demand,4)} \t d_2: {round(d_2/tot_demand,4)} d_u: {round(d_u/tot_demand,4)}")

# UTFSP Links analysis
TN_MB = gf2.Transit_network(R_x_MB, F_x_MB, mx_dist, mx_demand, parameters_input, mx_walk) 
print_links_analysis(TN_MB)


# Post calculations
Route_lengths_MB = gf.calc_seperate_route_length(R_x_MB, mx_dist) 
buses_required_MB = gf2.f4_TBR(R_x_MB, F_x_MB, mx_dist, False)

for i in range(len(buses_required_MB)):  
    buses_required_MB[i] = math.ceil(buses_required_MB[i])
    
F_x_achieved_MB = buses_required_MB/Route_lengths_MB
objective_function_analysis(UTFSP_problem_MB, F_x_achieved_MB)


def print_routes_and_frequencies_latex_table(R_x, F_x, mx_dist):
    Route_lengths = gf.calc_seperate_route_length(R_x, mx_dist)  
        
    buses_required = gf2.f4_TBR(R_x, F_x, mx_dist, False)
    
    for i in range(len(buses_required)):  
        buses_required[i] = math.ceil(buses_required[i])
    
    """ Print the routes with buses required and the frequencies """
    print_str = ""
    comma = ","
    rowcol_boolean = True
    for i in range(len(buses_required)): 
        freq_temp = round(buses_required[i]/(Route_lengths[i]*2),3)
        headway_temp = round(1/(buses_required[i]/(Route_lengths[i]*2)),3) 
        
        temp_str = f"{string.ascii_uppercase[i]} & $\\left\\langle {comma.join([str(x) for x in R_x[i]])}\\right\\rangle$ & {Route_lengths[i]} & {math.ceil(buses_required[i])} & {freq_temp} & {headway_temp}\\\\"
        if rowcol_boolean:
            temp_str = "".join([temp_str, "\\rowcol"])
        rowcol_boolean = not rowcol_boolean
        print_str = "\n".join([print_str, temp_str])
    
    print(print_str)
    
print_routes_and_frequencies_latex_table(R_x_MB, F_x_MB, mx_dist)
