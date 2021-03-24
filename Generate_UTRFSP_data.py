# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:08:07 2021

@author: 17832020
"""

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
name_input_data = ["Mandl_UTRFSP_no_walk"][0]  # set the name of the input data
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
"Choice_consider_walk_links" : False,
"Choice_import_dictionaries" : False,
"Choice_print_full_data_for_analysis" : True,
"Set_name" : "Overall_Pareto_set_for_case_study_GA.csv" # the name of the set in the main working folder
}

# Disables walk links
if not(Decisions["Choice_consider_walk_links"]):
    mx_walk = False

# Load the respective input data (dictionaries) for the instance
if Decisions["Choice_import_dictionaries"]:
    parameters_constraints = json.load(open("./Input_Data/"+name_input_data+"/parameters_constraints.json"))
    parameters_input = json.load(open("./Input_Data/"+name_input_data+"/parameters_input.json"))
    parameters_GA = json.load(open("./Input_Data/"+name_input_data+"/parameters_GA.json"))

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
    'alpha_const_inter' : 0.5, # constant for interarrival times relationship 0.5 (Spiess 1989)
    'wt' : 0,
    'tp' : 5
    }
    
    '''State the various GA input parameters for frequency setting''' 
    parameters_GA={
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

R_x = gf.normalise_route_set(R_x)
R_routes = gf2.Routes(R_x)

parameters_GA["mutation_probability"] = 1/(len(R_routes.routes))

# %% Initialise the decision variables
'''Initialise the decision variables'''
F_frequencies = gf2.Frequencies(parameters_constraints['con_r']) 

F_frequencies.set_frequencies(np.full(len(R_x), 1/5)) 
    
F_x = F_frequencies.frequencies

parameters_GA["number_of_variables"] = len(F_x)

#%% Define the UTFSP Problem      
UTRFSP_problem_1 = gf2.UTFSP_problem()
UTRFSP_problem_1.problem_data = gf2.Problem_data(mx_dist, mx_demand, mx_coords, mx_walk)
UTRFSP_problem_1.problem_constraints = gf2.Problem_constraints(parameters_constraints)
UTRFSP_problem_1.problem_inputs = gf2.Problem_inputs(parameters_input)
UTRFSP_problem_1.problem_GA_parameters = gf2.Problem_GA_inputs(parameters_GA)
UTRFSP_problem_1.mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes
UTRFSP_problem_1.R_routes = R_routes
UTRFSP_problem_1.frequency_set = np.array([5,6,7,8,9,10,12,14,16,18,20,25,30])
UTRFSP_problem_1.add_text = "" # define the additional text for the file name

#%% Define the objective functions
def fn_obj_f1_f2(routes, UTNDP_problem_input):
    return (ev.evalObjs(routes, 
            UTNDP_problem_input.problem_data.mx_dist, 
            UTNDP_problem_input.problem_data.mx_demand, 
            UTNDP_problem_input.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)

def fn_obj_f3_f4(frequencies, UTFSP_problem_input):
    return (gf2.f3_ETT(UTFSP_problem_input.R_routes.routes,
                       frequencies, 
                       UTFSP_problem_input.problem_data.mx_dist, 
                       UTFSP_problem_input.problem_data.mx_demand, 
                       UTFSP_problem_input.problem_inputs.__dict__,
                       UTFSP_problem_input.problem_data.mx_walk), #f3_ETT
            gf2.f4_TBR(UTFSP_problem_input.R_routes.routes, 
                       frequencies, 
                           UTFSP_problem_input.problem_data.mx_dist)) #f4_TBR

def fn_objectives(routes, frequencies, UTRFSP_input):
    """Evaluates the four main objectives in the UTRFSP problem

    Args:
        routes (list): The route set
        frequencies (Array of float64): The frequency set
        UTRFSP_input (UTRFSP_problem): The problem instance parameters
    
    Returns:
        f_1 (float): The Average Travel Time objective function value
        f_2 (float): The Total Route Time objective function value
        f_3 (float): The Average Expected Travel Time objective function value
        f_4 (float): The Total Buses Required objective function value
    """
    
    UTRFSP_input.R_routes = gf2.Routes(R_x)
    f_1_f_2 = fn_obj_f1_f2(routes, UTRFSP_input)
    f_3_f_4 = fn_obj_f3_f4(frequencies, UTRFSP_input)
    return [f_1_f_2[0], f_1_f_2[1], f_3_f_4[0], f_3_f_4[1]]


#%% Test 
R_x = gf.convert_routes_str2list("13-14-10-8-15*4-2-3-6*13-11-12*7-15-9*5-4-6-15*3-2-1*")
R_x = gf.normalise_route_set(R_x)
F_x = np.random.randint(0, len(UTRFSP_problem_1.frequency_set), len(R_x))
F_x = 1/UTRFSP_problem_1.frequency_set[F_x]
#F_x = F_x.reshape(len(F_x),1) # if you reshape, the numpy broadcasting will start to activate

columns_list = ["R_x", "F_1", "F_2", "F_3", "F_4"]
columns_list[1:1] = ["f_"+str(x) for x in range(len(R_x))]
data_UTRFSP = pd.DataFrame(columns=columns_list)

t1 = time.time()
time_start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
iterations = 15000

for counter in range(iterations):
    R_x = gc.Routes.return_feasible_route_robust(UTRFSP_problem_1)
    F_x = np.random.randint(0, len(UTRFSP_problem_1.frequency_set), len(R_x))
    F_x = 1/UTRFSP_problem_1.frequency_set[F_x]
    
    
    obj_values = fn_objectives(R_x, F_x, UTRFSP_problem_1)
    data_row = [gf.convert_routes_list2str(R_x)]
    data_row.extend(list(F_x))
    data_row.extend(obj_values)
                                         
    data_UTRFSP.loc[len(data_UTRFSP)] = data_row
    
t2 = time.time()
time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if Decisions["Choice_consider_walk_links"]:
    data_UTRFSP.to_csv("../Data/Data_generation_UTRFSP_walk/Data_UTRFSP_"+time_stamp+".csv")
else:
    data_UTRFSP.to_csv("../Data/Data_generation_UTRFSP_no_walk/Data_UTRFSP_"+time_stamp+".csv")

print(f"Duration for {iterations} iterations: {round(t2-t1,4)}s /t Avg Time: {round((t2-t1)/iterations, 4)}")

    
