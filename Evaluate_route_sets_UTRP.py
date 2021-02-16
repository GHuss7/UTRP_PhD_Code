# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 09:14:03 2020

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
import DSS_UTNDP_Classes as gc
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import EvaluateRouteSet as ev

# %% Load the respective files
name_input_data = "Mandl_Data"      # set the name of the input data
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)
del name_input_data
# %% Set input parameters
Choice_generate_initial_set = True 
Choice_print_results = True 
Choice_conduct_sensitivity_analysis = True 
  
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
'Problem_name' : "Mandl", # Specify the name of the problem currently being addresses
'ref_point_max_f1_ATT' : 15, # max f1_ATT for the Hypervolume calculations
'ref_point_min_f1_ATT' : 10, # min f1_ATT for the Hypervolume calculations
'ref_point_max_f2_TRT' : 224, # max f2_TRT for the Hypervolume calculations
'ref_point_min_f2_TRT' : 63, # min f2_TRT for the Hypervolume calculations
'walkFactor' : 3, # factor it takes longer to walk than to drive
'boardingTime' : 0.1, # assume boarding and alighting time = 6 seconds
'alightingTime' : 0.1, # problem when alighting time = 0 (good test 0.5)(0.1 also works)
'large_dist' : int(mx_dist.max()), # the large number from the distance matrix
'alpha_const_inter' : 0.5 # constant for interarrival times relationship 0.5 (Spiess 1989)
}

'''State the various GA input parameters for frequency setting''' 
parameters_GA_route_design={
"method" : "GA",
"population_size" : 200, #should be an even number STANDARD: 200 (John 2016)
"generations" : 200, # STANDARD: 200 (John 2016)
"number_of_runs" : 20, # STANDARD: 20 (John 2016)
"crossover_probability" : 1.0, 
"crossover_distribution_index" : 5,
"mutation_probability" : 1/parameters_constraints["con_r"], # John: 1/|Route set| -> set later
"mutation_distribution_index" : 10,
"tournament_size" : 2,
"termination_criterion" : "StoppingByEvaluations",
"max_evaluations" : 25000,
"number_of_variables" : parameters_constraints["con_r"],
"number_of_objectives" : 2, # this could still be automated in the future
"Number_of_initial_solutions" : 10000 # number of initial solutions to be generated and chosen from
}


'''Set the reference point for the Hypervolume calculations'''
max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])



