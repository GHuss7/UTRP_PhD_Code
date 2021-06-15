# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:05:03 2021

@author: gunth
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

# import cupy

import DSS_UTNDP_Functions as gf
import EvaluateRouteSet as ev


# %% Load the respective files
name_input_data = ["Mandl_UTRP", #0
                   "Mumford0_UTRP", #1
                   "Mumford1_UTRP", #2
                   "Mumford2_UTRP", #3
                   "Mumford3_UTRP",][0]   # set the name of the input data


# %% Set input parameters
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

# %% Load the respective files
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)

routes = 

#%% Define the Objective UTNDP functions
    def fn_obj(routes, mx_dist, mx_demand, parameters_input):
        return (ev.evalObjs(routes, 
                mx_dist, 
                mx_demand, 
                parameters_input)) # returns (f1_ATT, f2_TRT)
