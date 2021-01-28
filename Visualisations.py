# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 07:52:33 2020

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

# %% Load the respective files
name_input_data = "Mandl_Data"      # set the name of the input data
name_input_data = "SSML_STB_1200_UTFSP"      # set the name of the input data
name_input_data = "Mumford0"
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)

# Load the respective dictionaries for the instance
if False:
    parameters_constraints = json.load(open("./Input_Data/"+name_input_data+"/parameters_constraints.json"))
    parameters_input = json.load(open("./Input_Data/"+name_input_data+"/parameters_input.json"))
    parameters_GA_frequencies = json.load(open("./Input_Data/"+name_input_data+"/parameters_GA_frequencies.json"))

else:
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
    'Problem_name' : name_input_data, # Specify the name of the problem currently being addresses
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
    "min_accepts" : 10, # minimum number of accepted moves per epoch; Danie PhD (pg. 98): Dreo et al. chose 12N (N being some d.o.f.)
    "max_attempts" : 3, # maximum number of attempted moves per epoch
    "max_reheating_times" : 5, # the maximum number of times that reheating can take place
    "max_poor_epochs" : 400, # maximum number of epochs which may pass without the acceptance of any new solution
    "Temp" : 10,  # starting temperature and a geometric cooling schedule is used on it # M = 1000 gives 93.249866 from 20 runs
    "M_iterations_for_temp" : 1000, # the number of initial iterations to establish initial starting temperature
    "Cooling_rate" : 0.97, # the geometric cooling rate 0.97 has been doing good, but M =1000 gives 0.996168
    "Reheating_rate" : 1.05, # the geometric reheating rate
    "number_of_initial_solutions" : 2, # sets the number of initial solutions to generate as starting position
    "Feasibility_repair_attempts" : 2, # the max number of edges that will be added and/or removed to try and repair the route feasibility
    "number_of_runs" : 1, # number of runs to complete John 2016 set 20
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


# %% Route visualisations
'''Graph visualisation'''

for entry in mx_coords: # loop to transpose the coords over x-axis
    entry[1] = -1 * entry[1]

''' TRANSIT ROUTE NETWORK '''
name = 'UTRP_TRANSIT_NETWORK'
routes_R = "*" 
routes_R = gf.convert_routes_str2list(routes_R)
gv.plotRouteSetAndSavePDF_road_network(mx_dist, routes_R, mx_coords, name)

gv.plot_igraph_from_dist_mx(mx_dist)

if name_input_data == "Mandl_Data":

    ''' DBMOSA UTRP ATT MIN ROUTE '''
    name = 'UTRP_DBMOSA_ATT_MIN'
    routes_R = "0-1-4-3-5-14-6-9-13*0-1-2-5-7-9-13-12-10-11*0-1-2-5-7-14-6-9-10-12*8-14-6-9-10-11-3-1-0*0-1-2-5-14-8*2-1-4-3-5-7-9-10-12*" 
    routes_R = gf.convert_routes_str2list(routes_R)
    gv.plotRouteSetAndSavePDF(mx_dist, routes_R, mx_coords, name)
    objs = ev.evalObjs(routes_R,mx_dist,mx_demand,parameters_input)
    evaluation = ev.fullPassengerEvaluation(routes_R, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
    # OUTPUT: return(ATT,d0,d1,d2,drest,noRoutes,longest,shortest)
    print(f'{name}: {round(objs[0], 6)} & {round(objs[1], 2)} & {round(evaluation[1], 2)} & {round(evaluation[2], 2)} & {round(evaluation[3], 2)} & {round(evaluation[4], 2)}')
    
    ''' DBMOSA UTRP TRT MIN ROUTE '''
    name = 'UTRP_DBMOSA_TRT_MIN'
    routes_R = "12-10-9-6-14-7-5-2-1*0-1*1-3-4*8-14*11-10*13-12*" 
    routes_R = gf.convert_routes_str2list(routes_R)
    gv.plotRouteSetAndSavePDF(mx_dist, routes_R, mx_coords, name)
    objs = ev.evalObjs(routes_R,mx_dist,mx_demand,parameters_input)
    evaluation = ev.fullPassengerEvaluation(routes_R, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
    print(f'{name}: {round(objs[0], 6)} & {round(objs[1], 2)} & {round(evaluation[1], 2)} & {round(evaluation[2], 2)} & {round(evaluation[3], 2)} & {round(evaluation[4], 2)}')
    
    ''' NSGA II UTRP ATT MIN ROUTE '''
    name = 'UTRP_NSGAII_ATT_MIN'
    routes_R = "12-13-9-6-14-5-2-1-0*0-1-3-11-10-12*11-10-9-6-14-8*0-1-4-3-5-7-14-6*10-9-7-5-3-4*0-1-2-5-7-9-10-12-13*" 
    routes_R = gf.convert_routes_str2list(routes_R)
    gv.plotRouteSetAndSavePDF(mx_dist, routes_R, mx_coords, name)
    objs = ev.evalObjs(routes_R,mx_dist,mx_demand,parameters_input)
    evaluation = ev.fullPassengerEvaluation(routes_R, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
    print(f'{name}: {round(objs[0], 6)} & {round(objs[1], 2)} & {round(evaluation[1], 2)} & {round(evaluation[2], 2)} & {round(evaluation[3], 2)} & {round(evaluation[4], 2)}')
    
    ''' NSGA II UTRP TRT MIN ROUTE '''
    name = 'UTRP_NSGAII_TRT_MIN'
    routes_R = "10-11*3-1-2-5-7-14-6-9-10-12*13-12*0-1*14-8*3-4*" 
    routes_R = gf.convert_routes_str2list(routes_R)
    gv.plotRouteSetAndSavePDF(mx_dist, routes_R, mx_coords, name)
    objs = ev.evalObjs(routes_R,mx_dist,mx_demand,parameters_input)
    evaluation = ev.fullPassengerEvaluation(routes_R, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
    print(f'{name}: {round(objs[0], 6)} & {round(objs[1], 2)} & {round(evaluation[1], 2)} & {round(evaluation[2], 2)} & {round(evaluation[3], 2)} & {round(evaluation[4], 2)}')
    
    
    # %% Extra
    print("John 2016, best operator route set:")
    routes_R = "4-3-1*13-12*8-14*9-10-12*9-6-14-7-5-2-1-0*10-11*"
    routes_R = gf.convert_routes_str2list(routes_R)
    gv.plotRouteSetAndSavePDF(mx_dist, routes_R, mx_coords, "John_2016_best_operator_obj")
    objs = ev.evalObjs(routes_R,mx_dist,mx_demand,parameters_input)
    evaluation = ev.fullPassengerEvaluation(routes_R, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
    print(f'{round(objs[0], 6)} & {round(objs[1], 2)} & {round(evaluation[1], 2)} & {round(evaluation[2], 2)} & {round(evaluation[3], 2)} & {round(evaluation[4], 2)}')
    
    print("John 2016, best passenger route set:")
    routes_R = "10-9-7-5-3-4-1-0*9-13-12-10-11-3-1-0*5-3-11-10-9-6-14-8*6-14-7-5-3-4-1-2*12-10-9-7-5-2-1-0*0-1-2-5-14-6-9-7*"
    routes_R = gf.convert_routes_str2list(routes_R)
    gv.plotRouteSetAndSavePDF(mx_dist, routes_R, mx_coords, "John_2016_best_passenger_obj")
    objs = ev.evalObjs(routes_R,mx_dist,mx_demand,parameters_input)
    evaluation = ev.fullPassengerEvaluation(routes_R, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
    print(f'{round(objs[0], 6)} & {round(objs[1], 2)} & {round(evaluation[1], 2)} & {round(evaluation[2], 2)} & {round(evaluation[3], 2)} & {round(evaluation[4], 2)}')


#%% Case study evaluation
if name_input_data == "SSML_STB_1200_UTFSP":
    print("Case study chosen route set:")
    routes_R = "5-7-2-8-3-9*1-7*7-6*4-8-5*7-0*7-8*1-0*2-1-0-6-8-5*"
    routes_R = gf.convert_routes_str2list(routes_R)
    gv.plotRouteSetAndSavePDF(mx_dist, routes_R, mx_coords, "Case_study_chosen_route_set")
    objs = ev.evalObjs(routes_R,mx_dist,mx_demand,parameters_input)
    evaluation = ev.fullPassengerEvaluation(routes_R, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
    print(f'{round(objs[0], 6)} & {round(objs[1], 2)} & {round(evaluation[1], 2)} & {round(evaluation[2], 2)} & {round(evaluation[3], 2)} & {round(evaluation[4], 2)}')


#%% Extra evaluations
if name_input_data == 'Mumford0':
    print("Mumford0 Route set:")
    routes_R = "3-1*19-12*2-29-27-16-28-17-19-12-8*8-12-0-13-6-10-2-29-27-7-14-23-1-3*15-10*3-9*4-24-14-7-16-6-5-21*11-25-28-16-15-21*9-1-23-14-11-17-22-0-26*19-18-0-25-7-20-24-3*10-21*10-15*"
    routes_R = gf.convert_routes_str2list(routes_R)
    #R_1 = gc.Routes(routes_R)
    #R_1.plot_routes_no_coords(UTNDP_problem_1,layout_style="kk")    
    
    gv.plotRouteSetAndSavePDF(UTNDP_problem_1.problem_data.mx_dist, routes_R, mx_coords, "Mumford0_attempt")
    objs = ev.evalObjs(routes_R,mx_dist,mx_demand,parameters_input)
    evaluation = ev.fullPassengerEvaluation(routes_R, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
    print(f'{round(objs[0], 6)} & {round(objs[1], 2)} & {round(evaluation[1], 2)} & {round(evaluation[2], 2)} & {round(evaluation[3], 2)} & {round(evaluation[4], 2)}')
    
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
    gv.format_igraph_custom_experiment(g_tn)    
    gv.add_route_edges_to_igraph(g_tn, routes_R)
    ig.plot(g_tn, f"Plots/{name}_plot.pdf", inline=False, layout=mx_coords)  #
    
    g_tn.es["curved"] = [0.15]
    
    g_tn.es[0].attributes()
    
    g_tn.vs[0]
    
    g_tn.get_eids
