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

#route_to_mutate = gf.convert_routes_str2list("13-9-7-5-3-4*0-1-2-5-7-9-10-12-13*6-14-5-2-1-4*6-14-5-3-1-0*6-14-7*")
route_to_mutate = gf.convert_routes_str2list("13-9-7-5-3-4*0-1-2-5-7-9-12-13*8-14-5-2-1-4*6-14-5-3-1-0*6-14-7*")

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

#%% Yen's k-shortest path algorithm
def path_cost(graph, path, weights=None):

    pathcost = 0
    if weights is None:
        pathcost = len(path)-1
    else:
        for i in range(len(path)):
            if i > 0:
                edge = graph.es.find(_source=min(path[i-1], path[i]),
                                     _target=max(path[i-1], path[i]))
                pathcost += edge[weights]

    return pathcost


def in_lists(list1, list2):

    result = False
    node_result = -1

    if len(list1) < len(list2):
        toIter = list1
        toRefer = list2
    else:
        toIter = list2
        toRefer = list1

    for element in toIter:
        result = element in toRefer
        if result:
            node_result = element
            break

    return result, node_result


def yen_igraph(graph, source, target, num_k, weights):
    import queue

    #Shortest path from the source to the target
    A = [graph.get_shortest_paths(source,
                                  to=target,
                                  weights=weights,
                                  output="vpath")[0]]
    A_costs = [path_cost(graph, A[0], weights)]

    #Initialize the heap to store the potential kth shortest path
    B = queue.PriorityQueue()

    for k in range(1, num_k):
        # The spur node ranges from the first node to the next to last node in
        # the shortest path
        for i in range(len(A[k-1])-1):
            #Spur node is retrieved from the previous k-shortest path, k - 1
            spurNode = A[k-1][i]
            # The sequence of nodes from the source to the spur node of the
            # previous k-shortest path
            rootPath = A[k-1][:i]

            #We store the removed edges
            removed_edges = []

            for path in A:
                if len(path) - 1 > i and rootPath == path[:i]:
                    # Remove the links that are part of the previous shortest
                    # paths which share the same root path
                    edge = graph.es.select(_source=min(path[i], path[i+1]),
                                           _target=max(path[i], path[i+1]))
                    if len(edge) == 0:
                        continue
                    edge = edge[0]
                    removed_edges.append((path[i],
                                     path[i+1],
                                     edge.attributes()))
                    edge.delete()

            #Calculate the spur path from the spur node to the sink
            while True:
                spurPath = graph.get_shortest_paths(spurNode,
                                                to=target,
                                                weights=weights,
                                                output="vpath")[0]
                [is_loop, loop_element] = in_lists(spurPath, rootPath)

                if not is_loop:
                    break
                else:
                    loop_index = spurPath.index(loop_element)
                    edge = graph.es.select(_source=min(spurPath[loop_index],
                                                       spurPath[loop_index-1]),
                                           _target=max(spurPath[loop_index],
                                                       spurPath[loop_index-1]))

                    if len(edge) == 0:
                        continue

                    edge = edge[0]
                    removed_edges.append((spurPath[loop_index],
                                         spurPath[loop_index-1],
                                         edge.attributes()))
                    edge.delete()

            #Add back the edges that were removed from the graph
            for removed_edge in removed_edges:
                node_start, node_end, cost = removed_edge
                graph.add_edge(node_start, node_end)
                edge = graph.es.select(_source=min(node_start, node_end),
                                   _target=max(node_start, node_end))[0]
                edge.update_attributes(cost)

            if len(spurPath) > 0:
                #Entire path is made up of the root path and spur path
                totalPath = rootPath + spurPath
                totalPathCost = path_cost(graph, totalPath, weights)
                #Add the potential k-shortest path to the heap
                B.put((totalPathCost, totalPath))

        #Sort the potential k-shortest paths by cost
        #B is already sorted
        #Add the lowest cost path becomes the k-shortest path.
        while True:
            if B.qsize() == 0:
                break
            cost_, path_ = B.get()
            if path_ not in A:
                #We found a new path to add
                A.append(path_)
                A_costs.append(cost_)
                break

        if not len(A) > k:
            break

    return A, A_costs


# Implementation
g_tn = gf.create_igraph_from_dist_mx(mx_dist,edge_name="weights")

A, A_costs = yen_igraph(g_tn, 0, 14, num_k=6, weights="weights") #weights=g_tn.es["weights"]

# -*- coding: utf-8 -*-
"""
A NetworkX based implementation of Yen's algorithm for computing K-shortest paths.   
Yen's algorithm computes single-source K-shortest loopless paths for a 
graph with non-negative edge cost. For more details, see: 
http://en.m.wikipedia.org/wiki/Yen%27s_algorithm
"""
__author__ = 'Guilherme Maia <guilhermemm@gmail.com>'

__all__ = ['k_shortest_paths']

from heapq import heappush, heappop
from itertools import count

import networkx as nx

def k_shortest_paths(G, source, target, k=1, weight='weight'):
    """Returns the k-shortest paths from source to target in a weighted graph G.
    Parameters
    ----------
    G : NetworkX graph
    source : node
       Starting node
    target : node
       Ending node
       
    k : integer, optional (default=1)
        The number of shortest paths to find
    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight
    Returns
    -------
    lengths, paths : lists
       Returns a tuple with two lists.
       The first list stores the length of each k-shortest path.
       The second list stores each k-shortest path.  
    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.
    Examples
    --------
    >>> G=nx.complete_graph(5)    
    >>> print(k_shortest_paths(G, 0, 4, 4))
    ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])
    Notes
    ------
    Edge weight attributes must be numerical and non-negative.
    Distances are calculated as sums of weighted edges traversed.
    """
    if source == target:
        return ([0], [[source]]) 
       
    length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
    if target not in path: #TODO: changed length to path
        raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))
        
    lengths = [length[target]]
    paths = [path[target]]
    c = count()        
    B = []                        
    G_original = G.copy()    
    
    for i in range(1, k):
        for j in range(len(paths[-1]) - 1):            
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]
            
            edges_removed = []
            for c_path in paths:
                if len(c_path) > j and root_path == c_path[:j + 1]:
                    u = c_path[j]
                    v = c_path[j + 1]
                    if G.has_edge(u, v):
                        edge_attr = G.edge[u][v]
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))
            
            for n in range(len(root_path) - 1):
                node = root_path[n]
                # out-edges
                for u, v, edge_attr in G.edges_iter(node, data=True):
                    G.remove_edge(u, v)
                    edges_removed.append((u, v, edge_attr))
                
                if G.is_directed():
                    # in-edges
                    for u, v, edge_attr in G.in_edges_iter(node, data=True):
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))
            
            spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)            
            if target in spur_path and spur_path[target]:
                total_path = root_path[:-1] + spur_path[target]
                total_path_length = get_path_length(G_original, root_path, weight) + spur_path_length[target]                
                heappush(B, (total_path_length, next(c), total_path))
                
            for e in edges_removed:
                u, v, edge_attr = e
                G.add_edge(u, v, edge_attr)
                       
        if B:
            (l, _, p) = heappop(B)        
            lengths.append(l)
            paths.append(p)
        else:
            break
    
    return (lengths, paths)

def get_path_length(G, path, weight='weight'):
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            
            length += G.get_edge_data(u,v).get(weight)
    
    return length


g_tn_nx = gf.create_nx_graph_from_adj_matrix(mx_dist)
nx_adj_mx = copy.deepcopy(mx_dist)
for i in range(len(nx_adj_mx)):   
    for j in range(len(nx_adj_mx)):
        if nx_adj_mx[i,j] == np.max(nx_adj_mx): 
            nx_adj_mx[i,j] = 0
G = nx.from_numpy_matrix(np.asarray(nx_adj_mx))
k_shortest_paths(G, 0, 1, k=4, weight='weight')

G.get_edge_data(0,1)
nx.single_source_dijkstra(G, 0, 14, weight='weight')

# https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

for path in nx.all_simple_paths(G, 0, 14, cutoff=10):
    print(path)
