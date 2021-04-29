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
# %% Load the respective files
name_input_data = ["Mandl_UTRP", #0
                   "Mumford0_UTRP", #1
                   "Mumford1_UTRP", #2
                   "Mumford2_UTRP", #3
                   "Mumford3_UTRP",][3]   # set the name of the input data
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)
# del name_input_data

if True:
    parameters_constraints = json.load(open("./Input_Data/"+name_input_data+"/parameters_constraints.json"))
    parameters_input = json.load(open("./Input_Data/"+name_input_data+"/parameters_input.json"))
    parameters_GA = json.load(open("./Input_Data/"+name_input_data+"/parameters_GA.json"))
    if not os.path.exists("./Input_Data/"+name_input_data+"/K_shortest_paths.csv"): 
        print("Creating k_shortest paths and saving csv file...")
        #df_k_shortest_paths = gf.create_k_shortest_paths_df(mx_dist, mx_demand, parameters_constraints["con_maxNodes"])
        #df_k_shortest_paths.to_csv("./Input_Data/"+name_input_data+"/K_shortest_paths_prelim.csv")
        df_k_shortest_paths = False
    else:
        df_k_shortest_paths = pd.read_csv("./Input_Data/"+name_input_data+"/K_shortest_paths.csv")

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
#UTNDP_problem_1.problem_SA_parameters = gc.Problem_metaheuristic_inputs(parameters_SA_routes)
#UTNDP_problem_1.k_short_paths = gc.K_shortest_paths(df_k_shortest_paths)
UTNDP_problem_1.mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes
UTNDP_problem_1.max_objs = max_objs
UTNDP_problem_1.min_objs = min_objs
UTNDP_problem_1.add_text = "" # define the additional text for the file name
# UTNDP_problem_1.R_routes = R_routes


# %% Generate_initial_feasible_route_set_test
if False:
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


# %% K shortest path tests
g_tn_nx = gf.create_nx_graph_from_adj_matrix(mx_dist)
nx_adj_mx = copy.deepcopy(mx_dist)
for i in range(len(nx_adj_mx)):   
    for j in range(len(nx_adj_mx)):
        if nx_adj_mx[i,j] == np.max(nx_adj_mx): 
            nx_adj_mx[i,j] = 0
del i, j
G = nx.from_numpy_matrix(np.asarray(nx_adj_mx))

#df_k_shortest_paths_prelim = gf.create_k_shortest_paths_df(mx_dist, mx_demand, 10)
#df_k_shortest_paths_prelim.to_csv("./Input_Data/"+name_input_data+"/K_shortest_paths_prelim.csv")

#df_k_shortest_paths = pd.read_csv("./Input_Data/"+name_input_data+"/K_shortest_paths.csv")
if False:
    k_short_paths = gc.K_shortest_paths(df_k_shortest_paths)
    k_short_paths.create_paths_bool(len(UTNDP_problem_1.mapping_adjacent))
    
    k_shortest_paths_all = k_short_paths.paths

#%% Yens KSP
# dependencies for our dijkstra's implementation
from queue import PriorityQueue
from math import inf
# graph dependency  
import networkx as nx

__author__ = 'blkrt'

def backtrace2(parent, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def dijkstra2(graph, source, target, return_prev_and_dist=True, print_progress=False):
    queue = []
    visited = {}
    distance = {}
    shortest_distance = {}
    parent = {}

    for node in range(len(graph)):
        distance[node] = None
        visited[node] = False
        parent[node] = None
        shortest_distance[node] = float("inf")

    queue.append(source)
    distance[source] = 0
    while len(queue) != 0:
        current = queue.pop(0)
        visited[current] = True
        if current == target:
            if print_progress: print(backtrace2(parent, source, target))
            #break
        for neighbor in graph[current]:
            if visited[neighbor] == False:
                distance[neighbor] = distance[current] + 1
                if distance[neighbor] < shortest_distance[neighbor]:
                    shortest_distance[neighbor] = distance[neighbor]
                    parent[neighbor] = current
                    queue.append(neighbor)
    
    if print_progress:
        print(distance)
        print(shortest_distance)
        print(parent)
        print(target)
    
    path_to_return = backtrace2(parent, source, target)
    if return_prev_and_dist:
        return distance, parent, path_to_return
    else:
        return path_to_return, distance[target]

"""Dijkstra's shortest path algorithm"""
def dijkstra(graph: nx.classes.graph.Graph, start: str, end: str, return_prev_and_dist=True, print_progress=False) -> list:
    """Get the shortest path of nodes by going backwards through prev list
    credits: https://github.com/blkrt/dijkstra-python/blob/3dfeaa789e013567cd1d55c9a4db659309dea7a5/dijkstra.py#L5-L10"""
    def backtrace(prev, start, end):
        node = end
        path = []
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(node) 
        path.reverse()
        return path
        
    """get the cost of edges from node -> node
    cost(u,v) = edge_weight(u,v)"""
    def get_cost(u, v):
        return graph.get_edge_data(u,v).get('weight')
        
    """main algorithm"""
    # predecessor of current node on shortest path 
    prev = {} 
    # initialize distances from start -> given node i.e. dist[node] = dist(start, node)
    dist = {v: inf for v in list(nx.nodes(graph))} 
    # nodes we've visited
    visited = set() 
    #visited.add(start) #TODO
    # prioritize nodes from start -> node with the shortest distance!
    ## elements stored as tuples (distance, node) 
    pq = PriorityQueue()  
    
    dist[start] = 0  # dist from start -> start is zero
    pq.put((dist[start], start))
    
    while 0 != pq.qsize():
        curr_cost, curr = pq.get()
        visited.add(curr)
        if print_progress: print(f'visiting {curr}')
        # look at curr's adjacent nodes
        neighbors = dict(graph.adjacency()).get(curr)
        if bool(neighbors):
            for neighbor in neighbors:
                # if we found a shorter path 
                path = dist[curr] + get_cost(curr, neighbor)
                if path < dist[neighbor]:
                    # update the distance, we found a shorter one!
                    dist[neighbor] = path
                    # update the previous node to be prev on new shortest path
                    prev[neighbor] = curr
                    # if we haven't visited the neighbor
                    if neighbor not in visited:
                        # insert into priority queue and mark as visited
                        visited.add(neighbor)
                        pq.put((dist[neighbor],neighbor))
                    # otherwise update the entry in the priority queue
                    else:
                        #if len(visited) != len(graph.nodes()): # this was added to avoid trying to get something that is not there
                        # insert new
                        pq.put((dist[neighbor],neighbor))
                        # remove old
                        _ = pq.get((dist[neighbor],neighbor))
                            
                        
    if print_progress:
        print("=== Dijkstra's Algo Output ===")
        print("Distances")
        print(dist)
        print("Visited")
        print(visited)
        print("Previous")
        print(prev)
        
    # we are done after every possible path has been checked 
    if len(prev)+1 == len(graph.nodes()):
        path_to_return = backtrace(prev, start, end)
        if return_prev_and_dist:
            return dist, prev, path_to_return
        else:
            return path_to_return, dist[end]
    else:
        if return_prev_and_dist:
            return dist, False, False
        else:
            return False, dist[end]
    
def dijkstra3(graph: nx.classes.graph.Graph, start: str, end: str, return_prev_and_dist=True, print_progress=False) -> list:
    """Get the shortest path of nodes by going backwards through prev list
    credits: https://github.com/blkrt/dijkstra-python/blob/3dfeaa789e013567cd1d55c9a4db659309dea7a5/dijkstra.py#L5-L10"""
    def backtrace(prev, start, end):
        node = end
        path = []
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(node) 
        path.reverse()
        return path
        
    """get the cost of edges from node -> node
    cost(u,v) = edge_weight(u,v)"""
    def get_cost(u, v):
        return graph.get_edge_data(u,v).get('weight')
        
    """main algorithm""" 

    # predecessor of current node on shortest path 
    prev = {} 
    # initialize distances from start -> given node i.e. dist[node] = dist(start, node)
    dist = {v: inf for v in list(nx.nodes(graph))} 
    # nodes we've visited
    visited = set() 
    visited_bool = {}
    shortest_distance = {}

    # prioritize nodes from start -> node with the shortest distance!
    ## elements stored as tuples (distance, node) 
    pq = PriorityQueue()  
    
    for node in range(len(graph)):
        #dist[node] = None
        visited_bool[node] = False
        prev[node] = None
        shortest_distance[node] = float("inf")
    
    dist[start] = 0  # dist from start -> start is zero
    pq.put((dist[start], start))
    
    while 0 != pq.qsize():
        curr_cost, curr = pq.get()
        visited.add(curr)
        visited_bool[curr] = True
        if curr == end:
            if print_progress: print(backtrace2(prev, start, end))
            #break #TODO
        if print_progress: print(f'visiting {curr}')
        # look at curr's adjacent nodes
        neighbors = dict(graph.adjacency()).get(curr)

        for neighbor in neighbors:
            if visited_bool[neighbor] == False:
                path_dist = dist[curr] + get_cost(curr, neighbor)
                
                # if we found a shorter path 
                if path_dist < dist[neighbor]:
                    # update the distance, we found a shorter one!
                    dist[neighbor] = path_dist
                    # update the previous node to be prev on new shortest path
                    prev[neighbor] = curr
                    # if we haven't visited the neighbor
                    if neighbor not in visited:
                        # insert into priority queue and mark as visited
                        visited.add(neighbor)
                        pq.put((dist[neighbor],neighbor)) #TODO TAB
                    # otherwise update the entry in the priority queue
                    else:
                        # if len(visited) != len(graph.nodes()): # this was added to avoid trying to get something that is not there
                        # insert new
                        pq.put((dist[neighbor],neighbor))
                        # remove old
                        _ = pq.get((dist[neighbor],neighbor))


                            
                        
    if print_progress:
        print("=== Dijkstra's Algo Output ===")
        print("Distances")
        print(dist)
        print("Visited")
        print(visited)
        print("Previous")
        print(prev)
        
    # we are done after every possible path has been checked 
    #if len(prev)+1 == len(graph.nodes()):
    path_to_return = backtrace2(prev, start, end)
    if return_prev_and_dist:
        return dist, prev, path_to_return
    else:
        return path_to_return, dist[end]
    # else:
    #     if return_prev_and_dist:
    #         return dist, False, False
    #     else:
    #         return False, dist[end]
 
    
def dijkstra4(graph: nx.classes.graph.Graph, start: str, end: str, return_prev_and_dist=True, print_progress=False) -> list:
    """Get the shortest path of nodes by going backwards through prev list
    credits: https://github.com/blkrt/dijkstra-python/blob/3dfeaa789e013567cd1d55c9a4db659309dea7a5/dijkstra.py#L5-L10"""
    def backtrace(prev, start, end):
        node = end
        path = []
        while node != start:
            path.append(node)
            
            try: node = prev[node]
            except: 
                print(f"Prev: \n{len(prev)} of {len(mx_dist)} \nStart: {start} \nEnd: {end}")
                return False
        path.append(node) 
        path.reverse()
        return path
        
    """get the cost of edges from node -> node
    cost(u,v) = edge_weight(u,v)"""
    def get_cost(u, v):
        return graph.get_edge_data(u,v).get('weight')
        
    """main algorithm"""
    # predecessor of current node on shortest path 
    prev = {} 
    # initialize distances from start -> given node i.e. dist[node] = dist(start, node)
    dist = {v: inf for v in list(nx.nodes(graph))} 
    # nodes we've visited
    visited = set() 
    #visited.add(start) #TODO
    # prioritize nodes from start -> node with the shortest distance!
    ## elements stored as tuples (distance, node) 
    pq = PriorityQueue()  
    
    dist[start] = 0  # dist from start -> start is zero
    pq.put((dist[start], start))
    
    while 0 != pq.qsize():
        curr_cost, curr = pq.get()
        visited.add(curr)
        if print_progress: print(f'visiting {curr}')
        # look at curr's adjacent nodes
        neighbors = dict(graph.adjacency()).get(curr)
        if bool(neighbors):
            for neighbor in neighbors:
                # if we found a shorter path 
                path = dist[curr] + get_cost(curr, neighbor)
                if path < dist[neighbor]:
                    # update the distance, we found a shorter one!
                    dist[neighbor] = path
                    # update the previous node to be prev on new shortest path
                    prev[neighbor] = curr
                    # if we haven't visited the neighbor
                    if neighbor not in visited:
                        # insert into priority queue and mark as visited
                        visited.add(neighbor)
                        pq.put((dist[neighbor],neighbor))
                    # otherwise update the entry in the priority queue
                    else:
                        #if len(visited) != len(graph.nodes()): # this was added to avoid trying to get something that is not there
                        # insert new
                        pq.put((dist[neighbor],neighbor))
                        # remove old
                        _ = pq.get((dist[neighbor],neighbor))
                            
                        
    if print_progress:
        print("=== Dijkstra's Algo Output ===")
        print("Distances")
        print(dist)
        print("Visited")
        print(visited)
        print("Previous")
        print(prev)
        
    # we are done after every possible path has been checked 
    #if len(prev)+1 == len(graph.nodes()):
    path_to_return = backtrace(prev, start, end)
    if return_prev_and_dist:
        return dist, prev, path_to_return
    else:
        return path_to_return, dist[end]
    # else:
    #     if return_prev_and_dist:
    #         return dist, False, False
    #     else:
    #         return False, dist[end]
    
def itemgetter(*items):
    if len(items) == 1:
        item = items[0]
        def g(obj):
            return obj[item]
    else:
        def g(obj):
            return tuple(obj[item] for item in items)
    return g

def remove_edge_and_return_cost(graph, u, v):
    try:
        cost = graph.get_edge_data(u,v).get('weight')
        graph.remove_edge(u, v)
        return cost
    except:
        return -1
    

#G_nx = copy.deepcopy(G)
#ksp_yen(graph=G_nx, node_start=0, node_end=5, max_k=2)
#graph=G_nx 
#node_start=10 
#node_end=5 
#max_k=10
#large_weight=10000

def ksp_yen(graph, node_start, node_end, max_k=2, large_weight = 10000):
    """credits https://stackoverflow.com/questions/15878204/k-shortest-paths-implementation-in-igraph-networkx-yens-algorithm"""
    
    distances, previous, path = dijkstra4(graph, node_start, node_end) #nx.single_source_dijkstra(graph, node_start, target, weight=weight)
    
    A = [{'cost': distances[node_end], 
          'path': path}]
    B = []
    
    graph_copy = copy.deepcopy(graph)
    
    if not A[0]['path']: 
        print(A) #return A 
        print(f"Node start: {node_start}, Node end:{node_end}") 
    
    for k in range(1, max_k):
        edges_removed = []
        for i in range(0, len(A[-1]['path']) - 1):
            node_spur = A[-1]['path'][i]
            path_root = A[-1]['path'][:i+1]
    
            #edges_removed = []
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i+1]:
                    #cost = remove_edge_and_return_cost(graph_copy, curr_path[i], curr_path[i+1])
                    #if cost == -1:
                    #    continue
                    #edges_removed.append([curr_path[i], curr_path[i+1], cost])
                    
                    #TODO:
                    edges_removed.append((curr_path[i], curr_path[i+1]))
                    graph_copy.edges[curr_path[i], curr_path[i+1]]['weight'] = large_weight
    
            path_only_spur, dist_only_spur = dijkstra4(graph_copy, node_spur, node_end, return_prev_and_dist=False, print_progress=False)
    
            path_spur = {'cost': dist_only_spur, 
                  'path': path_only_spur}
    
            if path_spur['path']:
                path_total = path_root[:-1] + path_spur['path']
                if len(path_total) == len(set(path_total)):
                    dist_total = distances[node_spur] + path_spur['cost']
                    potential_k = {'cost': dist_total, 'path': path_total}
                    
                    if dist_total < large_weight:  # avoids useless distances             
                        if not (potential_k in B):
                            B.append(potential_k)    
                
            # returns graph to original state
            graph_copy = copy.deepcopy(graph)
    
        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            A.append(B[0])
            B.pop(0)
        else:
            break
    
    return A

def ksp_yen_2(graph, node_start, node_end, max_k=2, large_weight = 10000):
    """credits https://stackoverflow.com/questions/15878204/k-shortest-paths-implementation-in-igraph-networkx-yens-algorithm"""
    
    distances, previous, path = dijkstra4(graph, node_start, node_end) #nx.single_source_dijkstra(graph, node_start, target, weight=weight)
    
    A = [{'cost': distances[node_end], 
          'path': path}]
    B = []
    
    graph_copy = copy.deepcopy(graph)
    
    if not A[0]['path']: 
        print(A) 
        print(f"Node start: {node_start}, Node end:{node_end}") 
        return A
    
    for k in range(1, max_k):
        edges_removed = []
        for i in range(0, len(A[-1]['path']) - 1):
            node_spur = A[-1]['path'][i]
            path_root = A[-1]['path'][:i+1]
    
            #edges_removed = []
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i+1]:
                    #cost = remove_edge_and_return_cost(graph_copy, curr_path[i], curr_path[i+1])
                    #if cost == -1:
                    #    continue
                    #edges_removed.append([curr_path[i], curr_path[i+1], cost])
                    
                    #TODO:
                    edges_removed.append((curr_path[i], curr_path[i+1]))
                    graph_copy.edges[curr_path[i], curr_path[i+1]]['weight'] = large_weight
    
            path_only_spur, dist_only_spur = dijkstra4(graph_copy, node_spur, node_end, return_prev_and_dist=False, print_progress=False)
    
            path_spur = {'cost': dist_only_spur, 
                  'path': path_only_spur}
    
            if path_spur['path']:
                path_total = path_root[:-1] + path_spur['path']
                if len(path_total) == len(set(path_total)):
                    dist_total = distances[node_spur] + path_spur['cost']
                    potential_k = {'cost': dist_total, 'path': path_total}
                    
                    if dist_total < large_weight:  # avoids useless distances             
                        if not (potential_k in B):
                            B.append(potential_k)    
                
            # returns graph to original state
            graph_copy = copy.deepcopy(graph)
    
        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            A.append(B[0])
            B.pop(0)
        else:
            break
    
    return A

#G_nx
#max_k=10
#large_weight=10000

def ksp_yen_all(G_nx, max_k=2, large_weight=10000):
    num_vertices = len(G_nx.nodes())
    
    A_all = []
    
    for vertex_i in range(num_vertices):
        for vertex_j in range(num_vertices):
            if vertex_i != vertex_j:
                print(f"=== Vertex {vertex_i} -> {vertex_j} ==========", end =" ")
                A = ksp_yen(G_nx, node_start=vertex_i, node_end=vertex_j, 
                            max_k=max_k,large_weight=large_weight)
                print(f"KSPs: {len(A)}")
                if vertex_i > vertex_j:
                    for path_A in A:
                       path_A['path'].reverse()
                A_all.extend(A)
    
    return A_all

G_nx=G 
max_k=10 
large_weight=10000
def ksp_yen_all_2(G_nx, max_k=2, large_weight=10000):
    num_vertices = len(G_nx.nodes())
    
    A_all = []
    
    for vertex_i in range(num_vertices):
        for vertex_j in range(num_vertices):
            if vertex_i != vertex_j:
                print(f"=== Vertex {vertex_i} -> {vertex_j} ==========", end =" ")
                A = ksp_yen_2(G_nx, node_start=vertex_i, node_end=vertex_j, 
                            max_k=max_k,large_weight=large_weight)
                print(f"KSPs: {len(A)}")
                A = [path_A for path_A in A if path_A['path']] # deletes False paths
                if vertex_i > vertex_j:
                    for path_A in A:
                        if path_A['path']:
                            path_A['path'].reverse()
                A_all.extend(A)
    
    return A_all

def remove_duplicates_ksp_yen(A_all):
    # remove duplicates
    A_unique = []
    
    for A_all_i in A_all:
        if A_all_i not in A_unique:
                A_unique.append(A_all_i)
                
    return A_unique

"""Tests"""
if False:
    dist, prev, path_to_return = dijkstra(graph, node_start, node_end, print_progress=True)
    A = ksp_yen(G_nx, node_start=45, node_end=0, max_k=10)
    A_all = ksp_yen_all(G_nx, max_k=10)
    A_unique = remove_duplicates_ksp_yen(A_all)

    A = ksp_yen_2(G, node_start=45, node_end=7, max_k=10)


# %% Formatting functions
def get_k_shortest_paths(G, source, target, k_cutoff):
    """Get all the k-shortest paths from a graph in terms of vertex counts for
    a given source and target vertex with a cutoff of k vertices
    Returns the paths and lengths between source and target """

    A = ksp_yen(G, node_start=source, node_end=target, max_k=k_cutoff)
    k_shortest_paths = []
    k_shortest_paths_lengths = []
     
    for i in range(len(A)):
        k_shortest_paths.append(A[i]['path'])
        k_shortest_paths_lengths.append(A[i]['cost'])

    return k_shortest_paths, k_shortest_paths_lengths

def get_all_k_shortest_paths(G, k_cutoff):
    """Get all the k-shortest paths from a graph in terms of vertex counts for
    all vertices with a cutoff of k vertices
    Returns all the paths and lengths"""
   
    A_all = ksp_yen_all_2(G, max_k=k_cutoff)
    A_unique = remove_duplicates_ksp_yen(A_all)

    k_shortest_paths_all = []
    k_shortest_paths_lengths_all = []

    for i in range(len(A_unique)):
        k_shortest_paths_all.append(A_unique[i]['path'])
        k_shortest_paths_lengths_all.append(A_unique[i]['cost'])

    return k_shortest_paths_all, k_shortest_paths_lengths_all

def create_k_shortest_paths_df(mx_dist, mx_demand, k_cutoff): 
    df_k_shortest_paths = pd.DataFrame(columns=["Source", "Target", "Travel_time", "Demand", "Demand_per_minute", "Routes"])

    nx_adj_mx = copy.deepcopy(mx_dist)
    for i in range(len(nx_adj_mx)):   
        for j in range(len(nx_adj_mx)):
            if nx_adj_mx[i,j] == np.max(nx_adj_mx): 
                nx_adj_mx[i,j] = 0
    G = nx.from_numpy_matrix(np.asarray(nx_adj_mx))    

    k_shortest_paths_all, k_shortest_paths_lengths = get_all_k_shortest_paths(G, k_cutoff)
    k_shortest_paths_demand = gf.determine_demand_per_route(k_shortest_paths_all, mx_demand)
    demand_per_minute = np.asarray(k_shortest_paths_demand) / np.asarray(k_shortest_paths_lengths)
    
    for index_i in range(len(k_shortest_paths_all)):
        df_k_shortest_paths.loc[index_i] = [k_shortest_paths_all[index_i][0],
                                             k_shortest_paths_all[index_i][-1],
                                             k_shortest_paths_lengths[index_i],
                                             k_shortest_paths_demand[index_i],
                                             demand_per_minute[index_i],
                                             gf.convert_path_list2str(k_shortest_paths_all[index_i])]
        
        df_k_shortest_paths["Travel_time"] = np.float64(df_k_shortest_paths["Travel_time"].values)
        df_k_shortest_paths["Demand"] = np.float64(df_k_shortest_paths["Demand"].values)    
        
    df_k_shortest_paths = df_k_shortest_paths.sort_values(["Source", "Target", "Demand_per_minute"])
        
    return df_k_shortest_paths

k_cutoff = 20
#def create_k_shortest_paths_df(mx_dist, mx_demand, k_cutoff): 
df_k_shortest_paths = pd.DataFrame(columns=["Source", "Target", "Travel_time", "Demand", "Demand_per_minute", "Routes"])

nx_adj_mx = copy.deepcopy(mx_dist)
for i in range(len(nx_adj_mx)):   
    for j in range(len(nx_adj_mx)):
        if nx_adj_mx[i,j] == np.max(nx_adj_mx): 
            nx_adj_mx[i,j] = 0
G = nx.from_numpy_matrix(np.asarray(nx_adj_mx))    

k_shortest_paths_all, k_shortest_paths_lengths = get_all_k_shortest_paths(G, k_cutoff)
k_shortest_paths_demand = gf.determine_demand_per_route(k_shortest_paths_all, mx_demand)
demand_per_minute = np.asarray(k_shortest_paths_demand) / np.asarray(k_shortest_paths_lengths)

for index_i in range(len(k_shortest_paths_all)):
    df_k_shortest_paths.loc[index_i] = [k_shortest_paths_all[index_i][0],
                                         k_shortest_paths_all[index_i][-1],
                                         k_shortest_paths_lengths[index_i],
                                         k_shortest_paths_demand[index_i],
                                         demand_per_minute[index_i],
                                         gf.convert_path_list2str(k_shortest_paths_all[index_i])]
    
    df_k_shortest_paths["Travel_time"] = np.float64(df_k_shortest_paths["Travel_time"].values)
    df_k_shortest_paths["Demand"] = np.float64(df_k_shortest_paths["Demand"].values)    
    
df_k_shortest_paths = df_k_shortest_paths.sort_values(["Source", "Target", "Demand_per_minute"])
df_k_shortest_paths.to_csv("./Input_Data/"+name_input_data+"/K_shortest_paths_prelim_"+str(k_cutoff)+".csv")
   
#return df_k_shortest_paths

if False:
    k_cutoff=10
    df_k_shortest_paths_prelim = create_k_shortest_paths_df(mx_dist, mx_demand, k_cutoff)
    df_k_shortest_paths_prelim.to_csv("./Input_Data/"+name_input_data+"/K_shortest_paths_prelim_"+str(k_cutoff)+".csv")

#df_k_shortest_paths = pd.read_csv("./Input_Data/"+name_input_data+"/K_shortest_paths.csv")

#%% Mutation tests
if False:    
    # route_to_mutate = gf.convert_routes_str2list("13-9-7-5-3-4*1-3-11-10-9-6-14-8*0-1-2-5-7-9-10-12-13*6-14-5-2-1-4*6-14-5-3-1-0*6-14-7*")
    
    route_to_mutate = gf.convert_routes_str2list("13-9-7-5-3-4*0-1-2-5-7-9-10-12-13*6-14-5-2-1-4*6-14-5-3-1-0*6-14-7*")
    #route_to_mutate = gf.convert_routes_str2list("13-9-7-5-3-4*0-1-2-5-7-9-12-13*8-14-5-2-1-4*6-14-5-3-1-0*6-14-7*")
    
    R_to_mutate = gc.Routes(route_to_mutate)
    R_to_mutate.plot_routes(UTNDP_problem_1)
    
    
    all_nodes = [y for x in route_to_mutate for y in x] # flatten all the elements in route
        
    # Initial test for all nodes present:
    if (len(set(all_nodes)) != n_nodes): # if not true, go on to testing for what nodes are ommited
        missing_nodes = list(set(range(n_nodes)).difference(set(all_nodes))) # find all the missing nodes
    
    indices_of_compatible_routes = []
    for path_index in range(len(k_shortest_paths_all)):
        if set(missing_nodes).issubset(set(k_shortest_paths_all[path_index])):
            indices_of_compatible_routes.append(path_index)
    
    path_to_add = k_shortest_paths_all[random.choice(indices_of_compatible_routes)]
    new_route = copy.deepcopy(route_to_mutate)
    new_route.extend([path_to_add])
    
    R_new = gc.Routes(new_route)
    R_new.plot_routes(UTNDP_problem_1)
    
    def add_path_to_route_set_random(route_to_mutate, UTNDP_problem_1, k_shortest_paths_all):
        
        n_nodes = len(UTNDP_problem_1.mapping_adjacent)    
        all_nodes = [y for x in route_to_mutate for y in x] # flatten all the elements in route
        
        # Initial test for all nodes present:
        if (len(set(all_nodes)) != n_nodes): # if not true, go on to testing for what nodes are ommited
            missing_nodes = list(set(range(n_nodes)).difference(set(all_nodes))) # find all the missing nodes
        
        indices_of_compatible_routes = []
        for path_index in range(len(k_shortest_paths_all)):
            if set(missing_nodes).issubset(set(k_shortest_paths_all[path_index])):
                indices_of_compatible_routes.append(path_index)
        
        path_to_add = k_shortest_paths_all[random.choice(indices_of_compatible_routes)]
        new_route = copy.deepcopy(route_to_mutate)
        new_route.extend([path_to_add])
        
        return new_route
    
    add_path_to_route_set_random(route_to_mutate, UTNDP_problem_1, k_shortest_paths_all)
    
    
    # %% Test another route generation procedure
    new_gen_route = gf.routes_generation_unseen_prob(k_shortest_paths_all, k_shortest_paths_all, UTNDP_problem_1.problem_constraints.con_r)
    R_set_4 = gc.Routes(new_gen_route)
    R_set_4.plot_routes(UTNDP_problem_1)
