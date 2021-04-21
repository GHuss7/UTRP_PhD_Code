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
                   "Mumford3_UTRP",][0]   # set the name of the input data
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
G = nx.from_numpy_matrix(np.asarray(nx_adj_mx))

#df_k_shortest_paths_prelim = gf.create_k_shortest_paths_df(mx_dist, mx_demand, 10)
#df_k_shortest_paths_prelim.to_csv("./Input_Data/"+name_input_data+"/K_shortest_paths_prelim.csv")

#df_k_shortest_paths = pd.read_csv("./Input_Data/"+name_input_data+"/K_shortest_paths.csv")

k_short_paths = gc.K_shortest_paths(df_k_shortest_paths)
k_short_paths.create_paths_bool(len(UTNDP_problem_1.mapping_adjacent))

k_shortest_paths_all = k_short_paths.paths

if False:
    #%% Yen's K-shortest path algorithm
    # -*- coding: utf-8 -*-
    """
    A NetworkX based implementation of Yen's algorithm for computing K-shortest paths.   
    Yen's algorithm computes single-source K-shortest loopless paths for a 
    graph with non-negative edge cost. For more details, see: 
    http://networkx.github.io
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
        if target not in path: # change
            raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))
            
        lengths = [length] # change
        paths = [path] # change
        c = count()        
        B = []                        
        G_original = G.copy()    
        
        for i_cutoff in range(1, k): # change
            G = G_original.copy()    
        
            for j in range(len(paths[-1]) - 1):            
                spur_node = paths[-1][j]
                root_path = paths[-1][:j + 1]
                
                edges_removed = []
                for c_path in paths:
                    if len(c_path) > j and root_path == c_path[:j + 1]:
                        u = c_path[j]
                        v = c_path[j + 1]
                        if G.has_edge(u, v):
                            edge_attr = G.get_edge_data(u,v)
                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))
                
                for n in range(len(root_path) - 1):
                    node = root_path[n]
                    # out-edges
                    for u, v, edge_attr in G_original.edges(data=True): #G.edges_iter(node, data=True)
                        if u == node:    
                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))
                    
                    if G.is_directed():
                        # in-edges
                        for u, v, edge_attr in G.in_edges_iter(node, data=True):
                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))
                
                spur_path_length, spur_path = nx.single_source_dijkstra(G_original, spur_node, target, weight=weight)            
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path # change
                    total_path_length = get_path_length(G_original, root_path, weight) + spur_path_length # change                
                    heappush(B, (total_path_length, next(c), total_path))
                    
                #for e in edges_removed:
                #    u, v, edge_attr = e
                #    G.add_edge(u, v, edge_attr)
                           
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
                
                length += G.get_edge_data(u,v).get(weight)  #Change
                
        return length    
    
    k_shortest_paths(G, 0, 14, k=10, weight='weight')

#%% Yens KSP
# dependencies for our dijkstra's implementation
from queue import PriorityQueue
from math import inf
# graph dependency  
import networkx as nx


"""Dijkstra's shortest path algorithm"""
def dijkstra(graph: nx.classes.graph.Graph, start: str, end: str, return_prev_and_dist=True) -> list:
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
    # prioritize nodes from start -> node with the shortest distance!
    ## elements stored as tuples (distance, node) 
    pq = PriorityQueue()  
    
    dist[start] = 0  # dist from start -> start is zero
    pq.put((dist[start], start))
    
    while 0 != pq.qsize():
        curr_cost, curr = pq.get()
        visited.add(curr)
        print(f'visiting {curr}')
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
                        # remove old
                        _ = pq.get((dist[neighbor],neighbor))
                        # insert new
                        pq.put((dist[neighbor],neighbor))
                        
    print("=== Dijkstra's Algo Output ===")
    print("Distances")
    print(dist)
    print("Visited")
    print(visited)
    print("Previous")
    print(prev)
    # we are done after every possible path has been checked 
    path_to_return = backtrace(prev, start, end)
    if return_prev_and_dist:
        return dist, prev, path_to_return
    else:
        return path_to_return, dist[end]
    
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
    

def ksp_yen(graph, node_start, node_end, max_k=2):
    """credits https://stackoverflow.com/questions/15878204/k-shortest-paths-implementation-in-igraph-networkx-yens-algorithm"""
    distances, previous, path = dijkstra(graph, node_start, node_end) #nx.single_source_dijkstra(graph, node_start, target, weight=weight)

    A = [{'cost': distances[node_end], 
          'path': path}]
    B = []

    if not A[0]['path']: return A

    for k in range(1, max_k):
        for i in range(0, len(A[-1]['path']) - 1):
            node_spur = A[-1]['path'][i]
            path_root = A[-1]['path'][:i+1]

            edges_removed = []
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i+1]:
                    cost = remove_edge_and_return_cost(graph, curr_path[i], curr_path[i+1])
                    if cost == -1:
                        continue
                    edges_removed.append([curr_path[i], curr_path[i+1], cost])

            path_spur = dijkstra(graph, node_spur, node_end)

            if path_spur['path']:
                path_total = path_root[:-1] + path_spur['path']
                dist_total = distances[node_spur] + path_spur['cost']
                potential_k = {'cost': dist_total, 'path': path_total}

                if not (potential_k in B):
                    B.append(potential_k)

            for edge in edges_removed:
                graph.add_edge(edge[0], edge[1], edge[2])

        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            A.append(B[0])
            B.pop(0)
        else:
            break

    return A

G_nx = copy.deepcopy(G)
dijkstra(G_nx,0,14)

ksp_yen(graph=G_nx, node_start=0, node_end=14, max_k=2)



#%% Mutation tests
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
