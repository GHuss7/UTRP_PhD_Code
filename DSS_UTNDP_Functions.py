# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:44:36 2019

@author: 17832020
"""
# %% Import libraries
import re
import numpy as np
import pandas as pd 
import igraph as ig
import matplotlib.pyplot as plt
import math
import random
from itertools import compress
import networkx as nx
import copy
import datetime
import multiset
import os
import pickle
import itertools
import scipy.sparse


# import pygmo as pg


from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.util.dominator import Dominator
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.misc import random_permuations
from pymoo.factory import get_performance_indicator


import DSS_UTNDP_Classes as gc
import EvaluateRouteSet as ev

# %% Fast list of lists copy function
# R_1 = [[0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123],
#        [0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123,0,1,2,3,4,5,6,7,8,10,12,34,545,35,24,12,123]]
# %timeit R_copy = copy.deepcopy(R_1)
# R_copy = R_1[:]
# R_copy[0][0] = 10
# R_1

# %timeit R_copy = [x[:] for x in R_1]
#R_copy[0][0] = 10
#assert R_1[0][0] != R_copy[0][0]

def lol_copy(list_of_lists):
    '''A function that returns a deepcopy of a list of lists, but faster than deepcopy'''
    list_of_lists_copy = [x[:] for x in list_of_lists]
    return list_of_lists_copy

# %% Range functions 
def rangeEx(a,b,c):
    # A function for creating a range between a and b and excluding a number c
    x = np.array(range(a,b))
    x = np.delete(x, np.argwhere(x == c))
    return x    
    
def rangeExArray(a,b,c):
    # A function for creating a range between a and b and excluding an array c
    x = np.array(range(a,b))
    for i in range(len(c)):
        x = np.delete(x, np.argwhere(x == c[i]))
    return x

def arrayEx(a,c):
    # A function for creating an array a and excluding an array c
    for i in range(len(c)):
        a = np.delete(a, np.argwhere(a == c[i]))
    return a

#%% Read input data functions

def format_mx_dist(matrix_dist): 
    # Format distance mx by removing first collumn
    n = matrix_dist.shape[1]
    return matrix_dist.iloc[:,1:n]

def format_mx_coords(mx_coords):
    coords_list = list()
    for i in range(len(mx_coords)):
        coords_list.append([ mx_coords.iloc[i,0], mx_coords.iloc[i,1]])
    return coords_list

def read_problem_data_to_matrices(problem_name):
    # A function to read in and format the relevant data into matrices
    mx_dist = pd.read_csv("./Input_Data/"+problem_name+"/Distance_Matrix.csv") 
    mx_dist = format_mx_dist(mx_dist)
    mx_dist = mx_dist.values


    mx_demand = pd.read_csv("./Input_Data/"+problem_name+"/OD_Demand_Matrix.csv") 
    mx_demand = format_mx_dist(mx_demand)
    mx_demand = mx_demand.values

    mx_coords = pd.read_csv("./Input_Data/"+problem_name+"/Node_Coords.csv")
    mx_coords = format_mx_coords(mx_coords)
    
    return mx_dist, mx_demand, mx_coords

def read_problem_data_to_matrices_direct(problem_name):
    # A function to read in and format the relevant data into matrices
    mx_dist = pd.read_csv("./Input_Data/"+problem_name+"/Distance_Matrix.csv") 
    mx_dist = format_mx_dist(mx_dist)
    mx_dist = mx_dist.values


    mx_demand = pd.read_csv("./Input_Data/"+problem_name+"/OD_Demand_Matrix.csv") 
    mx_demand = format_mx_dist(mx_demand)
    mx_demand = mx_demand.values

    mx_coords = pd.read_csv("./Input_Data/"+problem_name+"/Node_Coords.csv")
    mx_coords = format_mx_coords(mx_coords)
    
    return mx_dist, mx_demand, mx_coords


    
# %% Graph-based functions
def get_links_list_and_distances(matrix_dist):
    # Creates a list of all the links in a given adjacency matrix and a 
    # corresponding vector of distances associated with each link
    #  Output: (links_list, links_distances) [list,float64]
    
    max_distance = matrix_dist.max().max() # gets the max value in the matrix
    matrix_dist_shape = (len(matrix_dist),len(matrix_dist[0])) # row = entry 0, col = entry 1, stores the values (efficiency)
    links_list_dist_mx = list() # create an empty list to store the links

    # from the distance matrix, get the links list
    for i in range(matrix_dist_shape[0]):
        for j in range(matrix_dist_shape[1]):
            val = matrix_dist[i,j]
            if val != 0 and val != max_distance and i>j: # i > j yields only single edges, and not double arcs
                links_list_dist_mx.append((i,j))

    # Create the array to store all the links' distances
    links_list_distances = np.int64(np.empty(shape=(len(links_list_dist_mx)))) # create the array

    # from the distance matrix, store the distances for each link
    for i in range(len(links_list_dist_mx)): 
        links_list_distances[i] = matrix_dist[links_list_dist_mx[i]]
    
    return links_list_dist_mx, links_list_distances

def get_graph_distance_levels_from_vertex_u(vertex_u, max_depth, mapping_adjacent):
    """A function to determine the different level of distances of each vertex,
    i.e. d(u,v) and then give the levels in a list, with each level being d = 0,1,2 etc
    Input:
        vertex_u: the vertex from which distance will be measured
        max_depth: maximum depth that the distance list is allowed to reach
        mapping_adjacent: list of all the adjacent vertices in the graph to each vertex
    """

    visited = {vertex_u}
    distance_list = [[vertex_u]] 
    n_vertices = len(mapping_adjacent) # the number of vertices contained in the graph
    
    while (len(visited) < n_vertices) & (len(distance_list) < max_depth):
        next_level = []
        for vertex_v in distance_list[len(distance_list)-1]:
            next_level.extend(mapping_adjacent[vertex_v])
            
        next_level = list(set(next_level).difference(visited))
            
        visited = visited.union(set(next_level))    
        distance_list.append(next_level)
        
    return distance_list
    
# %% Create igraph from dist matrix 
def create_igraph_from_dist_mx(matrix_dist, edge_name="distance"): 
    # Creates an iGraph from the given distance matrix
    # Output: iGraph
    
    # Gets the links and their associated distances
    links_list_dist_mx, links_list_distances = get_links_list_and_distances(matrix_dist)
    
    # Create the transit network graph
    g_tn = ig.Graph() 

    g_tn.add_vertices(range(matrix_dist.shape[0])) # Add vertices

    g_tn.add_edges(links_list_dist_mx) # Add edges

    g_tn.es[edge_name] = links_list_distances
    
    return g_tn

def create_igraph_from_demand_mx(matrix_demand, edge_name="demand"): 
    # Creates an iGraph from the given demand matrix
    # Output: iGraph
    
    # Gets the links and their associated demands
    links_list_dist_mx, links_list_distances = get_links_list_and_distances(matrix_demand)
    
    # Create the transit network graph
    g_tn = ig.Graph() 

    g_tn.add_vertices(range(matrix_demand.shape[0])) # Add vertices

    g_tn.add_edges(links_list_dist_mx) # Add edges

    g_tn.es[edge_name] = links_list_distances
    
    return g_tn

# %% Generate all the shortest paths

def get_all_shortest_paths(g_n, criteria="distance"):
    # Takes as input an iGraph object
    # Output: list of all the shortest routes
    paths_shortest_all = list()
    for i in range(g_n.vcount()):
        paths_shortest_all.extend(g_n.get_all_shortest_paths(i, g_n.vs, criteria)) # figure out this function.... https://pythonhosted.org/python-igraph/igraph.GraphBase-class.html#get_all_paths_shortest_all
    
    for i in range(len(paths_shortest_all)-1,-1,-1):
        if len(paths_shortest_all[i]) < 2:  
            del paths_shortest_all[i]
    
    return paths_shortest_all  

def get_all_longest_paths(g_n, criteria="demand"):
    # Takes as input an iGraph object
    # Output: list of all the shortest routes
    paths_longest_all = list()
    for i in range(g_n.vcount()):
        paths_longest_all.extend(g_n.get_all_longest_paths(i, g_n.vs, criteria)) # figure out this function.... https://pythonhosted.org/python-igraph/igraph.GraphBase-class.html#get_all_paths_longest_all
    
    for i in range(len(paths_longest_all)-1,-1,-1):
        if len(paths_longest_all[i]) < 2:  
            del paths_longest_all[i]
    
    return paths_longest_all

#%% K-shortest paths with networkx and memory error
def get_path_length(G, path, weight='weight'):
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            
            length += G.get_edge_data(u,v).get(weight)
    
    return length

def get_k_shortest_paths_nx(G, source, target, k_cutoff):
    """Get all the k-shortest paths from a graph in terms of vertex counts for
    a given source and target vertex with a cutoff of k vertices
    Returns the paths and lengths between source and target.
    
    NB: This runs into a memory error for larger graphs as all the simple paths
    are generated"""
    # https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

    k_shortest_paths = []
    k_shortest_paths_lengths = []
    for path in nx.all_simple_paths(G, source, target, cutoff=k_cutoff):
        k_shortest_paths.append(path)
        k_shortest_paths_lengths.append(get_path_length(G, path, weight='weight'))
        
    return k_shortest_paths, k_shortest_paths_lengths

def get_all_k_shortest_paths_nx(G, k_cutoff):
    """Get all the k-shortest paths from a graph in terms of vertex counts for
    all vertices with a cutoff of k vertices
    Returns all the paths and lengths"""
    num_nodes = len(G.nodes)
    k_shortest_paths_all = []
    k_shortest_paths_lengths_all = []
    
    for v_i in range(num_nodes):
        for v_j in range(num_nodes):
            if v_i < v_j:
                k_shortest_paths, k_shortest_paths_lengths = get_k_shortest_paths_nx(G, v_i, v_j, k_cutoff)
                k_shortest_paths_all.extend(k_shortest_paths)
                k_shortest_paths_lengths_all.extend(k_shortest_paths_lengths)
                
    return k_shortest_paths_all, k_shortest_paths_lengths_all

def create_k_shortest_paths_df_nx(mx_dist, mx_demand, k_cutoff): 
    df_k_shortest_paths = pd.DataFrame(columns=["Source", "Target", "Travel_time", "Demand", "Demand_per_minute", "Routes"])

    nx_adj_mx = copy.deepcopy(mx_dist)
    for i in range(len(nx_adj_mx)):   
        for j in range(len(nx_adj_mx)):
            if nx_adj_mx[i,j] == np.max(nx_adj_mx): 
                nx_adj_mx[i,j] = 0
    G = nx.from_numpy_matrix(np.asarray(nx_adj_mx))    

    k_shortest_paths_all, k_shortest_paths_lengths = get_all_k_shortest_paths_nx(G, k_cutoff)
    k_shortest_paths_demand = determine_demand_per_route(k_shortest_paths_all, mx_demand)
    demand_per_minute = np.asarray(k_shortest_paths_demand) / np.asarray(k_shortest_paths_lengths)
    
    for index_i in range(len(k_shortest_paths_all)):
        df_k_shortest_paths.loc[index_i] = [k_shortest_paths_all[index_i][0],
                                             k_shortest_paths_all[index_i][-1],
                                             k_shortest_paths_lengths[index_i],
                                             k_shortest_paths_demand[index_i],
                                             demand_per_minute[index_i],
                                             convert_path_list2str(k_shortest_paths_all[index_i])]
        
        df_k_shortest_paths["Travel_time"] = np.float64(df_k_shortest_paths["Travel_time"].values)
        df_k_shortest_paths["Demand"] = np.float64(df_k_shortest_paths["Demand"].values)    
        
    return df_k_shortest_paths

# %% Calculate the shortest distance matrix
def calculate_shortest_dist_matrix(paths_shortest_all, mx_dist):
    # Takes as input list of all shortest paths and the associated distance matrix
    # Output: matrix of all shortest distances
    mx_shortest_distances = np.full((len(mx_dist),len(mx_dist)), 0)
    
    for i in range(len(paths_shortest_all)):
        path = paths_shortest_all[i]
        dist = 0
        length_path = len(path)
        for j in range(length_path-1):
            dist = dist + mx_dist[path[j],path[j+1]]
    
        mx_shortest_distances[path[0],path[length_path-1]] = mx_shortest_distances[path[length_path-1],path[0]] = dist
    
    return mx_shortest_distances  

# %% Feasibility of connected route set
def test_graph_connectedness(R,N):
    
    # from Fan, Lang Mumford, Christine L. 2010
    # Function that tests the feasibility of all the nodes being included and connected
    # N is the number of nodes in network 
    # R is imported candidate route set in list format for a solution
      
    # Using Algorithm 1 from Fan and Mumford 2010 to determine whether all nodes are included --------
    # The while loop was adapted because it does not make sense to have the feasibility as the while
    # loop logic test because infeasible solutions will result in infinite loops
      
    foundNode = np.full(N,0)                              # records nodes that have been found
    exploredNode = np.full(N,0)                           # records nodes that have been explored
    iRoutesFound = []                                     # vector to keep track of the routes found containing node i
      
    iNode = R[0][0]                                       # select an arbitrary node i present in at least one route
    feas = False
    switchTest = False
    counter = 1
      
    while switchTest == False:
        exploredNode[iNode] = foundNode[iNode] = 1
        iRoutesFound =[]                                  # Find all routes containing node i
          
        for k in range(len(R)):                           # loop over the number of routes in analysis
            for g in range(len(R[k])):                    # set flags in found node to record nodes found in the routes containing i 
                if R[k][g] == iNode:
                    iRoutesFound.append(k)
                  
        # Set flags in found-node to record all the nodes found in those routes
        for k in range(len(iRoutesFound)):               # loop over the number of routes in analysis
            for g in range(len(R[iRoutesFound[k]])):     # set flags in found node to record nodes found in the routes containing i 
                foundNode[R[iRoutesFound[k]][g]] = 1 
                  
        # Select any node from found-node that is absent from explored-node          
        for i in range(N):
            if foundNode[i] != exploredNode[i]:  
                iNode = i
            
        if sum(foundNode) == N:
            feas = True
            switchTest = True
               
        if counter > N*N and sum(foundNode == exploredNode) == N:
            switchTest = True   
               
        counter = counter + 1
           
    return feas  

# %% Generate candidate route set
def generate_candidate_route_solution(paths_shortest_all, con_r):
    # paths_shortest_all is the route set containing all possible shortest routes from node i to j 
    # subject to certain constraints that should have been predefined
    # con_r is the number of allowed routes
    
    paths_candidate = list()        # create the decision variables to keep the candidate solutions
    
    k = random.sample(range(len(paths_shortest_all)),con_r)
    for i in range(con_r):
        paths_candidate.append(paths_shortest_all[k[i]])
        
    return paths_candidate

# Crossover type route generation based on unseen vertices probability
def routes_generation_unseen_prob(parent_i, parent_j, solution_len):
    """Crossover function for routes based on Mumford 2013's Crossover function
    for routes based on alternating between parents and including a route from
    each parent that maximises the unseen vertices added to the child route
    Note: only generates one child, needs to be tested for feasibility and repaired if needed"""
    parents = []  
    parents.append(copy.deepcopy(parent_i))
    parents.append(copy.deepcopy(parent_j))
    parent_index = random.randint(0,1) # counter to help alternate between parents  
    
    child_i = [] # define child
    parent_len = len(parent_i)
    
    # Randomly select the first seed solution for the child
    random_index = random.randint(0,parent_len-1)
    child_i.append(parents[parent_index][random_index]) # adds seed solution to parent
    del(parents[parent_index][random_index]) # removes the route from parent so that it is not evaluated again
    
    # Alternates parent solutions
    parent_index = looped_incrementor(parent_index, 1)
    
    
    # Calculate the unseen proportions to select next route for inclusion into child
    while len(child_i) < solution_len:
        # Determines all nodes present in the child
        all_nodes_present = set([y for x in child_i for y in x]) # flatten all the elements in child
    
        parent_curr = parents[parent_index] # sets the current parent
        
        proportions = []
        for i_candidate in range(len(parent_curr)):
            R_i = set(parent_curr[i_candidate])
            if bool(R_i.intersection(all_nodes_present)): # test whether there is a common transfer point
                proportions.append(len(R_i - all_nodes_present) / len(R_i)) # calculate the proportion of unseen vertices
            else:
                proportions.append(0) # set proportion to zero so that it won't be chosen
        
        # Get route that maximises the proportion of unseen nodes included
        max_indices = set([i for i, j in enumerate(proportions) if j == max(proportions)]) # position of max proportion/s
        max_index = random.sample(max_indices, 1)[0] # selects only one index randomly between a possible tie, else the only one
        
        # Add the route to the child
        child_i.append(parent_curr[max_index]) # add max proportion unseen nodes route to the child
        del(parents[parent_index][max_index]) # removes the route from parent so that it is not evaluated again
        
        # Alternates parent solutions
        parent_index = looped_incrementor(parent_index, 1)
    
    return child_i

def routes_generation_unseen_probabilistic(parent_i, parent_j, solution_len):
    """Crossover function for routes based on Mumford 2013's Crossover function
    for routes based on alternating between parents and including a route from
    each parent that maximises the unseen vertices added to the child route probabilisticly
    Note: only generates one child, needs to be tested for feasibility and repaired if needed"""
    parents = []  
    parents.append(copy.deepcopy(parent_i))
    parents.append(copy.deepcopy(parent_j))
    parent_index = random.randint(0,1) # counter to help alternate between parents  
    
    child_i = [] # define child
    parent_len = len(parent_i)
    
    # Randomly select the first seed solution for the child
    random_index = random.randint(0,parent_len-1)
    child_i.append(parents[parent_index][random_index]) # adds seed solution to parent
    del(parents[parent_index][random_index]) # removes the route from parent so that it is not evaluated again
    
    # Alternates parent solutions
    parent_index = looped_incrementor(parent_index, 1)
    
    
    # Calculate the unseen proportions to select next route for inclusion into child
    while len(child_i) < solution_len:
        # Determines all nodes present in the child
        all_nodes_present = set([y for x in child_i for y in x]) # flatten all the elements in child
    
        parent_curr = parents[parent_index] # sets the current parent
        
        proportions = []
        for i_candidate in range(len(parent_curr)):
            R_i = set(parent_curr[i_candidate])
            if bool(R_i.intersection(all_nodes_present)): # test whether there is a common transfer point
                proportions.append(len(R_i - all_nodes_present) / len(R_i)) # calculate the proportion of unseen vertices
            else:
                proportions.append(0) # set proportion to zero so that it won't be chosen
        
        # Get route that maximises the proportion of unseen nodes included
        if sum(proportions) == 0:
            max_indices = [i for i, j in enumerate(proportions)] # position of all proportion/s
            max_index = random.sample(max_indices, 1)[0] # selects only one index randomly between a possible tie, else the only one

        else:
            max_indices = [i for i, j in enumerate(proportions)] # position of max proportion/s
            proportions = np.array(proportions)/sum(proportions)     
            max_index = random.choices(max_indices, weights=proportions, k=1)[0]
            
        
        # Add the route to the child
        child_i.append(parent_curr[max_index]) # add max proportion unseen nodes route to the child
        del(parents[parent_index][max_index]) # removes the route from parent so that it is not evaluated again
        
        # Alternates parent solutions
        parent_index = looped_incrementor(parent_index, 1)
    
    return child_i

def normalise_route_set(R_x):
    all_nodes = [y for x in R_x for y in x]
    if 0 not in all_nodes:
        for i in range(len(R_x)): # get routes in the correct format
            R_x[i] = [x - 1 for x in R_x[i]] # subtract 1 from each element in the list
        del i
    return R_x

 # Load or create and save initial population
def load_obj_pickle(name, directory):
    '''Function to easily load object from pickle file'''
    with open(directory / (name+".pickle"),'rb') as read_file:
        obj =  pickle.load(read_file) # load the object
        return obj

def save_obj_pickle(obj, name, directory):
    '''Function to easily save object to pickle file'''
    pickle.dump(obj, open(directory / (name+".pickle"), "ab"))

def load_UTRP_pop_or_create(name, directory, main_problem, route_gen_func, fn_obj, pop_size_to_create=False):
    '''A function that loads the population data if it exists, and creates it
    otherwise. This will help longterm to save time.'''
    if (directory / (name+".pickle")).exists():
        try:
            pop_1 = load_obj_pickle(name, directory)
            print(f'LOADED: Population {name} loaded from {directory}')
        except EOFError as err:
            print(f"Exception in load_UTRP_pop_or_create:: {err}")
            try: 
                directory.mkdir() 
            except OSError as error: 
                print(error) 
            pop_1 = gc.PopulationRoutes(main_problem)  
            pop_1.generate_initial_population_smart(main_problem, fn_obj, route_gen_func, new_pop_size=pop_size_to_create)
            save_obj_pickle(pop_1, name, directory)
            print(f'SAVED: Population by {route_gen_func.__name__} saved to {directory}')
    
    else:
        try: 
            directory.mkdir() 
        except OSError as error: 
            print(error) 
        pop_1 = gc.PopulationRoutes(main_problem)  
        pop_1.generate_initial_population_smart(main_problem, fn_obj, route_gen_func, new_pop_size=pop_size_to_create)
        save_obj_pickle(pop_1, name, directory)
        print(f'SAVED: Population by {route_gen_func.__name__} saved to {directory}')
        
    return pop_1

def load_UTRP_supplemented_pop_or_create(name, directory, main_problem, route_gen_func, fn_obj, pop_loaded):
    '''A function that loads the supplemented population data if it exists, and creates it
    otherwise. This will help longterm to save time.'''
    if (directory / (name+".pickle")).exists():
        pop_1 = load_obj_pickle(name, directory)
        print(f'LOADED: Supplemented population {name} loaded from {directory}')
    
    else:
        try: 
            directory.mkdir() 
        except OSError as error: 
            print(error) 
        pop_1 = gc.PopulationRoutes(main_problem)  
        pop_1.supplement_initial_population_smart(main_problem, fn_obj, pop_loaded, route_gen_func)
        save_obj_pickle(pop_1, name, directory)
        print(f'SAVED: Supplemented population by {route_gen_func.__name__} saved to {directory}')
        
    return pop_1


# %% Try to generate a feasible solution 
def generate_solution(paths_shortest_all, con_r, N , iterations):
    # Generate a feasible solution where all routes are connected
    # paths_shortest_all are the set of all possible routes one can choose from
    # con_r is the number of routes you want generated
    # N is the number of nodes in the network
    # i are the iterations that should be performed
    # Output: feasible_solution or False if none exist
  
    for i in range(iterations):
        routes_R = generate_candidate_route_solution(paths_shortest_all, con_r)
        if test_all_nodes_and_connectedness_nx(routes_R,N):
            return routes_R
    return False            # returns false if a feasible solution is not generated

# %% Generate a feasible solution succesfully ################################################
def generate_feasible_solution(paths_shortest_all, con_r, N , iterations):
    # This code is used to generate a set of routes that are connected with each other
    # from the possible set of candidate routes presented
    # under the the constraint of the number of allowed routes specified
    # in a network consisting of N nodes
    # with iterations being the iterations allowed
    
    for i in range(N):
        routes_R = generate_solution(paths_shortest_all, con_r, N , iterations)

        if routes_R is False:    # tests if the solution was feasible, if not, leverage the
                                # constraint by adding one more allowable route
            con_r = con_r + 1
            print("No feasible solution found, increasing number of routes allowed by 1.")
        else:
            if test_all_nodes_and_connectedness_nx(routes_R,N):
                break
    return routes_R


def generate_feasible_solution_smart(pop, main_problem):
    """Crossover function for entire route population"""
    selection = tournament_selection_g2(pop, n_select=int(main_problem.problem_GA_parameters.population_size))
    
    if random.random() < main_problem.problem_GA_parameters.crossover_probability:
        offspring_variables = [None] * main_problem.problem_GA_parameters.population_size
    
        
        for i in range(0,int(main_problem.problem_GA_parameters.population_size)):
            parent_A = pop.variables[selection[i,0]]
            parent_B = pop.variables[selection[i,1]]
        
            offspring_variables[i] = crossover_routes_unseen_prob(parent_A, parent_B)
                # crossover_uniform_as_is(parent_A, parent_B, main_problem.R_routes.number_of_routes)
            
            while not test_all_four_constraints(offspring_variables[i], main_problem):
                offspring_variables[i] = repair_add_missing_from_terminal(offspring_variables[i], main_problem)
                
                if test_all_four_constraints(offspring_variables[i], main_problem):
                    continue
                else:
                    offspring_variables[i] = crossover_routes_unseen_prob(parent_A, parent_B)
        
        return offspring_variables
    
    else:
        return pop.variables


def generate_initial_route_sets(main_problem, printing=True, pareto_efficient=True):
    '''Generate initial route sets for input as initial solutions'''
    if printing:
        print("Started initial route set generation"+"("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")") 
    routes_R_initial_set = list()
    for i in range(main_problem.problem_SA_parameters.number_of_initial_solutions):
        routes_R_initial_set.append(gc.Routes.return_feasible_route_robust(main_problem))
    if printing:
        print(f"Initial route set generated [size: {len(routes_R_initial_set)}]"+"("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")   
    
    df_routes_R_initial_set =  pd.DataFrame(columns=["f_1","f_2","Routes"])   
    for i in range(len(routes_R_initial_set)):
        f_new = ev.evalObjs(routes_R_initial_set[i], main_problem.problem_data.mx_dist, main_problem.problem_data.mx_demand, main_problem.problem_inputs.__dict__)
        df_routes_R_initial_set.loc[i] = [f_new[0], f_new[1], convert_routes_list2str(routes_R_initial_set[i])]
    
    if printing: print("Started Pareto generation"+"("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")") 
    if pareto_efficient:
        df_routes_R_initial_set = df_routes_R_initial_set[is_pareto_efficient(df_routes_R_initial_set.iloc[:,0:2].values, True)] # reduce the pareto front from the total archive
    if printing: print("Ended Pareto generation"+"("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")") 
    
    routes_R_initial_set = list()
    for i in range(len(df_routes_R_initial_set)):
        routes_R_initial_set.append(convert_routes_str2list(df_routes_R_initial_set.iloc[i,2]))
    
    if printing: print("Initial route set generated with size: "+str(len(routes_R_initial_set)))
    return routes_R_initial_set, df_routes_R_initial_set

# Generate longer feasible solutions and covering demand with direct routes
def calc_cum_demand(r_i, mx_demand):
    '''Calc cumulative demand for a single path'''
    dem_tot = 0 # initiate demand count
    # for each connection, calculate demand between OD pairs direct route
    for i in r_i:
        for j in r_i:
            if i < j:
                dem_tot = dem_tot + mx_demand[i,j] + mx_demand[j,i]
    
    return dem_tot

def calc_cum_demand_route_set_slow(r_i, mx_demand):
    '''Calc cumulative demand for a route set -- slow for loop'''
    dem_tot = 0 # initiate demand count
    # for each connection, calculate demand between OD pairs direct route
    for path_i in range(len(r_i)):
        for i in r_i[path_i]:
            for j in r_i[path_i]:
                if i < j:
                    dem_tot = dem_tot + mx_demand[i,j] + mx_demand[j,i]
    
    return dem_tot

def calc_cum_demand_route_set(r_i, mx_demand):
    '''Calc cumulative demand for a route set -- faster with list comprehension'''
    dem_combinations = [i for j in [list(itertools.combinations(x,2)) for x in r_i] for i in j]
    return sum([mx_demand[tup] for tup in dem_combinations])*2 

def calc_cum_demand_route_set_vectorisation(r_i, mx_demand):
    dem_combinations = [i for j in [list(itertools.combinations(x,2)) for x in r_i] for i in j]
    A = np.zeros((mx_demand.shape)) # answers
    B = np.array(dem_combinations) # coordinates
    C = mx_demand # values to sum
    
    B_sparse = scipy.sparse.coo_matrix(B).todense()
    
    A = B_sparse*C
    
    np.ravel(B, order='C')
    
    
    i = np.arange(2)[:,np.newaxis]
    mask = (B == i).astype(int)
    for j in range(len(mx_demand)):
        A[:,j] = (C[j] * mask).sum(axis=-1)
    
    
    total = 0
    for tup in dem_combinations:
        total = total + mx_demand[tup]
    return (total*2)   


    

def remove_cum_demand(r_i, mx_demand):
    '''Remove cumulative demand for a single path'''
    # for each connection, remove demand between OD pairs direct route
    mx_demand_copy = copy.deepcopy(mx_demand)
    for i in r_i:
        for j in r_i:
            if i < j:
                mx_demand_copy[i,j] = 0
                mx_demand_copy[j,i] = 0
    
    return mx_demand_copy

def remove_cum_demand_route_set(r_i, mx_demand):
    '''Remove cumulative demand for a route set'''
    # for each connection, remove demand between OD pairs direct route
    mx_demand_copy = copy.deepcopy(mx_demand)
    for path_i in range(len(r_i)):
        for i in r_i[path_i]:
            for j in r_i[path_i]:
                if i < j:
                    mx_demand_copy[i,j] = 0
                    mx_demand_copy[j,i] = 0
    
    return mx_demand_copy

def get_vertex_with_max_unmet_demand(mx_demand):
    vetrex_list = []
    d_best = 0
    for i in range(len(mx_demand)):
        d = sum(mx_demand[i,:]) + sum(mx_demand[:,i])
        
        if d == d_best:
            vetrex_list.append(i)
        
        if d > d_best:
            d_best = d
            vetrex_list = [i]
            
    return random.choice(vetrex_list)
        
def generate_feasible_route_set_greedy_demand(main_problem):

    # Get constraints
    con_r = main_problem.problem_constraints.con_r # get number of routes
    con_max_v = main_problem.problem_constraints.con_maxNodes
    con_min_v = main_problem.problem_constraints.con_minNodes
    n_vertices = main_problem.problem_inputs.n
    mapping_adjacent = main_problem.mapping_adjacent 
    R_x = [] 
    counter = 0
    
    while not test_all_four_constraints(R_x, main_problem) and counter < 10000:                                                                      
    
        R_x = [] # create route set x
        mx_d_temp = copy.deepcopy(main_problem.problem_data.mx_demand)
        
        while len(R_x) < con_r:
            r_i = [] # create route i
            route_vertices = set([y for x in R_x for y in x]) # flatten elements in route
            missing_vs = list(set(range(n_vertices)).difference(route_vertices))
            
            try:
                v = random.choice(missing_vs)
            except:
                v = get_vertex_with_max_unmet_demand(mx_d_temp)
          
            r_i.append(v)
            flag_r_swop = 0 # flag to swop route around if v is terminal/ infeasible
            
            while len(r_i) < con_max_v:
                adjs = list(set(mapping_adjacent[r_i[-1]]).difference(set(r_i)))
                if len(adjs) == 0:
                    if not flag_r_swop:
                        flag_r_swop = 1
                        r_i.reverse()
                        break
                        
                random.shuffle(adjs)
                d_best = 0 # create variable to keep best demand satisfied thusfar
                
                # test for best additional vertex to include based on demand
                for adj in adjs:
                    r_temp = copy.deepcopy(r_i)
                    r_temp.append(adj)
                    d = calc_cum_demand(r_temp, mx_d_temp)
                    
                    if d > d_best:
                        d_best = d
                        r_best = copy.deepcopy(r_temp)
                    
                # if no demand can be met additionally
                if d_best == 0:
                    r_best = copy.deepcopy(r_temp) # random because adjs are shuffled
                
                r_i = copy.deepcopy(r_best)
                mx_d_temp = remove_cum_demand(r_i, mx_d_temp)
            
            if len(r_i) >= con_min_v:  
                R_x.append(r_i)
        
        counter += 1
        
    return R_x



def init_temp_trial_searches(UTNDP_problem, number_of_runs=1, P_0=0.999, P_N=0.001, N_search_epochs=1000):
    '''Test for starting temperature and cooling schedule'''
    """ number_of_runs: sets more trial runs for better estimates (averages)
        P_0: initial probability of acceptance
        P_N: final probability of acceptance
        N_search_epochs: roughly the number of desired search epochs
        returns: the starting temp and Beta coefficient for geometric cooling
    """
    print(f"Initiated trial search for initial temperature")
    
    def fn_obj(routes, UTNDP_problem):
        return (ev.evalObjs(routes, 
                UTNDP_problem.problem_data.mx_dist, 
                UTNDP_problem.problem_data.mx_demand, 
                UTNDP_problem.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)
    
    df_results = pd.DataFrame(columns=["Avg_E","T_0","T_N","Beta"])
    
    for run_nr in range(number_of_runs):
        
        routes_R_initial_set, df_routes_R_initial_set = generate_initial_route_sets(UTNDP_problem, False)
    
        for route_set_nr in range(len(routes_R_initial_set)):       
        
            routes_R = routes_R_initial_set[route_set_nr] # Choose the initial route set to begin with
            '''Initiate algorithm'''
            iteration_t = 1 # Initialise the number of iterations 
            counter_archive = 0
            
            df_energy_values = pd.DataFrame(columns=["Delta_E"]) 
            df_archive = pd.DataFrame(columns=["f1_ATT","f2_TRT","Routes"]) # create an archive in the correct format        
            
            f_cur = fn_obj(routes_R, UTNDP_problem)
            df_archive.loc[0] = [f_cur[0], f_cur[1], convert_routes_list2str(routes_R)]            
    
            while (iteration_t <= UTNDP_problem.problem_SA_parameters.M_iterations_for_temp):
                '''Generate neighbouring solution'''
                routes_R_new = perturb_make_small_change(routes_R, UTNDP_problem.problem_constraints.con_r, UTNDP_problem.mapping_adjacent)
                
                while not test_route_feasibility(routes_R_new, UTNDP_problem.problem_constraints.__dict__):    # tests whether the new route is feasible
                    for i in range(UTNDP_problem.problem_SA_parameters.Feasibility_repair_attempts): # this tries to fix the feasibility, but can be time consuming, 
                                            # could also include a "connectivity" characteristic to help repair graph
                        routes_R_new = perturb_make_small_change(routes_R_new, UTNDP_problem.problem_constraints.con_r, UTNDP_problem.mapping_adjacent)
                        if test_route_feasibility(routes_R_new, UTNDP_problem.problem_constraints.__dict__):
                            break
                    routes_R_new = perturb_make_small_change(routes_R, UTNDP_problem.problem_constraints.con_r, UTNDP_problem.mapping_adjacent) # if unsuccesful, start over
            
                f_new = fn_obj(routes_R_new, UTNDP_problem)
            
                df_energy_values.loc[len(df_energy_values)] = [abs(energy_function_for_initial_temp(df_archive, f_cur[0], f_cur[1], f_new[0], f_new[1]))]
            
                '''Test solution acceptance and add to archive if accepted and non-dominated'''
                routes_R = routes_R_new
                f_cur = f_new
                
                df_archive.loc[counter_archive] = [f_cur[0], f_cur[1], convert_routes_list2str(routes_R)] # adds the new solution
                counter_archive = counter_archive + 1 # this helps with speed 
                
                iteration_t = iteration_t + 1
        
            avg_E = df_energy_values[['Delta_E']].mean(axis=0)[0]
            T_0 = -(avg_E / math.log10(P_0))
            T_N = -(avg_E / math.log10(P_N))
            
            Beta = math.exp((math.log10(T_N) - math.log10(T_0)) / N_search_epochs)
        
        df_results.loc[run_nr] = [avg_E, T_0, T_N, Beta]

    result_means = df_results.mean(axis=0)
    return result_means[1], result_means[3] # T_0 and Beta cooling ratio

# %% Define the 1st Objective function, the sum of all the route lengths
def f1_total_route_length(routes_R, mx_dist):
    # Takes as input the route set R and the associated distance matrix
    # Output: an array of
    lengths_of_routes = np.full(len(routes_R), 0)
    
    for i in range(len(routes_R)):
        path = routes_R[i]
        dist = 0
        length_path = len(path)
        for j in range(length_path-1):
            dist = dist + mx_dist[path[j],path[j+1]]
    
        lengths_of_routes[i] = dist
    
    return sum(lengths_of_routes) 

# %% Define a function, the sum each route lengths
def calc_seperate_route_length(routes_R, mx_dist):
    # Takes as input the route set R and the associated distance matrix
    # Output: an array of
    num_of_routes = len(routes_R)
    lengths_of_routes = np.full(num_of_routes, 0)
    
    for i in range(num_of_routes):
        path = routes_R[i]
        dist = 0
        length_path = len(path)
        for j in range(length_path-1):
            dist = dist + mx_dist[path[j],path[j+1]]
    
        lengths_of_routes[i] = dist
    
    return lengths_of_routes

# %% Determine demand per route
def determine_demand_per_route(list_of_routes, mx_demand):
    """Takes as input a list of routes and a demand matrix and calculates the total demand that a route satisfies
    Returns a list containing the demand met by the route with the corresponding index """
    demand_for_routes_list = [0] * len(list_of_routes) # generates a list with only zeroes

    for list_counter in range(len(list_of_routes)):
        for i_counter in range(len(list_of_routes[list_counter])):
            for j_counter in range(len(list_of_routes[list_counter])):
                if i_counter != j_counter:  
                    demand_for_routes_list[list_counter] = demand_for_routes_list[list_counter] + mx_demand[list_of_routes[list_counter][i_counter], list_of_routes[list_counter][j_counter]]
    
    return demand_for_routes_list

# %% Generate bus network
def generate_bus_network_dist_mx(routes_R, mx_dist):
    # Generate Bus Route Network 
    # Calculate the allowed distances in the bus network only
    
    mx_dist_bus_network = np.full((len(mx_dist),len(mx_dist)),mx_dist.max())
    
    for i in range(len(routes_R)):
        for j in range(len(routes_R[i])-1):
            mx_dist_bus_network[routes_R[i][j],routes_R[i][j+1]] = \
            mx_dist_bus_network[routes_R[i][j+1],routes_R[i][j]] = mx_dist[routes_R[i][j],routes_R[i][j+1]]
    
    return mx_dist_bus_network

# %% Remove the duplicate that are in reverse order
def remove_duplicate_routes(shortest_routes, N):
    # removes the double entries in the shortest routes list, but which are in reverse order
    # shortest_routes is the shortest bus routes in the network from node i to all other nodes
    # N is the number of nodes in the network
    # Output: the shortest_routes where the duplicates are removed
    
    tick_list = list() # create a tick list to make sure all the routes are included
    test_list = list() # a list to keep in all the start and ending nodes of a route
    
    for i in range(len(shortest_routes)):
        test_list.append((shortest_routes[i][0],shortest_routes[i][len(shortest_routes[i])-1]))
    
    for i in reversed(range(len(shortest_routes))):
        if test_list[i] in tick_list or (test_list[i][1],test_list[i][0]) in tick_list:
            del shortest_routes[i]
        else:
            tick_list.append(test_list[i])
            
    return shortest_routes

def remove_half_duplicate_routes(shortest_routes):
    # removes the double entries in the shortest routes list, but which are in reverse order
    # keeps the doubles between same source and targer
    # shortest_routes is the shortest bus routes in the network from node i to all other nodes
    # N is the number of nodes in the network
    # Output: the shortest_routes where the duplicates are removed
    shortest_routes_half = copy.deepcopy(shortest_routes)
    
    for i in range(len(shortest_routes)-1,-1,-1):
        if shortest_routes[i][0] > shortest_routes[i][-1]:
            shortest_routes_half.pop(i)
            
    return shortest_routes_half

# %% Generate transfer matrix
def generate_transfer_mx(routes_R, paths_shortest_bus_routes, N):    
    # generate the transfer matrix that counts how much transfers each customer 
    # undergoes per OD pair
    # routes_R is the paths
    # paths_shortest_bus_routes is the shortest routes passengers can take in the network
    # N is is number of nodes in the network
    # Output: the transfer matrix
    
    mx_transfer = np.full((N,N),3) # three transfers is set as the limit and a penalty
    for i in range(N): mx_transfer[i,i] = 0 # make diagonals 0 
                    
    sbr = paths_shortest_bus_routes # shortest bus routes

    for i in range(len(sbr)):
        flag = 0
        for k in range(len(routes_R)):
            if all(x in routes_R[k] for x in sbr[i]): # test for 0 transfers
                        mx_transfer[sbr[i][0],sbr[i][len(sbr[i])-1]] = \
                        mx_transfer[sbr[i][len(sbr[i])-1],sbr[i][0]] = 0
                        flag = 1
                        break
            if not flag:
                for k in range(len(routes_R)):
                    for l in rangeEx(0,len(routes_R),k):
                        if all(x in routes_R[k] or x in routes_R[l] for x in sbr[i]):
                            mx_transfer[sbr[i][0],sbr[i][len(sbr[i])-1]] = \
                            mx_transfer[sbr[i][len(sbr[i])-1],sbr[i][0]] = 1
                            flag = 1
                            break
            if not flag:
                for k in range(len(routes_R)):
                    for l in rangeEx(0,len(routes_R),k):
                        for m in rangeExArray(0,len(routes_R),[k,l]):
                            if all(x in routes_R[k] or x in routes_R[l] or x in routes_R[m] for x in sbr[i]):
                                mx_transfer[sbr[i][0],sbr[i][len(sbr[i])-1]] = \
                                mx_transfer[sbr[i][len(sbr[i])-1],sbr[i][0]] = 2
                                flag = 1
                                break   
    
    return mx_transfer

# %% Generate transfer matrix 2
def generate_transfer_mx2(routes_R, paths_shortest_bus_routes, N):    
    # generate the transfer matrix that counts how much transfers each customer 
    # undergoes per OD pair
    # routes_R is the paths
    # paths_shortest_bus_routes is the shortest routes passengers can take in the network
    # N is is number of nodes in the network
    # Output: the transfer matrix
    
    mx_transfer = np.full((N,N),math.inf) # three transfers is set as the limit and a penalty
    for i in range(N): mx_transfer[i,i] = 0 # make diagonals 0 
                    
    sbr = paths_shortest_bus_routes # shortest bus routes

    route_links = convert_paths_to_links(routes_R)

    for i in range(len(routes_R)):
        for j in routes_R[i]:
            for k in rangeEx(0,routes_R[i],j):
                mx_transfer[j,k] = 777
                
    for i in range(N):
        for j in range(N):
            if i < j:    
                if (i,j) in routes_R or (j,i) in route_links:
                    print(i)
    
    
    
    
    return mx_transfer                    
                    
# %% Convert paths to links                    
def convert_paths_to_links(routes_R):
    # converts the routes list into a list of edges for each route
    routes_links = list()
    for r in range(len(routes_R)):
        route_links_holder = list()
        for i in range(len(routes_R[r])-1):
            route_links_holder.append((routes_R[r][i],routes_R[r][i+1]))
        routes_links.append(route_links_holder)
    return routes_links

def generate_initial_feasible_route_set(mx_dist, parameters_constraints):
    con_minNodes = parameters_constraints['con_minNodes']
    con_maxNodes = parameters_constraints['con_maxNodes']
    con_r = parameters_constraints['con_r']
    
  
    '''Create the transit network graph'''
    g_tn = create_igraph_from_dist_mx(mx_dist)
    
    paths_shortest_all = get_all_shortest_paths(g_tn) # Generate all the shortest paths
    
    # Shorten the candidate routes according to the constraints
    for i in range(len(paths_shortest_all)-1, -1, -1):
        if len(paths_shortest_all[i]) < con_minNodes or len(paths_shortest_all[i]) > con_maxNodes:  
            del paths_shortest_all[i]
    
    # Generate initial feasible solution
    routes_R = generate_feasible_solution(paths_shortest_all, con_r, len(mx_dist), 100000)

    return routes_R

def convert_path_list2str(path_list):
    # converts a routes list into a string standarised version
    path_str = str()

    for j in range(len(path_list)):
        if j != 0:
            path_str = path_str + "-" + str(path_list[j])
        else:
            path_str = path_str +  str(path_list[j])
    path_str = path_str + "*"
    return path_str

def convert_path_str2list(path_str):
    # converts a string standarised version of routes list into a routes list
    path_list = list()
    temp_list = list()
    flag_end_node = True
    for i in range(len(path_str)):
        if path_str[i] != "-" and path_str[i] != "*":
            if flag_end_node:
                temp_list.append(int(path_str[i]))
                flag_end_node = False
            else:
                temp_list[len(temp_list)-1] = int(str(temp_list[len(temp_list)-1]) + path_str[i])
        else:   
            if path_str[i] == "*":          # indicates the end of the route
                path_list.append(temp_list)
                temp_list = list()
                flag_end_node = True
            else:
                if path_str[i] == "-":
                    flag_end_node = True
    return path_list[0]

def convert_routes_list2str(routes_R_list):
    # converts a routes list into a string standarised version
    routes_R_str = str()
    r = len(routes_R_list)
    for i in range(r):
        for j in range(len(routes_R_list[i])):
            if j != 0:
                routes_R_str = routes_R_str + "-" + str(routes_R_list[i][j])
            else:
                routes_R_str = routes_R_str +  str(routes_R_list[i][j])
        routes_R_str = routes_R_str + "*"
    return routes_R_str
        
def convert_routes_str2list(routes_R_str):
    # converts a string standarised version of routes list into a routes list
    routes_R_list = list()
    temp_list = list()
    flag_end_node = True
    for i in range(len(routes_R_str)):
        if routes_R_str[i] != "-" and routes_R_str[i] != "*":
            if flag_end_node:
                temp_list.append(int(routes_R_str[i]))
                flag_end_node = False
            else:
                temp_list[len(temp_list)-1] = int(str(temp_list[len(temp_list)-1]) + routes_R_str[i])
        else:   
            if routes_R_str[i] == "*":          # indicates the end of the route
                routes_R_list.append(temp_list)
                temp_list = list()
                flag_end_node = True
            else:
                if routes_R_str[i] == "-":
                    flag_end_node = True
    return routes_R_list


# %% Get the mapping of all the adjacent nodes of each node in the form of a list

def get_mapping_of_adj_edges(mx_dist):
    val_n = len(mx_dist)
    mapping_adjacent = [[]]*val_n   # creates a mapping index of which nodes are adjacent
    bool1 = mx_dist.max()*np.ones([val_n,val_n], dtype=int) != mx_dist + mx_dist.max()*np.eye(val_n) # tests where the edges are
    for i in range(val_n):
        test = bool1[i]
        mapping_adjacent[i] = list(compress(range(len(test)), test)) # gets the positions of the true values
    return mapping_adjacent

# %% Make change functions
    
def change_add_node_to_first_node(routes_R, r, mapping_adjacent): # Add node to start of route
    # routes_R is the routes set list
    # r is the number of routes present
    i = random.randrange(r) # gets a random route position
    node_set = set(mapping_adjacent[routes_R[i][0]]) - set(routes_R[i]) # gets the difference in sets
    if node_set: # can be evaluated as a boolean expression
        node_a = random.sample(node_set, 1) # gets an adjacent node to first node not in route
        routes_R[i].insert(0, node_a[0])
    else:
        routes_tabu = [i]
        i = random.sample(set(range(r)) - set(routes_tabu), 1)[0]
        while set(range(r)) - set(routes_tabu):
            node_set = set(mapping_adjacent[routes_R[i][0]]) - set(routes_R[i]) # gets the difference in sets
            if node_set: # can be evaluated as a boolean expression   
                node_a = random.sample(node_set, 1) # gets an adjacent node to first node not in route
                routes_R[i].insert(0, node_a[0])
                break
            else:
                routes_tabu.append(i)
                if set(range(r)) - set(routes_tabu):
                    i = random.sample(set(range(r)) - set(routes_tabu), 1)[0]
                else:
                    # print("Could not add node to any first node")
                    break
               

def change_add_node_to_last_node(routes_R, r, mapping_adjacent): # Add node to end of route
    # routes_R is the routes set list
    # r is the number of routes present
    i = random.randrange(r) # gets a random route position
    R_i_len = len(routes_R[i])-1 # index of the last node
    node_set = set(mapping_adjacent[routes_R[i][R_i_len]]) - set(routes_R[i]) # gets the difference in sets
    if node_set: # can be evaluated as a boolean expression
        node_a = random.sample(node_set, 1) # gets an adjacent node to first node not in route
        routes_R[i].append(node_a[0])
    else:
        routes_tabu = [i]
        i = random.sample(set(range(r)) - set(routes_tabu), 1)[0]
        R_i_len = len(routes_R[i])-1 # index of the last node
        while set(range(r)) - set(routes_tabu):
            node_set = set(mapping_adjacent[routes_R[i][R_i_len]]) - set(routes_R[i]) # gets the difference in sets
            if node_set: # can be evaluated as a boolean expression   
                node_a = random.sample(node_set, 1) # gets an adjacent node to first node not in route
                routes_R[i].append(node_a[0])
                break
            else:
                routes_tabu.append(i)
                if set(range(r)) - set(routes_tabu):
                    i = random.sample(set(range(r)) - set(routes_tabu), 1)[0]
                    R_i_len = len(routes_R[i])-1 # index of the last node
                else:
                    # print("Could not add node to any last node")
                    break
                
def change_delete_node_from_front(routes_R, r): # Delete node from end of route
    # routes_R is the routes set list
    # r is the number of routes present
    i = random.randrange(r) # gets a random route position
    if len(routes_R[i]) >= 3:
        del routes_R[i][0]
    else:
        routes_tabu = [i]
        i = random.sample(set(range(r)) - set(routes_tabu), 1)[0]
        while set(range(r)) - set(routes_tabu):
            if len(routes_R[i]) >= 3:
                del routes_R[i][0]
                break
            else:
                routes_tabu.append(i)
                if set(range(r)) - set(routes_tabu):
                    i = random.sample(set(range(r)) - set(routes_tabu), 1)[0]
                else:
                    # print("Could not delete node from any first node")
                    break
     
def change_delete_node_from_back(routes_R, r): # Delete node from end of route
    # routes_R is the routes set list
    # r is the number of routes present
    i = random.randrange(r) # gets a random route position
    if len(routes_R[i]) >= 3:
        del routes_R[i][-1]
    else:
        routes_tabu = [i]
        i = random.sample(set(range(r)) - set(routes_tabu), 1)[0]
        while set(range(r)) - set(routes_tabu):
            if len(routes_R[i]) >= 3:
                del routes_R[i][-1]
                break
            else:
                routes_tabu.append(i)
                if set(range(r)) - set(routes_tabu):
                    i = random.sample(set(range(r)) - set(routes_tabu), 1)[0]
                else:
                    # print("Could not delete node from any last node")
                    break  
                
def perturb_make_small_change(routes_R, r, mapping_adjacent):
    # makes a small change to the route set
    R = copy.deepcopy(routes_R)
    p = random.uniform(0,1)
    if p < 0.25:
        change_add_node_to_first_node(R, r, mapping_adjacent)
    elif p < 0.5:
        change_add_node_to_last_node(R, r, mapping_adjacent)
    elif p < 0.75:
        change_delete_node_from_front(R, r)
    else:
        change_delete_node_from_back(R, r)
    return R    

def perturb_make_small_add_terminal(routes_R, main_problem):
    # makes a small change to the route set by adding terminal vertex
    r = main_problem.problem_constraints.con_r
    mapping_adjacent = main_problem.mapping_adjacent
    
    R = copy.deepcopy(routes_R)
    p = random.uniform(0,1)
    if p < 0.50:
        change_add_node_to_first_node(R, r, mapping_adjacent)
    else:
        change_add_node_to_last_node(R, r, mapping_adjacent)
    return R 

def perturb_make_small_del_terminal(routes_R, main_problem):
    # makes a small change to the route set by deleting terminal vertex
    r = main_problem.problem_constraints.con_r
    
    R = copy.deepcopy(routes_R)
    p = random.uniform(0,1)
    if p < 0.50:
        change_delete_node_from_front(R, r)
    else:
        change_delete_node_from_back(R, r)
        
    return R 

def test_connectedness_nx(routes_R):
    route_links = convert_paths_to_links(routes_R)
    G = nx.Graph()
    for i in range(len(routes_R)):
        G.add_edges_from(route_links[i])
    return nx.is_connected(G)

def test_connectedness_sets(routes_R, n_nodes):
    R_copy = list(routes_R)
    R_copy.sort(key = len)
    #flag = False
    
    #print(R_copy)
    for i, r_i in reversed(list(enumerate(R_copy))):
        if i == len(R_copy)-1:
            V_set = set(r_i)
            #del R_copy[i]
            #print(R_copy)
            #print(r_i)
        for j, r_j in reversed(list(enumerate(R_copy))):
            #print(i,j)
            #print(R_copy)
            #print(r_j)

            if i > j:
                if V_set.intersection(set(r_j)):
                    V_set = V_set.union(set(r_j))
                    #del r_j
                    if len(V_set) == n_nodes:
                        #print(f"final:{i} {j}")
                        #flag = True
                        return True
        # if flag == True:
        #     break
    return False

#R_x = mutated_variables[0]
#test_connectedness_sets(R_x, n_nodes=15)

def test_all_nodes_present(routes_R, N):
    all_nodes = np.zeros(N)
    r = len(routes_R)
    for i in range(r):
        all_nodes[routes_R[i]] = 1
        if all(all_nodes):
            return True
    return False

def test_all_nodes_present_set(routes_R, n_nodes):
    """The function uses sets to determine whether all nodes are present
    Note: much faster and more elegant than a for loop"""
    all_nodes = [y for x in routes_R for y in x] # flatten all the elements in route
    
    # Initial test for all nodes present:
    if (len(set(all_nodes)) == n_nodes):
        return True
    else:
        return False

def test_all_nodes_and_connectedness_nx(routes_R, N):
    all_nodes = np.zeros(N)
    r = len(routes_R)
    for i in range(r):
        all_nodes[routes_R[i]] = 1
        if all(all_nodes):
            break
    
    if all(all_nodes):
        route_links = convert_paths_to_links(routes_R)
        G = nx.Graph()
        for i in range(r):
            G.add_edges_from(route_links[i])
        return nx.is_connected(G)
    else:
        return False

# todo --- nx is faster than my own algorithm -- incorporate it
        
def test_route_feasibility(routes_R, parameters_constraints):
    # tests the route feasibility wrt number of nodes, all nodes and connectedness
    for i in range(parameters_constraints['con_r']):
        len_r = len(routes_R[i])
        if len_r < parameters_constraints['con_minNodes'] or len_r > parameters_constraints['con_maxNodes']:
            return False
    return test_all_nodes_and_connectedness_nx(routes_R, parameters_constraints['con_N_nodes'])

def get_graph_node_and_edge_connectivity(routes_R):
    route_links = convert_paths_to_links(routes_R)
    G = nx.Graph()
    for i in range(len(routes_R)):
        G.add_edges_from(route_links[i])
    return (nx.node_connectivity(G), nx.edge_connectivity(G))

def calc_min_span_tree_edge_weights(mx_dist):
    links_list_dist_mx, links_list_distances = get_links_list_and_distances(mx_dist)
    G = nx.Graph()
    for i in range(len(links_list_dist_mx)): # add edges
        G.add_edge(links_list_dist_mx[i][0], links_list_dist_mx[i][1], weight = links_list_distances[i])
    T = nx.minimum_spanning_tree(G)
    return T.size(weight='weight')

def create_nx_graph_from_routes_list(routes_R):
    """Take note: MultiGraph and NOT Graph is used, allowing multiple edges"""
    route_links = convert_paths_to_links(routes_R)
    G = nx.MultiGraph()
    for i in range(len(routes_R)):
        G.add_edges_from(route_links[i])
    return G

def create_nx_graph_from_adj_matrix(mx_dist):
    """Create a weigthed graph from adj matrix"""
    links_list_dist_mx, links_list_distances = get_links_list_and_distances(mx_dist)
    G = nx.Graph()
    for i in range(len(links_list_dist_mx)): # add edges
        G.add_edge(links_list_dist_mx[i][0], links_list_dist_mx[i][1], weight = links_list_distances[i])
    return G

def visualise_nx_multi_graph(routes_R_nx, mx_coords):
    """Input: networkx multigraph and list of coords"""
    pos = mx_coords
    
    nx.draw_networkx_nodes(routes_R_nx, pos, node_color = 'r', node_size = 100, alpha = 1)
    ax = plt.gca()
    for e in routes_R_nx.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                    ),
                                    ),
                    )
    plt.axis('off')
    plt.show()
    
# %% Feasibility tests:
def test_separate_route_lengths(routes_R, parameters_constraints):
    """Test route lenghts in terms of nodes"""
    for i in range(len(routes_R)):
        len_r = len(routes_R[i])
        if len_r < parameters_constraints['con_minNodes'] or len_r > parameters_constraints['con_maxNodes']:
            return False
        
    return True

def test_cycles_and_backtracks(routes_R):   
    for i in routes_R:
        if len(set(i)) != len(i):
            return False
            # Could customise this function to return the route giving the problem
    return True

def test_cycles_and_backtracks_sets(routes_R):
    return not any([len(set(i)) != len(i) for i in routes_R])
    

def test_all_four_constraints(routes_R, main_problem):
    """Function to test for all four constraints"""
    parameters_constraints = main_problem.problem_constraints.__dict__
    
    if not test_all_nodes_present_set(routes_R, parameters_constraints['con_N_nodes']):
        #print("Not all nodes present")
        return False
    if not test_connectedness_sets(routes_R, parameters_constraints['con_N_nodes']):
        return False
    if not test_separate_route_lengths(routes_R, parameters_constraints):
        #print("Route length violation")
        return False
    if not test_cycles_and_backtracks_sets(routes_R):
        #print("Cycle or backtracks exist")
        return False
    return True

#def test_all_constraints_fast(routes_R, main_problem):


def test_all_four_constraints_slow(routes_R, main_problem):
    """Function to test for all four constraints"""
    parameters_constraints = main_problem.problem_constraints.__dict__
    
    if not test_all_nodes_present_set(routes_R, parameters_constraints['con_N_nodes']):
        #print("Not all nodes present")
        return False
    if not test_connectedness_nx(routes_R):
        #print("All nodes not connected")
        return False
    if not test_separate_route_lengths(routes_R, parameters_constraints):
        #print("Route length violation")
        return False
    if not test_cycles_and_backtracks(routes_R):
        #print("Cycle or backtracks exist")
        return False
    return True


def test_all_four_constraints_debug(routes_R, main_problem):
    """Function to test for all four constraints and prints violation"""
    parameters_constraints = main_problem.problem_constraints.__dict__

    if not test_all_nodes_present_set(routes_R, parameters_constraints['con_N_nodes']):
        print("Not all nodes present")
        return False
    if not test_connectedness_nx(routes_R):
        print("All nodes not connected")
        return False
    if not test_cycles_and_backtracks(routes_R):
        print("Cycle or backtracks exist")
        return False
    if not test_separate_route_lengths(routes_R, parameters_constraints):
        print("Route length violation")
        return False
    return True

# %% Dominance based multi-objective Simulated Annealing functions
    
def test_min_min_non_dominated_wrong(df_archive, f1_ATT, f2_TRT):
    # Test whether the solution with f1 and f2 is a non-dominated solution in a min min problem
    return not any((df_archive.iloc[:,0] < f1_ATT) & (df_archive.iloc[:,1] < f2_TRT)) 

def count_min_min_dominating_solutions_wrong(df_archive, f1_ATT, f2_TRT):
    # Count the number of solutions that dominate solution x 
    # when both functions need to be minimised
    # NB: the df_archive should be a dataframe so that it can be easily accessed
    return sum((df_archive.iloc[:,0] < f1_ATT) & (df_archive.iloc[:,1] < f2_TRT))

def energy_function_wrong(df_archive, f1_cur, f2_cur, f1_new, f2_new):
    # The energy function for SA to compare the current and new solution
    return (count_min_min_dominating_solutions(df_archive, f1_cur, f2_cur) - \
            count_min_min_dominating_solutions(df_archive, f1_new, f2_new)) / len(df_archive) #NB should be new - cur
              
def energy_function_in_one_wrong(df_archive, f_cur, f_new):
    #if f_new[0] in df_archive.iloc[:,0]:
    return sum((df_archive.iloc[:,0] < f_cur[0]) & (df_archive.iloc[:,1] < f_cur[1])) 
    
def prob_accept_neighbour_wrong(df_archive, f_cur, f_new, SA_Temp):
    # Generates a probability of measuring acceptance for the SA algorithm
    try:
        ans = math.exp(-energy_function(df_archive, f_cur[0], f_cur[1], f_new[0], f_new[1])/SA_Temp)
    except OverflowError:
        ans = float('inf')
    return min(1, ans)

"""Edited dominance functions for perfect accuracy"""
def test_min_min_non_dominated(df_archive, f1_ATT, f2_TRT):
    # Test whether the solution with f1 and f2 is a non-dominated solution in a min min problem
    set_archive = set((x,y) for x, y in zip(df_archive.iloc[:,0], df_archive.iloc[:,1]))
    cur_sol = {(f1_ATT, f2_TRT)}
                
    if set_archive.intersection(cur_sol): # tests whether the solution is already in the archive
        return False 
    
    else:
        return not any((df_archive.iloc[:,0] < f1_ATT) & (df_archive.iloc[:,1] <= f2_TRT) | 
                       (df_archive.iloc[:,0] <= f1_ATT) & (df_archive.iloc[:,1] < f2_TRT))

def count_min_min_dominating_solutions(df_archive, f1_ATT, f2_TRT):
    # Count the number of solutions that dominate solution x 
    # when both functions need to be minimised
    # NB: the df_archive should be a dataframe so that it can be easily accessed
    return sum((df_archive.iloc[:,0] < f1_ATT) & (df_archive.iloc[:,1] <= f2_TRT) | 
               (df_archive.iloc[:,0] <= f1_ATT) & (df_archive.iloc[:,1] < f2_TRT))  
        
def energy_function(df_archive, f1_cur, f2_cur, f1_new, f2_new):
    # The energy function for SA to compare the current and new solution
    
    set_archive = set((x,y) for x, y in zip(df_archive.iloc[:,0], df_archive.iloc[:,1]))
    cur_sol = {(f1_cur, f2_cur)}
    new_sol = {(f1_new, f2_new)}
    
    df_A_union = df_archive.iloc[:,0:2]
    
    if new_sol.intersection(cur_sol):  # tests whether the two solutions are the same 
        return float('inf')
    
    else:
        if not set_archive.intersection(cur_sol): # tests whether the current solution is already in the archive
            df_A_union.loc[len(df_A_union)] = [f1_cur, f2_cur]
        if not set_archive.intersection(new_sol): # tests whether the new solution is already in the archive
            df_A_union.loc[len(df_A_union)] = [f1_new, f2_new]

    
    return (count_min_min_dominating_solutions(df_A_union, f1_new, f2_new) - \
            count_min_min_dominating_solutions(df_A_union, f1_cur, f2_cur)) / len(df_A_union)  # should be new_sol - cur_sol

def energy_function_for_initial_temp(df_archive, f1_cur, f2_cur, f1_new, f2_new):
    # The energy function for SA to compare the current and new solution
    
    set_archive = set((x,y) for x, y in zip(df_archive.iloc[:,0], df_archive.iloc[:,1]))
    cur_sol = {(f1_cur, f2_cur)}
    new_sol = {(f1_new, f2_new)}
    
    df_A_union = df_archive.iloc[:,0:2]
    
    if not set_archive.intersection(cur_sol): # tests whether the current solution is already in the archive
        df_A_union.loc[len(df_A_union)] = [f1_cur, f2_cur]
    if not set_archive.intersection(new_sol): # tests whether the new solution is already in the archive
        df_A_union.loc[len(df_A_union)] = [f1_new, f2_new]

    return (count_min_min_dominating_solutions(df_A_union, f1_new, f2_new) - \
            count_min_min_dominating_solutions(df_A_union, f1_cur, f2_cur)) / len(df_A_union)  # should be new_sol - cur_sol

 
# df_archive = copy.deepcopy(Overall_Pareto_setcsv)
# df_archive = df_archive.iloc[:,1:4]

# set_archive = set((x,y) for x, y in zip(df_archive.iloc[:,0], df_archive.iloc[:,1]))

# cur_sol = (10.5, df_archive.iloc[0,1])
# nr_dom_sols_f_cur = count_min_min_dominating_solutions(df_archive, cur_sol[0], cur_sol[1])
# nr_dom_sols_f_cur_true = count_min_min_dominating_solutions_true(df_archive, cur_sol[0], cur_sol[1])

# cur_sol = {(df_archive.iloc[0,0], df_archive.iloc[0,1])}
# union = cur_sol.union(set_archive)

# f1_cur = df_archive.iloc[0,0]
# f2_cur = df_archive.iloc[0,1]
# f1_new = df_archive.iloc[0,0]
# f2_new = df_archive.iloc[0,1]+10

# energy_function_true(df_archive, df_archive.iloc[0,0], df_archive.iloc[0,1], f1_new, f2_new)
# prob_accept_neighbour_true(df_archive, (f1_cur, f2_cur), (f1_new,f2_new), 100)

# f_cur = (22, 83)
# f_new = (20, 200)


# set_archive = set((x,y) for x, y in zip(df_archive.iloc[:,0], df_archive.iloc[:,1]))
# cur_sol = {(22, 83)}
# new_sol = {(21, 84)}


# nr_dom_sols_f_cur = count_min_min_dominating_solutions(df_archive, f_cur[0], f_cur[1])
# nr_dom_sols_f_new = count_min_min_dominating_solutions(df_archive, f_new[0], f_new[1])
# energy = energy_function(df_archive, f_cur[0], f_cur[1], f_new[0], f_new[1])
# prob_accept_neighbour_true(df_archive, f_cur, f_new, 5)



def prob_accept_neighbour(df_archive, f_cur, f_new, SA_Temp):
    # Generates a probability of measuring acceptance for the SA algorithm
    try:
        ans = math.exp(-energy_function(df_archive, f_cur[0], f_cur[1], f_new[0], f_new[1])/SA_Temp)
    except OverflowError:
        ans = float('inf')
    return min(1, ans)

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
        Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

''' Normalise data '''
def normalise_first_two_columns(df):
    df_copy = df.copy()
    max_objs = df_copy.iloc[:,0:2].max()
    min_objs = df_copy.iloc[:,0:2].min()
    df_copy.iloc[:,0:2] = (df_copy.iloc[:,0:2] - min_objs)/(max_objs - min_objs) # normalise the data
    return df_copy

def normalise_first_two_columns_by_ref_point(df, max_objs, min_objs):
    # NB, max_objs and min_objs should be 2D np.array([])
    df_copy = df.copy()
    df_copy.iloc[:,0:2] = (df_copy.iloc[:,0:2].values - min_objs)/(max_objs - min_objs) # normalise the data
    return df_copy

def normalise_first_two_columns_by_ref_point2(df, max_objs, min_objs):
    # NB, max_objs and min_objs should be pandas series objects obtained by df.iloc[:,0:2].max()
    df_copy = df.copy()
    df_copy.iloc[:,0:2] = (df_copy.iloc[:,0:2] - min_objs)/(max_objs - min_objs) # normalise the data
    return df_copy

def norm_and_calc_2d_hv(df_norm, max_objs, min_objs):
    """Use this when input values are dataframe values"""
    df_norm_copy = (df_norm.values - min_objs)/(max_objs - min_objs) # normalise the data
    # take note, the df has now changed to a numpy array
    df_norm_copy[df_norm_copy < 0] = 0 # to avoid errors in HV computation
    df_norm_copy[df_norm_copy > 1] = 1 # to avoid errors in HV computation
        
    hv = get_performance_indicator("hv", ref_point=np.array([1, 1])) # create the HV object
    return hv.calc(df_norm_copy) # assume minimisation and compute
    
def norm_and_calc_2d_hv_np(numpy_objs, max_objs, min_objs):
    """Use this when input values are numpy arrays"""
    numpy_objs_copy = (numpy_objs - min_objs)/(max_objs - min_objs) # normalise the data

    # take note, the df has now changed to a numpy array
    numpy_objs_copy[numpy_objs_copy < 0] = 0 # to avoid errors in HV computation
    numpy_objs_copy[numpy_objs_copy > 1] = 1 # to avoid errors in HV computation
      
    hv = get_performance_indicator("hv", ref_point=np.array([1, 1]))        
    return hv.calc(numpy_objs_copy) # assume minimisation and compute
    
''' Calculate Hypervolume using Pygmo ''' 
def calc_hv_from_df(df_archive):
    points = list()
    for i in range(len(df_archive)):
        points.append([df_archive.iloc[i,0], df_archive.iloc[i,1]])
    
    hv = pg.hypervolume(points = points)
    ref_point = hv.refpoint()
    return hv.compute(ref_point)

def calc_hv_from_df_from_ref_point(df_archive, ref_point):
    points = list()
    for i in range(len(df_archive)):
        points.append([df_archive.iloc[i,0], df_archive.iloc[i,1]])
    
    hv = pg.hypervolume(points = points)
    return hv.compute(ref_point)

def calc_hv_from_normalised_df(df_archive_norm):
    df_archive_norm[df_archive_norm < 0] = 0
    df_archive_norm[df_archive_norm > 1] = 1
    
    points = list()
    for i in range(len(df_archive_norm)):
        points.append([df_archive_norm.iloc[i,0], df_archive_norm.iloc[i,1]])
    
    hv = pg.hypervolume(points = points)
    return hv.compute((1,1)) # assume minimisation

def calc_hv_from_normalised_df2(df_archive_norm):
    # send in only the values and not the routes
    df_archive_norm[df_archive_norm < 0] = 0
    df_archive_norm[df_archive_norm > 1] = 1
    
    points = df_archive_norm.iloc[:,0:2].values
    
    hv = pg.hypervolume(points = points)
    return hv.compute((1,1)) # assume minimisation

'''Own function to calculate Hypervolume'''
def calc_hypervolume_from_archive(df_archive, ref_point):
    # ref_point should be a tuple
    df_archive_sorted = df_archive.sort_values(by='f1_ATT')
    df_archive_sorted = df_archive_sorted.loc[(df_archive_sorted['f1_ATT'] < ref_point[0]) & (df_archive_sorted['f2_TRT'] < ref_point[1])]
    vector_holder = np.zeros(len(df_archive_sorted)) # create a vector that can help vectorise calculations of HV
    vector_holder[0] = ref_point[1] 
    vector_holder[1:] = df_archive_sorted.iloc[0:-1,1].values
    
    return sum((ref_point[0] - df_archive_sorted.iloc[:,0].values)*(vector_holder - df_archive_sorted.iloc[:,1].values))

# %% GA for UTRP

# %% Crowding distance
def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-24)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding


def rank_from_fronts(fronts, n):
    # create the rank array and set values
    rank = np.full(n, 1e16, dtype=np.int64)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank


# %% Binary tournament Selection
def binary_tournament_g2(pop, P, tournament_type='comp_by_dom_and_crowding', **kwargs):
    """Note: P is random permutation combination array"""
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    # tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        # if pop[a].CV > 0.0 or pop[b].CV > 0.0:
        #     S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)
        if False: # placeholder
            pass
        
        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(pop.objectives[a,], pop.objectives[b,])
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, pop.rank[a,0], b, pop.rank[b,0],
                               method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop.crowding_dist[a,0], b, pop.crowding_dist[b,0],
                               method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int64, copy=False)

def get_survivors(pop, n_survive, D=None, **kwargs):

    # get the objective space values and objects
    # F = pop.get("F").astype(np.float, copy=False)
    F = pop.objectives

    # the final indices of surviving individuals
    survivors = []

    # do the non-dominated sorting until splitting front
    fronts = gc.NonDominated_Sorting().do(F, n_stop_if_ranked=n_survive)

    for k, front in enumerate(fronts):

        # calculate the crowding distance of the front
        crowding_of_front = calc_crowding_distance(F[front, :])

        # save rank and crowding in the individual class
        for j, i in enumerate(front):
            pop.rank[i] = k
            pop.crowding_dist[i] = crowding_of_front[j]
            
        # current front sorted by crowding distance if splitting
        if len(survivors) + len(front) > n_survive:
            I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
            I = I[:(n_survive - len(survivors))]

        # otherwise take the whole front unsorted
        else:
            I = np.arange(len(front))

        # extend the survivors by all or selected individuals
        survivors.extend(front[I])

    # return pop[survivors]
    return survivors

def get_survivors_norm(pop, n_survive, D=None, **kwargs):

    # get the objective space values and objects
    # F = pop.get("F").astype(np.float, copy=False)
    F = pop.objs_norm

    # the final indices of surviving individuals
    survivors = []

    # do the non-dominated sorting until splitting front
    fronts = gc.NonDominated_Sorting().do(F, n_stop_if_ranked=n_survive)

    for k, front in enumerate(fronts):

        # calculate the crowding distance of the front
        crowding_of_front = calc_crowding_distance(F[front, :])

        # save rank and crowding in the individual class
        for j, i in enumerate(front):
            pop.rank[i] = k
            pop.crowding_dist[i] = crowding_of_front[j]
            
        # current front sorted by crowding distance if splitting
        if len(survivors) + len(front) > n_survive:
            I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
            I = I[:(n_survive - len(survivors))]

        # otherwise take the whole front unsorted
        else:
            I = np.arange(len(front))

        # extend the survivors by all or selected individuals
        survivors.extend(front[I])

    # return pop[survivors]
    return survivors

# %% Tournament selection
def tournament_selection_g2(pop, n_select, n_parents=2, pressure=2):
    # pop should be in Günther's format: pop.objectives -> array
    # number of random individuals needed
    n_random = n_select * n_parents * pressure
    
    # number of permutations needed
    n_perms = math.ceil(n_random / pop.population_size)
    
    # get random permutations and reshape them
    P = random_permuations(n_perms, pop.population_size)[:n_random]
    P = np.reshape(P, (n_select * n_parents, pressure))
    
    # compare using tournament function
    S = binary_tournament_g2(pop, P)
    
    return np.reshape(S, (n_select, n_parents))


def pop_from_array_or_individual(array, pop=None):
    # the population type can be different - (different type of individuals)
    if pop is None:
        pop = Population()

    # provide a whole population object - (individuals might be already evaluated)
    if isinstance(array, Population):
        pop = array
    elif isinstance(array, np.ndarray):
        pop = pop.new("X", np.atleast_2d(array))
    elif isinstance(array, Individual):
        pop = Population(1)
        pop[0] = array
    else:
        return None

    return pop




def keep_individuals(pop, survivor_indices):
    # Function that only keeps to individuals with the specified indices
    pop.variables_str = [pop.variables_str[i] for i in survivor_indices]
    pop.variables = [pop.variables[i] for i in survivor_indices]
    pop.objectives = pop.objectives[survivor_indices,] 
    pop.objs_norm = pop.objs_norm[survivor_indices,]  
    pop.rank = pop.rank[survivor_indices]
    pop.crowding_dist = pop.crowding_dist[survivor_indices]

# %% Repair functions
"""Repair functions"""

def repair_add_missing_from_terminal(routes_R, main_problem):
    """ A function that searches for all the missing nodes, and tries to connect 
    them with one route's terminal node """
    n_nodes = main_problem.problem_inputs.n
    mapping_adjacent = main_problem.mapping_adjacent                                                                         
    
    all_nodes = [y for x in routes_R for y in x] # flatten all the elements in route
    
    # Initial test for all nodes present:
    if (len(set(all_nodes)) != n_nodes): # if not true, go on to testing for what nodes are ommited
    
        missing_nodes = list(set(range(n_nodes)).difference(set(all_nodes))) # find all the missing nodes
        random.shuffle(missing_nodes) # shuffles the nodes for randomness
        
        for missing_node in missing_nodes:
            # Find adjacent nodes of the missing nodes
            adj_nodes = mapping_adjacent[missing_node] # get the required nodes that are adjacent to 
            
            # Get terminal nodes
            terminal_nodes_front = [x[0] for x in routes_R] # get all the terminal nodes in the first position
            terminal_nodes_back = [x[-1] for x in routes_R] # get all the terminal nodes in the last position
            
            # Two cases, one from the front and one from the back
            terminal_nodes_front_candidates = set(terminal_nodes_front).intersection(adj_nodes) # Find intersection between first terminal nodes and adj nodes
            terminal_nodes_back_candidates = set(terminal_nodes_back).intersection(adj_nodes) # Find intersection between last terminal nodes and adj nodes
    
            if random.random() < 0.5:   # adds randomness to either front or back, and not just always one direction
                if bool(terminal_nodes_front_candidates):
                    random_adj_node = random.sample(terminal_nodes_front_candidates, 1)[0]
                    terminal_nodes_front.index(random_adj_node)
                    # Insert missing node at the front of adjacent terminal node
                    routes_R[terminal_nodes_front.index(random_adj_node)].insert(0, missing_node) 
                    
                elif bool(terminal_nodes_back_candidates):
                    random_adj_node = random.sample(terminal_nodes_back_candidates, 1)[0]
                    # Insert missing node at the back of adjacent terminal node
                    routes_R[terminal_nodes_back.index(random_adj_node)].append(missing_node) 
                    
            else:
                if bool(terminal_nodes_back_candidates):
                    random_adj_node = random.sample(terminal_nodes_back_candidates, 1)[0]
                    # Insert missing node at the back of adjacent terminal node
                    routes_R[terminal_nodes_back.index(random_adj_node)].append(missing_node) 
                
                elif bool(terminal_nodes_front_candidates):
                    random_adj_node = random.sample(terminal_nodes_front_candidates, 1)[0]
                    terminal_nodes_front.index(random_adj_node)
                    # Insert missing node at the front of adjacent terminal node
                    routes_R[terminal_nodes_front.index(random_adj_node)].insert(0, missing_node) 
                    
    return routes_R


def find_path_from_dist_list(path, distance_list_vertex_u, distance_depth, mapping_adjacent):
    """A recursive function to find the path from the closest terminal vertex to the missing vertex, given a list
    of different distances each vertex in the graph is away from the missing vertex. 
    Input:
        path: list that is created by adding the shortest path in distance, initiate the first list with only the starting vertex
        distance_list_vertex_u: a list of lists where entry 0 is the missing vertex, and entry 1 the vertices with distance 1 to the missing vertex, etc.
        distance_depth: the current distance depth in the distance_list_vertex_u used to find the next set of adjacent vertices
        mapping_adjacent: a list where mapping_adjacent[i] is the vertices adjacent to vertex i
    Output:
        path: the input path appended with an adjacent vertex one distance less to the missing vertex than the former vertex
        distance_depth: the updated distance depth
        """
    vertex_j = path[-1]
    adj_possibilities = set(mapping_adjacent[vertex_j]).intersection(set(distance_list_vertex_u[distance_depth-1]))
    vertex_i = random.choice(tuple(adj_possibilities))
    path.append(vertex_i)
    
    distance_depth = distance_depth - 1
    
    if distance_depth == 0: 
        return path, distance_depth
    
    else:    
        path, distance_depth = find_path_from_dist_list(path, distance_list_vertex_u, distance_depth, mapping_adjacent)
        return path, distance_depth


def repair_add_missing_from_terminal_multiple_debug(routes_R, UTNDP_problem):
    """ A function that searches for all the missing nodes, and tries to connect 
    them with one route's terminal node by trying to add one or more vertices to terminals"""
    
    max_depth = UTNDP_problem.problem_constraints.con_maxNodes - UTNDP_problem.problem_constraints.con_minNodes
    
    all_nodes = [y for x in routes_R for y in x] # flatten all the elements in route
    
    # Initial test for all nodes present:
    if (len(set(all_nodes)) != UTNDP_problem.problem_constraints.con_N_nodes): # if not true, go on to testing for what nodes are ommited
        
        missing_nodes = list(set(range(UTNDP_problem.problem_constraints.con_N_nodes)).difference(set(all_nodes))) # find all the missing nodes
        random.shuffle(missing_nodes) # shuffles the nodes for randomness
    
        for missing_node in missing_nodes:
            
            all_nodes = [y for x in routes_R for y in x] # flatten all the elements in the updated route set
            if missing_node in all_nodes: # test whether the missing node was already included, and continue loop
                continue
            
            distance_list_vertex_u = get_graph_distance_levels_from_vertex_u(missing_node,max_depth,UTNDP_problem.mapping_adjacent)
            print(f"\nMissing node: {missing_node} | Missing nodes: {missing_nodes}")
            # Get terminal nodes
            terminal_nodes_front = [x[0] for x in routes_R] # get all the terminal nodes in the first position
            terminal_nodes_back = [x[-1] for x in routes_R] # get all the terminal nodes in the last position
            terminal_nodes_all = terminal_nodes_front + terminal_nodes_back
    
    
            for distance_depth in range(len(distance_list_vertex_u)-1):
                intersection_terminal_dist_list = list(set(terminal_nodes_all).intersection(set(distance_list_vertex_u[distance_depth+1])))
                print(f"Dist depth: {distance_depth} | Terminal intersection dist: {intersection_terminal_dist_list}")
    
                if intersection_terminal_dist_list:
                    random.shuffle(intersection_terminal_dist_list) # shuffles the terminal nodes list for randomness
                    
                    for random_terminal_node in intersection_terminal_dist_list:
                    
                        path_to_add = find_path_from_dist_list([random_terminal_node], distance_list_vertex_u, distance_depth+1, UTNDP_problem.mapping_adjacent)[0]
                        # print(path_to_add)
        
                        """Adds the connecting path to the correct route"""  
                        # Finds all the routes with the correct terminal vertex for the connection, ensuring no infeasible connections
                        feasible_routes_terminal_vertex = [path_to_add[0]==x for x in terminal_nodes_all]
                        #feasible_routes_terminal_vertex = [feasible_routes_terminal_vertex[i] or feasible_routes_terminal_vertex[i+len(routes_R)] for i in range(len(routes_R))]
                        
                        # Finds all the routes that do not contain the path in itself, avoiding cycles
                        feasible_routes_no_cycle = [not bool(y) for y in [set(path_to_add[1:]).intersection(set(x)) for x in routes_R]] # determine which routes do not contain the path
                        feasible_routes_no_cycle = feasible_routes_no_cycle + feasible_routes_no_cycle
        
        
                        # Finds all possible routes to merge with
                        feasible_routes_to_add_path = [feasible_routes_terminal_vertex[i] and feasible_routes_no_cycle[i] for i in range(len(feasible_routes_terminal_vertex))]
                        
                        # Determines all the feasible indices the path may be added to
                        add_path_indices = [i for i, x in enumerate(feasible_routes_to_add_path) if x]
                        
                        if add_path_indices: # tests whether any legal additions exists, may be that a previous addition included more missing nodes
                            add_path_index = random.choice(add_path_indices)
                            if add_path_index < len(routes_R):
                                """Adds the path to the front end of a route"""
                                path_to_add.reverse()
                                routes_R[add_path_index] = path_to_add[:-1] + routes_R[add_path_index]
                                
                                print(f"....Path added: {path_to_add[:-1]}")
                            else:
                                """Adds the path to the back end of a route"""
                                routes_R[add_path_index - len(routes_R)].extend(path_to_add[1:]) 
                            
                                print(f"....Path added: {path_to_add[1:]}")
                            break
                        
                    else:
                        continue # continue when inner loop was not broken
                    
                    break # inner loop was broken, then break the outer loop too

    return routes_R

def repair_add_missing_from_terminal_multiple(routes_R, UTNDP_problem):
    """ A robust function that searches for all the missing nodes, and tries to connect 
    them with one route's terminal node by trying to add one or more vertices to terminal vertices"""
    
    max_depth = UTNDP_problem.problem_constraints.con_maxNodes - UTNDP_problem.problem_constraints.con_minNodes
    
    all_nodes = [y for x in routes_R for y in x] # flatten all the elements in route
    
    # Initial test for all nodes present:
    if (len(set(all_nodes)) != UTNDP_problem.problem_constraints.con_N_nodes): # if not true, go on to testing for what nodes are ommited
        
        missing_nodes = list(set(range(UTNDP_problem.problem_constraints.con_N_nodes)).difference(set(all_nodes))) # find all the missing nodes
        random.shuffle(missing_nodes) # shuffles the nodes for randomness
    
        for missing_node in missing_nodes:
            
            all_nodes = [y for x in routes_R for y in x] # flatten all the elements in the updated route set
            if missing_node in all_nodes: # test whether the missing node was already included, and continue loop
                continue
            
            distance_list_vertex_u = get_graph_distance_levels_from_vertex_u(missing_node,max_depth,UTNDP_problem.mapping_adjacent)
            
            # Get terminal nodes
            terminal_nodes_all = [x[0] for x in routes_R] + [x[-1] for x in routes_R] # get all the terminal nodes in the first and last positions
    
    
            for distance_depth in range(len(distance_list_vertex_u)-1):
                intersection_terminal_dist_list = list(set(terminal_nodes_all).intersection(set(distance_list_vertex_u[distance_depth+1])))
    
                if intersection_terminal_dist_list:
                    random.shuffle(intersection_terminal_dist_list) # shuffles the terminal nodes list for randomness
                    
                    for random_terminal_node in intersection_terminal_dist_list:
                    
                        path_to_add = find_path_from_dist_list([random_terminal_node], distance_list_vertex_u, distance_depth+1, UTNDP_problem.mapping_adjacent)[0]
        
                        """Adds the connecting path to the correct route"""  
                        # Finds all the routes with the correct terminal vertex for the connection, ensuring no infeasible connections
                        feasible_routes_terminal_vertex = [path_to_add[0]==x for x in terminal_nodes_all]
                        #feasible_routes_terminal_vertex = [feasible_routes_terminal_vertex[i] or feasible_routes_terminal_vertex[i+len(routes_R)] for i in range(len(routes_R))]
                        
                        # Finds all the routes that do not contain the path in itself, avoiding cycles
                        feasible_routes_no_cycle = [not bool(y) for y in [set(path_to_add[1:]).intersection(set(x)) for x in routes_R]] # determine which routes do not contain the path
                        feasible_routes_no_cycle = feasible_routes_no_cycle + feasible_routes_no_cycle
        
        
                        # Finds all possible routes to merge with
                        feasible_routes_to_add_path = [feasible_routes_terminal_vertex[i] and feasible_routes_no_cycle[i] for i in range(len(feasible_routes_terminal_vertex))]
                        
                        # Determines all the feasible indices the path may be added to
                        add_path_indices = [i for i, x in enumerate(feasible_routes_to_add_path) if x]
                        
                        if add_path_indices: # tests whether any legal additions exists, may be that a previous addition included more missing nodes
                            add_path_index = random.choice(add_path_indices)
                            if add_path_index < len(routes_R):
                                """Adds the path to the front end of a route"""
                                path_to_add.reverse()
                                routes_R[add_path_index] = path_to_add[:-1] + routes_R[add_path_index]
                                                               
                            else:
                                """Adds the path to the back end of a route"""
                                routes_R[add_path_index - len(routes_R)].extend(path_to_add[1:]) 
                            
                            break
                        
                    else:
                        continue # continue when inner loop was not broken
                    
                    break # inner loop was broken, then break the outer loop too

    return routes_R

def repair_add_path_to_route_set_ksp(route_to_repair, main_problem, k_shortest_paths_all):
    """A function that attempts to repair a route set that had one path removed
    by identifying the missing nodes and by attempting to maximise the coverage
    of the missing nodes by looking through the K-shortest paths list"""
    ksp = k_shortest_paths_all
    n_nodes = len(main_problem.mapping_adjacent)    
    all_nodes = [y for x in route_to_repair for y in x] # flatten all the elements in route
    
    # Initial test for all nodes present:
    if (len(set(all_nodes)) != n_nodes): # if not true, go on to testing for what nodes are ommited
        missing_nodes = list(set(range(n_nodes)).difference(set(all_nodes))) # find all the missing nodes
    
        indices_of_compatible_routes = []
        for path_index in range(len(ksp)): #TODO: Can be optimised
            if set(missing_nodes).issubset(set(ksp[path_index])):
                indices_of_compatible_routes.append(path_index)
        
        routes_to_search = [ksp[i] for i in indices_of_compatible_routes]
        repaired_route = add_path_satisfying_max_unmet_demand(route_to_repair, main_problem, routes_to_search)
        
        return repaired_route
    
    else:
        path_to_add = random.choice(ksp)
        repaired_route = copy.deepcopy(route_to_repair)
        repaired_route.extend([path_to_add])
        
        return repaired_route
    
def add_path_satisfying_max_unmet_demand(route_to_repair, main_problem, routes_to_search=False): 
    '''Function for adding a path to a route set r_i that has one less route than 
    the total number of routes requirement, based on maximising unmet demand.
    Routes to search is by default k-shortest paths, but another set of routes 
    may be imported and used to search for the best demand.'''
    print("add_path_satisfying_max_unmet_demand")
    if not routes_to_search:
        routes_to_search = main_problem.k_short_paths.paths
    mx_demand = main_problem.problem_data.mx_demand
    mx_demand_unmet = remove_cum_demand_route_set(route_to_repair, mx_demand)

    d_max = 0
    pot_route_indices = [] # a list of the indices of potential routes that may suffice
    
    for i in range(len(routes_to_search)):
        d_met = calc_cum_demand(routes_to_search[i], mx_demand_unmet)
        
        if d_met == d_max:
            pot_route_indices.extend([i])
        if d_met > d_max:
            pot_route_indices = [i]
            d_max = d_met

    path_to_add = routes_to_search[random.choice(pot_route_indices)]
    # print(f"{path_to_add}")
    repaired_route = copy.deepcopy(route_to_repair)
    repaired_route.extend([path_to_add])
    
    return repaired_route

def add_path_max_unmet_demand_limited_len(route_to_repair, main_problem, removed_path, routes_to_search=False): 
    '''Function for adding a path to a route set r_i that has one less route than 
    the total number of routes requirement, based on maximising unmet demand
    while also inserting a route that has a length on one more than the removed 
    route so that it won't prefer moves benefiting passenger cost more.
    Routes to search is by default k-shortest paths, but another set of routes 
    may be imported and used to search for the best demand.'''
    # NB: Need to make this function probabilistic. Putting the same route into set
    
    
    len_removed = len(removed_path)
    
    if not routes_to_search:
        routes_to_search = main_problem.k_short_paths.paths
    mx_demand = main_problem.problem_data.mx_demand
    mx_demand_unmet = remove_cum_demand_route_set(route_to_repair, mx_demand)

    d_max = 0
    pot_route_indices = [] # a list of the indices of potential routes that may suffice
    
    for i in range(len(routes_to_search)):
        if len(routes_to_search[i]) <= len_removed + 1:
            d_met = calc_cum_demand(routes_to_search[i], mx_demand_unmet)
            
            if d_met == d_max:
                pot_route_indices.extend([i])
            if d_met > d_max:
                pot_route_indices = [i]
                d_max = d_met

    path_to_add = routes_to_search[random.choice(pot_route_indices)]
    # print(f"{path_to_add}")
    repaired_route = copy.deepcopy(route_to_repair)
    repaired_route.extend([path_to_add])
    
    return repaired_route

def add_path_prob_unmet_demand_limited_len(route_to_repair, main_problem, removed_path, routes_to_search=False): 
    '''Function for adding a path to a route set r_i that has one less route than 
    the total number of routes requirement, based on probability of meeting max unmet demand
    while also inserting a route that has a length on one more than the removed 
    route so that it won't prefer moves benefiting passenger cost more.
    Routes to search is by default k-shortest paths, but another set of routes 
    may be imported and used to search for the best demand.'''
    
    len_removed = len(removed_path)
    
    if not routes_to_search:
        routes_to_search = main_problem.k_short_paths.paths
    mx_demand = main_problem.problem_data.mx_demand
    mx_demand_unmet = remove_cum_demand_route_set(route_to_repair, mx_demand)

    d_max = 0
    d_routes = np.zeros((len(routes_to_search),1))
    pot_route_indices = [] # a list of the indices of potential routes that may suffice
    
    for i in range(len(routes_to_search)):
        if len(routes_to_search[i]) <= len_removed + 1:
            d_met = calc_cum_demand(routes_to_search[i], mx_demand_unmet)
            d_routes[i,0] = d_met
            
            if d_met == d_max:
                pot_route_indices.extend([i])
            if d_met > d_max:
                pot_route_indices = [i]
                d_max = d_met
    
    if sum(d_routes) == 0:
        prob_d_routes = np.ones((len(d_routes), 1))*(1/len(d_routes))
    else:
        prob_d_routes = d_routes / sum(d_routes)
    
    path_to_add = random.choices(routes_to_search, weights=prob_d_routes, k=1)[0]
    
    # print(f"{path_to_add}")
    repaired_route = copy.deepcopy(route_to_repair)
    repaired_route.extend([path_to_add])
    
    return repaired_route

def replace_path_prob_unmet_demand_limited_len(route_to_replace, main_problem, replace_index, routes_to_search=False, limit_len=True): 
    '''Function for replacing a path in a route set r_i that at index i, 
    based on probability of meeting max unmet demand
    while also inserting a route that has a length on one more than the removed 
    route so that it won't prefer moves benefiting passenger cost more.
    Routes to search is by default k-shortest paths, but another set of routes 
    may be imported and used to search for the best demand.'''

    debug = False
    route_copy = copy.deepcopy(route_to_replace)
    replace_path = route_copy[replace_index]   
    if limit_len: 
        len_removed = len(replace_path)
    else:
        len_removed = np.inf
    
    del route_copy[replace_index] # removes the route that should be replaced
    
    if not routes_to_search:
        routes_to_search = main_problem.k_short_paths.paths # retrieves ksp as default
    mx_demand = main_problem.problem_data.mx_demand
    mx_demand_unmet = remove_cum_demand_route_set(route_copy, mx_demand) # calc unmet demand mx

    d_max = 0
    d_routes = np.zeros((len(routes_to_search),1)) # demand met by routes
    pot_route_indices = [] # a list of the indices of potential routes that may suffice
    
    for i in range(len(routes_to_search)):
        if len(routes_to_search[i]) <= len_removed + 1:
            d_met = calc_cum_demand(routes_to_search[i], mx_demand_unmet)
            d_routes[i,0] = d_met
            
            # calc max demand met
            if d_met == d_max:
                pot_route_indices.extend([i])
            if d_met > d_max:
                pot_route_indices = [i]
                d_max = d_met
    
    # use demand met proportions to select route for inclusion
    if sum(d_routes) == 0:
        prob_d_routes = np.ones((len(d_routes), 1))*(1/len(d_routes))
    else:
        prob_d_routes = d_routes / sum(d_routes)
    
    path_to_add = random.choices(routes_to_search, weights=prob_d_routes, k=1)[0]
    
    if debug: print(f"{path_to_add}")
    route_copy.insert(replace_index, path_to_add)
    
    return route_copy

def replace_path_prob_demand_per_cost(route_to_replace, main_problem, replace_index, routes_to_search=False): 
    '''Function for replacing a path in a route set r_i that at index i, 
    based on probability of meeting max unmet demand
    while also inserting a route that has a length on one more than the removed 
    route so that it won't prefer moves benefiting passenger cost more.
    Routes to search is by default k-shortest paths, but another set of routes 
    may be imported and used to search for the best demand.'''

    debug = False
    route_copy = copy.deepcopy(route_to_replace)  
    
    del route_copy[replace_index] # removes the route that should be replaced
    
    if not routes_to_search:
        routes_to_search = main_problem.k_short_paths.paths # retrieves ksp as default
    mx_demand = main_problem.problem_data.mx_demand
    mx_demand_unmet = remove_cum_demand_route_set(route_copy, mx_demand) # calc unmet demand mx
    
    mx_dist = main_problem.problem_data.mx_dist
    c_routes = calc_seperate_route_length(routes_to_search, mx_dist).reshape(((len(routes_to_search),1)))

    d_max = 0
    d_routes = np.zeros((len(routes_to_search),1)) # demand met by routes
    pot_route_indices = [] # a list of the indices of potential routes that may suffice
    
    for i in range(len(routes_to_search)):
        d_met = calc_cum_demand(routes_to_search[i], mx_demand_unmet)
        d_routes[i,0] = d_met
        
        # calc max demand met
        if d_met == d_max:
            pot_route_indices.extend([i])
        if d_met > d_max:
            pot_route_indices = [i]
            d_max = d_met
    dem_per_cost = d_routes / c_routes
    
    # use demand met proportions to select route for inclusion
    if sum(dem_per_cost) == 0:
        prob_routes = np.ones((len(dem_per_cost), 1))*(1/len(dem_per_cost))
    else:
        prob_routes = dem_per_cost / sum(dem_per_cost)
    
    path_to_add = random.choices(routes_to_search, weights=prob_routes, k=1)[0]
    
    if debug: print(f"{path_to_add}")
    route_copy.insert(replace_index, path_to_add)
    
    return route_copy

# %% Crossover functions

def crossover_routes_random(parent_i, parent_j):
    """Crossover function for routes that randomly mixes up routes from parents
    Take note: This is a bad crossover strategy as feasibility breaks easily"""
    parents_len = len(parent_i)
    n_crossover_routes = parents_len//2 # floor division rounds down to easily split list even when uneven length
    parent_i_indices = random.sample(range(parents_len), n_crossover_routes)
    parent_j_indices = random.sample(range(parents_len), n_crossover_routes)
    
    child_i = copy.deepcopy(parent_i) # create children from parents
    child_j = copy.deepcopy(parent_j) # create children from parents
    
    for i, j in zip(parent_i_indices, parent_j_indices):
        child_i[i] = parent_j[j]
        child_j[j] = parent_i[i]
        
    return child_i, child_j

def looped_incrementor(current, limit): 
    """A function that helps to alternate between numbers"""
    if current != limit: current += 1
    else: current = 0
    return current

def crossover_routes_unseen_prob(parent_i, parent_j):
    """Crossover function for routes based on Mumford 2013's Crossover function
    for routes based on alternating between parents and including a route from
    each parent that maximises the unseen vertices added to the child route
    Note: only generates one child, needs to be tested for feasibility and repaired if needed"""
    parents = []  
    parents.append(copy.deepcopy(parent_i))
    parents.append(copy.deepcopy(parent_j))
    parent_index = random.randint(0,1) # counter to help alternate between parents  
    
    child_i = [] # define child
    parent_len = len(parent_i)
    
    # Randomly select the first seed solution for the child
    random_index = random.randint(0,parent_len-1)
    child_i.append(parents[parent_index][random_index]) # adds seed solution to parent
    del(parents[parent_index][random_index]) # removes the route from parent so that it is not evaluated again
    
    # Alternates parent solutions
    parent_index = looped_incrementor(parent_index, 1)
    
    
    # Calculate the unseen proportions to select next route for inclusion into child
    while len(child_i) < parent_len:
        # Determines all nodes present in the child
        all_nodes_present = set([y for x in child_i for y in x]) # flatten all the elements in child
    
        parent_curr = parents[parent_index] # sets the current parent
        
        proportions = []
        for i_candidate in range(len(parent_curr)):
            R_i = set(parent_curr[i_candidate])
            if bool(R_i.intersection(all_nodes_present)): # test whether there is a common transfer point
                proportions.append(len(R_i - all_nodes_present) / len(R_i)) # calculate the proportion of unseen vertices
            else:
                proportions.append(0) # set proportion to zero so that it won't be chosen
        
        # Get route that maximises the proportion of unseen nodes included
        max_indices = set([i for i, j in enumerate(proportions) if j == max(proportions)]) # position of max proportion/s
        max_index = random.sample(max_indices, 1)[0] # selects only one index randomly between a possible tie, else the only one
        
        # Add the route to the child
        child_i.append(parent_curr[max_index]) # add max proportion unseen nodes route to the child
        del(parents[parent_index][max_index]) # removes the route from parent so that it is not evaluated again
        
        # Alternates parent solutions
        parent_index = looped_incrementor(parent_index, 1)
    
    return child_i

def crossover_routes_unseen_probabilistic(parent_i, parent_j):
    """Crossover function for routes based on Mumford 2013's Crossover function
    for routes based on alternating between parents and including a route from
    each parent that maximises the unseen vertices added to the child route probabilisticly
    Note: only generates one child, needs to be tested for feasibility and repaired if needed"""
    parents = []  
    parents.append(copy.deepcopy(parent_i))
    parents.append(copy.deepcopy(parent_j))
    parent_index = random.randint(0,1) # counter to help alternate between parents  
    
    child_i = [] # define child
    parent_len = len(parent_i)
    
    # Randomly select the first seed solution for the child
    random_index = random.randint(0,parent_len-1)
    child_i.append(parents[parent_index][random_index]) # adds seed solution to parent
    del(parents[parent_index][random_index]) # removes the route from parent so that it is not evaluated again
    
    # Alternates parent solutions
    parent_index = looped_incrementor(parent_index, 1)
    
    
    # Calculate the unseen proportions to select next route for inclusion into child
    while len(child_i) < parent_len:
        # Determines all nodes present in the child
        all_nodes_present = set([y for x in child_i for y in x]) # flatten all the elements in child
    
        parent_curr = parents[parent_index] # sets the current parent
        
        proportions = []
        for i_candidate in range(len(parent_curr)):
            R_i = set(parent_curr[i_candidate])
            if bool(R_i.intersection(all_nodes_present)): # test whether there is a common transfer point
                proportions.append(len(R_i - all_nodes_present) / len(R_i)) # calculate the proportion of unseen vertices
            else:
                proportions.append(0) # set proportion to zero so that it won't be chosen
        
        # Get route that maximises the proportion of unseen nodes included
        if sum(proportions) == 0:
            max_indices = [i for i, j in enumerate(proportions)] # position of all proportion/s
            max_index = random.sample(max_indices, 1)[0] # selects only one index randomly between a possible tie, else the only one

        else:
            max_indices = [i for i, j in enumerate(proportions)] # position of max proportion/s
            proportions = np.array(proportions)/sum(proportions)     
            max_index = random.choices(max_indices, weights=proportions, k=1)[0]
            
        # Add the route to the child
        child_i.append(parent_curr[max_index]) # add max proportion unseen nodes route to the child
        del(parents[parent_index][max_index]) # removes the route from parent so that it is not evaluated again
        
        # Alternates parent solutions
        parent_index = looped_incrementor(parent_index, 1)
    
    return child_i



def crossover_routes_unseen_prob_UTRFSP(parent_i_route, parent_j_route, parent_i_freq_args, parent_j_freq_args):
    """Crossover function for routes based on Mumford 2013's Crossover function
    for routes based on alternating between parents and including a route from
    each parent that maximises the unseen vertices added to the child route
    Note: only generates one child, needs to be tested for feasibility and repaired if needed"""
    parents_route = []  
    parents_route.append(copy.deepcopy(parent_i_route))
    parents_route.append(copy.deepcopy(parent_j_route))
    parents_freq_args = []
    parents_freq_args.append(copy.deepcopy(list(parent_i_freq_args)))
    parents_freq_args.append(copy.deepcopy(list(parent_j_freq_args)))
    parent_index = random.randint(0,1) # counter to help alternate between parents  
    
    child_i_route = [] # define child route
    child_i_freq_args = [] # define child frequency
    parent_len = len(parent_i_route)
    
    # Randomly select the first seed solution for the child
    random_index = random.randint(0,parent_len-1)
    child_i_route.append(parents_route[parent_index][random_index]) # adds seed solution to parent route
    child_i_freq_args.append(parents_freq_args[parent_index][random_index]) # adds seed solution to parent frequency
    del(parents_route[parent_index][random_index]) # removes the route from parent so that it is not evaluated again
    del(parents_freq_args[parent_index][random_index]) # removes the frequnecy from parent
    
    # Alternates parent solutions
    parent_index = looped_incrementor(parent_index, 1)
    
    
    # Calculate the unseen proportions to select next route for inclusion into child
    while len(child_i_route) < parent_len:
        # Determines all nodes present in the child
        all_nodes_present = set([y for x in child_i_route for y in x]) # flatten all the elements in child
    
        parent_curr = parents_route[parent_index] # sets the current parent
        
        proportions = []
        for i_candidate in range(len(parent_curr)):
            R_i = set(parent_curr[i_candidate])
            if bool(R_i.intersection(all_nodes_present)): # test whether there is a common transfer point
                proportions.append(len(R_i - all_nodes_present) / len(R_i)) # calculate the proportion of unseen vertices
            else:
                proportions.append(0) # set proportion to zero so that it won't be chosen
        
        # Get route that maximises the proportion of unseen nodes included
        max_indices = set([i for i, j in enumerate(proportions) if j == max(proportions)]) # position of max proportion/s
        max_index = random.sample(max_indices, 1)[0] # selects only one index randomly between a possible tie, else the only one
        
        # Add the route to the child
        child_i_route.append(parent_curr[max_index]) # add max proportion unseen nodes route to the child
        child_i_freq_args.append(parents_freq_args[parent_index][max_index]) # adds the frequency that corresponds to the route
        del(parents_route[parent_index][max_index]) # removes the route from parent so that it is not evaluated again
        del(parents_freq_args[parent_index][max_index]) # removes the frequnecy from parent

        
        # Alternates parent solutions
        parent_index = looped_incrementor(parent_index, 1)
    
    return child_i_route, child_i_freq_args


def crossover_uniform_as_is(parent_A, parent_B, parent_length):
    x_index = random.randint(1,parent_length-1)
    child_A = np.hstack((parent_A[0:x_index], parent_B[x_index:]))
    child_B = np.hstack((parent_B[0:x_index], parent_A[x_index:]))
    
    return child_A, child_B
    
    
def crossover_pop_uniform_as_is(pop, main_problem):
    selection = tournament_selection_g2(pop, n_select=int(main_problem.problem_GA_parameters.population_size/2))
    
    offspring_variable_args = np.empty([main_problem.problem_GA_parameters.population_size,
                                   main_problem.R_routes.number_of_routes]).astype(int)
    
    for i in range(0,int(main_problem.problem_GA_parameters.population_size/2)):
        parent_A = pop.variable_args[selection[i,0]]
        parent_B = pop.variable_args[selection[i,1]]
    
        offspring_variable_args[int(2*i),], offspring_variable_args[int(2*i+1),] =\
            crossover_uniform_as_is(parent_A, parent_B, main_problem.R_routes.number_of_routes)
            
    return offspring_variable_args


def crossover_pop_routes_cxp_1(pop, main_problem):
    """Crossover function for entire route population with crossover probability = 1"""
    selection = tournament_selection_g2(pop, n_select=int(main_problem.problem_GA_parameters.population_size))
    
    offspring_variables = [None] * main_problem.problem_GA_parameters.population_size

    
    for i in range(0,int(main_problem.problem_GA_parameters.population_size)):
        parent_A = pop.variables[selection[i,0]]
        parent_B = pop.variables[selection[i,1]]
    
        offspring_variables[i] = crossover_routes_unseen_prob(parent_A, parent_B)
            # crossover_uniform_as_is(parent_A, parent_B, main_problem.R_routes.number_of_routes)
        
        while not test_all_four_constraints(offspring_variables[i], main_problem):
            offspring_variables[i] = repair_add_missing_from_terminal(offspring_variables[i], main_problem)
            
            if test_all_four_constraints(offspring_variables[i], main_problem):
                continue
            else:
                offspring_variables[i] = crossover_routes_unseen_prob(parent_A, parent_B)
        
    return offspring_variables

def crossover_pop_routes(pop, main_problem):
    """Crossover function for entire route population (all or nothing)"""
    selection = tournament_selection_g2(pop, n_select=int(main_problem.problem_GA_parameters.population_size))
    
    if random.random() < main_problem.problem_GA_parameters.crossover_probability:
        offspring_variables = [None] * main_problem.problem_GA_parameters.population_size
    
        
        for i in range(0,int(main_problem.problem_GA_parameters.population_size)):
            parent_A = pop.variables[selection[i,0]]
            parent_B = pop.variables[selection[i,1]]
        
            offspring_variables[i] = crossover_routes_unseen_prob(parent_A, parent_B)
                # crossover_uniform_as_is(parent_A, parent_B, main_problem.R_routes.number_of_routes)
            
            while not test_all_four_constraints(offspring_variables[i], main_problem):
                offspring_variables[i] = repair_add_missing_from_terminal(offspring_variables[i], main_problem)
                
                if test_all_four_constraints(offspring_variables[i], main_problem):
                    continue
                else:
                    offspring_variables[i] = crossover_routes_unseen_prob(parent_A, parent_B)
        
        return offspring_variables
    
    else:
        return pop.variables
    

# Below crossover functions for testing
def crossover_mumford(parent_A, parent_B, main_problem):
    offspring_variable = crossover_routes_unseen_prob(parent_A, parent_B)        
    return offspring_variable

def crossover_unseen_probabilistic(parent_A, parent_B, main_problem):
    offspring_variable = crossover_routes_unseen_probabilistic(parent_A, parent_B)
    return offspring_variable

def crossover_mumford_rem_subsets_ksp(parent_A, parent_B, main_problem):
    debug = False
    offspring_variable = crossover_routes_unseen_prob(parent_A, parent_B) 

    _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variable)
    if max_sim == 1: 
        if debug: print("DUPLICATE: Crossover normal")
    
        # Attempt to replace any duplicate routes made by the crossover
        offspring_variable = mut_replace_path_subsets(offspring_variable, main_problem)
            
        _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variable)
        if debug: 
            if max_sim != 1: print("DUPLICATE: FIXED")
    else:
        offspring_variable = mut_replace_path_subsets(offspring_variable, main_problem)       
    return offspring_variable

def crossover_unseen_probabilistic_rem_subsets_ksp(parent_A, parent_B, main_problem):
    debug = False
    offspring_variable = crossover_routes_unseen_prob(parent_A, parent_B) 
    
    _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variable)
    if max_sim == 1: 
        if debug: print("DUPLICATE: Crossover normal")
    
        # Attempt to replace any duplicate routes made by the crossover
        offspring_variable = mut_replace_path_subsets(offspring_variable, main_problem)
            
        _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variable)
        if debug: 
            if max_sim != 1: print("DUPLICATE: FIXED")
    else:
        offspring_variable = mut_replace_path_subsets(offspring_variable, main_problem)       
    return offspring_variable

def crossover_mumford_rem_subsets(parent_A, parent_B, main_problem):
    debug = False
    offspring_variable = crossover_routes_unseen_prob(parent_A, parent_B) 

    _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variable)
    if max_sim == 1: 
        if debug: print("DUPLICATE: Crossover normal")
    
        # Attempt to replace any duplicate routes made by the crossover
        offspring_variable = mut_replace_path_subsets(offspring_variable, main_problem, parent_A+parent_B, limit_len=False)
            
        _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variable)
        if debug: 
            if max_sim != 1: print("DUPLICATE: FIXED")
    else:
        offspring_variable = mut_replace_path_subsets(offspring_variable, main_problem, parent_A+parent_B, limit_len=False)       
    return offspring_variable

def crossover_unseen_probabilistic_rem_subsets(parent_A, parent_B, main_problem):
    debug = False
    offspring_variable = crossover_routes_unseen_prob(parent_A, parent_B) 
    
    _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variable)
    if max_sim == 1: 
        if debug: print("DUPLICATE: Crossover normal")
    
        # Attempt to replace any duplicate routes made by the crossover
        offspring_variable = mut_replace_path_subsets(offspring_variable, main_problem, parent_A+parent_B, limit_len=False)
            
        _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variable)
        if debug: 
            if max_sim != 1: print("DUPLICATE: FIXED")
    else:
        offspring_variable = mut_replace_path_subsets(offspring_variable, main_problem, parent_A+parent_B, limit_len=False)       
    return offspring_variable
# Functions used in NSGA II for crossover

def crossover_pop_routes_individuals(pop, main_problem):
    """Crossover function applied to each route in population"""
    selection = tournament_selection_g2(pop, n_select=int(main_problem.problem_GA_parameters.population_size))
    
    offspring_variables = [None] * main_problem.problem_GA_parameters.population_size
     
    for i in range(0,int(main_problem.problem_GA_parameters.population_size)):
        
        if random.random() < main_problem.problem_GA_parameters.crossover_probability:
            parent_A = pop.variables[selection[i,0]]
            parent_B = pop.variables[selection[i,1]]
        
            offspring_variables[i] = crossover_routes_unseen_prob(parent_A, parent_B)
                # crossover_uniform_as_is(parent_A, parent_B, main_problem.R_routes.number_of_routes)
            
            while not test_all_four_constraints(offspring_variables[i], main_problem):
                offspring_variables[i] = repair_add_missing_from_terminal(offspring_variables[i], main_problem)
                
                if test_all_four_constraints(offspring_variables[i], main_problem):
                    continue
                else:
                    offspring_variables[i] = crossover_routes_unseen_prob(parent_A, parent_B)
    
        else:
            if random.random() < 0.5:
                offspring_variables[i] = pop.variables[selection[i,0]]
            else:
                offspring_variables[i] = pop.variables[selection[i,1]]
    
    return offspring_variables

def crossover_pop_routes_individuals_smart(pop, main_problem, crossover_func=crossover_mumford):
    """Crossover function applied to each route in population"""
    selection = tournament_selection_g2(pop, n_select=int(main_problem.problem_GA_parameters.population_size))
    
    offspring_variables = [None] * main_problem.problem_GA_parameters.population_size
     
    for i in range(0,int(main_problem.problem_GA_parameters.population_size)):
        
        if random.random() < main_problem.problem_GA_parameters.crossover_probability:
            parent_A = lol_copy(pop.variables[selection[i,0]])
            parent_B = lol_copy(pop.variables[selection[i,1]])
            counter = 0
        
            offspring_variables[i] = crossover_func(parent_A, parent_B, main_problem)
            
            while not test_all_four_constraints(offspring_variables[i], main_problem):
                offspring_variables[i] = repair_add_missing_from_terminal(offspring_variables[i], main_problem)
                counter += 1
                
                # A case for when no feasible route can be found in the odd occasion
                if counter > 100:
                    offspring_variables[i] = crossover_mumford(parent_A, parent_B, main_problem)
                    if test_all_four_constraints(offspring_variables[i], main_problem):
                        #print("Crossover by Mumford last resort success")
                        break
                    else:
                        if random.random() < 0.5:
                            offspring_variables[i] = pop.variables[selection[i,0]]
                        else:
                            offspring_variables[i] = pop.variables[selection[i,1]]
                        #print(f"No feasible route found with Crossover between \nparent_A:\n {parent_A} and \n parent_B:\n {parent_B}")
                        break
                    
                if test_all_four_constraints(offspring_variables[i], main_problem):
                    continue
                else:
                    offspring_variables[i] = crossover_func(parent_A, parent_B, main_problem)
    
        else:
            if random.random() < 0.5:
                offspring_variables[i] = pop.variables[selection[i,0]]
            else:
                offspring_variables[i] = pop.variables[selection[i,1]]
    
    return offspring_variables

def crossover_pop_routes_individuals_debug(pop, main_problem, debug=False):
    """Crossover function applied to each route in population"""
    selection = tournament_selection_g2(pop, n_select=int(main_problem.problem_GA_parameters.population_size))
    
    offspring_variables = [None] * main_problem.problem_GA_parameters.population_size
     
    for i in range(0,int(main_problem.problem_GA_parameters.population_size)):
        
        if random.random() < main_problem.problem_GA_parameters.crossover_probability:
            parent_A = pop.variables[selection[i,0]]
            parent_B = pop.variables[selection[i,1]]
        
            offspring_variables[i] = crossover_routes_unseen_probabilistic(parent_A, parent_B)

            _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variables[i])
            if max_sim == 1: 
                if debug: print("DUPLICATE: Crossover normal")
            
                # Attempt to replace any duplicate routes made by the crossover
                offspring_variables[i] = mut_replace_path_subsets(offspring_variables[i], main_problem)
                    
                _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variables[i])
                if debug: 
                    if max_sim != 1: print("DUPLICATE: FIXED")
            else:
                offspring_variables[i] = mut_replace_path_subsets(offspring_variables[i], main_problem)

            
            while not test_all_four_constraints(offspring_variables[i], main_problem):
                offspring_variables[i] = repair_add_missing_from_terminal(offspring_variables[i], main_problem)
                _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variables[i])
                if max_sim == 1: print("DUPLICATE: Crossover repaired")
                
                if test_all_four_constraints(offspring_variables[i], main_problem):
                    continue
                else:
                    offspring_variables[i] = crossover_routes_unseen_probabilistic(parent_A, parent_B)
                    _, max_sim = calc_path_similarity_matrix_for_mut(offspring_variables[i])
                    if max_sim == 1: print("DUPLICATE: Crossover retry")
        else:
            if random.random() < 0.5:
                offspring_variables[i] = pop.variables[selection[i,0]]
            else:
                offspring_variables[i] = pop.variables[selection[i,1]]
    
    return offspring_variables
    
    
def crossover_pop_routes_UTRFSP(pop, main_problem):
    """Crossover function for entire route population"""
    selection = tournament_selection_g2(pop, n_select=int(main_problem.problem_GA_parameters.population_size))
    
    offspring_variables_routes = [None] * main_problem.problem_GA_parameters.population_size
    offspring_variables_freq_args = [None] * main_problem.problem_GA_parameters.population_size
    
    for i in range(0,int(main_problem.problem_GA_parameters.population_size)):
        
        if random.random() < main_problem.problem_GA_parameters.crossover_probability_routes:
            
            parent_A_route = pop.variables_routes[selection[i,0]]
            parent_B_route = pop.variables_routes[selection[i,1]]
            parent_A_freq_args = pop.variable_freq_args[selection[i,0]]
            parent_B_freq_args = pop.variable_freq_args[selection[i,1]]
        
            offspring_variables_routes[i], offspring_variables_freq_args[i] = crossover_routes_unseen_prob_UTRFSP(parent_A_route, parent_B_route, parent_A_freq_args, parent_B_freq_args)
                # crossover_uniform_as_is(parent_A_route, parent_B_route, main_problem.R_routes.number_of_routes)
            
            while not test_all_four_constraints(offspring_variables_routes[i], main_problem):
                offspring_variables_routes[i] = repair_add_missing_from_terminal(offspring_variables_routes[i], main_problem)
                
                if test_all_four_constraints(offspring_variables_routes[i], main_problem):
                    continue
                else:
                    offspring_variables_routes[i], offspring_variables_freq_args[i] = crossover_routes_unseen_prob_UTRFSP(parent_A_route, parent_B_route, parent_A_freq_args, parent_B_freq_args)
        
        else:
            # Randomly choose one of the parents as the offspring
            if random.random() < 0.5:
                offspring_variables_routes[i] = pop.variables_routes[selection[i,0]]
                offspring_variables_freq_args[i] = pop.variable_freq_args[selection[i,0]]
            else:
                offspring_variables_routes[i] = pop.variables_routes[selection[i,1]]
                offspring_variables_freq_args[i] = pop.variable_freq_args[selection[i,1]]
                
    return offspring_variables_routes, np.array(offspring_variables_freq_args)


# %% Mutation functions

def mutate_routes_two_intertwine(routes_R, main_problem):
    """Mutate a route set by randomly choosing two routes that have a common 
    transfer point, and randomly exchanges the segments with each other"""
                                                       
    random_list = list(range(len(routes_R)))
    random.shuffle(random_list)
    
    candidate_routes_R = copy.deepcopy(routes_R)
    
    for i in random_list:
        for j in random_list:    
            if i < j:
                transfer_node = set(routes_R[i]).intersection(set(routes_R[j]))
                if bool(transfer_node): # test whether there are intersections
                    mutation_node = random.sample(transfer_node, 1)[0]
                    mutation_i_index = routes_R[i].index(mutation_node)
                    mutation_j_index = routes_R[j].index(mutation_node)
    
                    # Assigns the two segments across the mutation node
                    if random.random() < 0.5: # Randomises the mutation
                        route_front_i = routes_R[i][0:mutation_i_index]
                        route_end_i = routes_R[j][mutation_j_index:]
                        
                        route_front_j = routes_R[j][0:mutation_j_index]
                        route_end_j = routes_R[i][mutation_i_index:]

                    else:
                        reversed_route = routes_R[j][::-1] # Reverses the one route
                        mutation_j_index = reversed_route.index(mutation_node) # Recalc index node
                        
                        route_front_i = routes_R[i][0:mutation_i_index]
                        route_end_i = reversed_route[mutation_j_index:]
                        
                        route_front_j = reversed_route[0:mutation_j_index]
                        route_end_j = routes_R[i][mutation_i_index:]  
    
                    new_route_i = list(np.hstack((route_front_i, route_end_i)).astype(int))
                    new_route_j = list(np.hstack((route_front_j, route_end_j)).astype(int))
                    #print("mutation_node = {0} and i index = {1} and j index = {2}".format(str(mutation_node),str(mutation_i_index), str(mutation_j_index)))
                    #print("{0} to {1}".format(str(routes_R[i]),str(new_route_i)))
                    #print("{0} to {1}".format(str(routes_R[j]),str(new_route_j)))
                    
                    candidate_routes_R[i] = new_route_i
                    candidate_routes_R[j] = new_route_j
                    
                    if test_all_four_constraints(candidate_routes_R, main_problem):
                        return candidate_routes_R
                    else:
                        candidate_routes_R = repair_add_missing_from_terminal(candidate_routes_R, main_problem)
                        if test_all_four_constraints(candidate_routes_R, main_problem):
                            return candidate_routes_R
                        else:
                            candidate_routes_R = copy.deepcopy(routes_R) # reset the candidate routes if unsuccessful
                    
    return routes_R # if no successful mutation was found


def add_vertex_to_terminal(routes_R, main_problem):
    # Adds a vertex to a terminal vertex
    r = main_problem.problem_constraints.con_r
    mapping_adjacent = main_problem.mapping_adjacent
    
    R = copy.deepcopy(routes_R)
    p = random.uniform(0,1)
    if p < 0.5:
        change_add_node_to_first_node(R, r, mapping_adjacent)
    else:
        change_add_node_to_last_node(R, r, mapping_adjacent)
    return R  


def remove_vertex_from_terminal(routes_R, main_problem):
    # Removes a vertex from a terminal vertex
    r = main_problem.problem_constraints.con_r
    
    R = copy.deepcopy(routes_R)
    p = random.uniform(0,1)
    if p < 0.5:
        change_delete_node_from_front(R, r)
    else:
        change_delete_node_from_back(R, r)
    return R  


def get_common_terminal_vertices_pairs(routes_R):
    """A function to get all the pairs of the terminal vertices in a route set
    """
    # Get terminal nodes
    tf = [x[0] for x in routes_R] #get all terminal nodes in the first position
    tb = [x[-1] for x in routes_R] #get all terminal nodes in the last position
    
    fb=[] # list for keeping the front to back matches
    ff=[] # list for keeping the front to front matches
    bb=[] # list for keeping the back to back matches
    
    for i in range(len(tf)):
        for j in range(len(tf)):
            
            if tf[i] == tb[j]:
                fb.append((i,j))
            
            if i < j:
                if tf[i] == tf[j]:
                    ff.append((i,j))
                
                if tb[i] == tb[j]:
                    bb.append((i,j))
                    
    return {'fb':fb, 'ff':ff, 'bb':bb}
 
    
def mutate_merge_routes_at_common_terminal(route_to_mutate, UTNDP_problem_1):
    """A function that merges two routes if they share a common vertex based on
    Matthew P John's 2016 PhD Thesis"""
    ksp = UTNDP_problem_1.k_short_paths.paths
    ctv_pairs = get_common_terminal_vertices_pairs(route_to_mutate)  # common terminal vertex  
    
    potential_routes = []
    
    # OPTIMISE: NB - shuffle [ff, bb, fb] and then shuffle order in each 
    # then once a feasible solution is found, return and stop
    # as this will save time. As already randomised and dont need to get all potential solutions 
    
    for config in ['ff', 'bb', 'fb']:
        
        for pair in ctv_pairs[config]:
            routes_R = copy.deepcopy(route_to_mutate)
            
            if config == 'ff':
                P_new_front = routes_R[pair[0]]
                P_new_front.reverse()
                P_new_back = routes_R[pair[1]]
                P_new = P_new_front + P_new_back[1:]
                
            if config == 'bb':
                P_new_front = routes_R[pair[0]]
                P_new_back = routes_R[pair[1]]
                P_new_back.reverse()
                P_new = P_new_front + P_new_back[1:]
                
            if config == 'fb':
                P_new_front = routes_R[pair[1]]
                P_new_back = routes_R[pair[0]]
                P_new = P_new_front + P_new_back[1:]
                
            if len(P_new) != 0:
                routes_R = [i for j, i in enumerate(routes_R) if j not in pair]
                
                if len(set(P_new)) == len(P_new):
                    routes_R.append(P_new)
                    routes_R = repair_add_path_to_route_set_ksp(routes_R, UTNDP_problem_1, ksp)
                    if test_all_four_constraints(routes_R, UTNDP_problem_1):
                        potential_routes.append(routes_R)
                
            P_new = []
    
    if len(potential_routes):           
        return random.choice(potential_routes)
    else:
        return route_to_mutate
    
    
def mut_replace_lowest_demand(route_to_mutate, main_problem):
    '''Mutation function for replacing the route with the lowest demand
    with a route from the k shortest paths that aims to maximise demand'''
    routes_R = copy.deepcopy(route_to_mutate)
    
    mx_demand = main_problem.problem_data.mx_demand
    ksp = main_problem.k_short_paths.paths

    d = np.zeros((len(routes_R)))
    
    for i in range(len(routes_R)):
        d[i] = calc_cum_demand(routes_R[i], mx_demand)
        
    indices = np.where(d == np.min(d)) # indices_of_lowest_demand_routes
    path_to_del = random.choice(list(indices[0]))
    del routes_R[path_to_del] # remove the path
    
    routes_R = add_path_prob_unmet_demand_limited_len(routes_R, main_problem, route_to_mutate[path_to_del], routes_to_search=False)
    # routes_R = repair_add_path_to_route_set_ksp(routes_R, main_problem, ksp)
    
    return routes_R


def mut_remove_lowest_demand_terminal(route_to_mutate, main_problem):
    '''Mutation function that removes an overlapping terminal vertex that 
    either has the lowest demand route cost associated'''

    mx_demand = main_problem.problem_data.mx_demand
    mx_dist = main_problem.problem_data.mx_dist

    # calc tot initial direct demand
    d_init = calc_cum_demand_route_set(route_to_mutate, mx_demand) 
    candidates = [] # list to keep the candidates

    for i in range(len(route_to_mutate)):
        
        # front terminal vertex eval
        route_copy = copy.deepcopy(route_to_mutate)
        route_copy[i] = route_copy[i][1:]
        # test feasibility
        if test_all_four_constraints(route_copy, main_problem):
            d = calc_cum_demand_route_set(route_copy, mx_demand)
            d_cont = d_init - d # calc direct demand contribution
            c = mx_dist[route_to_mutate[i][0], route_copy[i][0]] # get edge cost
            candidates.append({'route_nr': i, 'front':True, 
                               'dem_contribution':d_cont, 'cost':c}) 
        
        # end terminal vertex eval
        route_copy = copy.deepcopy(route_to_mutate)
        route_copy[i] = route_copy[i][:-1]
        # test feasibility
        if test_all_four_constraints(route_copy, main_problem):
            d = calc_cum_demand_route_set(route_copy, mx_demand)
            d_cont = d_init - d # calc direct demand contribution
            c = mx_dist[route_to_mutate[i][-1], route_copy[i][-1]] # get edge cost
            candidates.append({'route_nr': i, 'front':False, 
                               'dem_contribution':d_cont, 'cost':c})
    
    # if no candidates, return initial route
    if len(candidates) == 0: 
        return route_to_mutate
        
    # find terminals with lowest demand or cost
    criteria = 'dem_contribution'
    demands = [x[criteria] for x in candidates]    
    candidates_min = [x for x in candidates if x[criteria] == min(demands)]
    
    candidate = random.choice(candidates_min)
    
    # extract candidate details and return mutated route
    mut_route = copy.deepcopy(route_to_mutate)
    if candidate['front']:
        mut_route[candidate['route_nr']] = mut_route[candidate['route_nr']][1:]
    else:
        mut_route[candidate['route_nr']] = mut_route[candidate['route_nr']][:-1]
    
    return mut_route

def mut_remove_largest_cost_terminal(route_to_mutate, main_problem):
    '''Mutation function that removes an overlapping terminal vertex that 
    either has the largest route cost associated'''

    mx_demand = main_problem.problem_data.mx_demand
    mx_dist = main_problem.problem_data.mx_dist

    # calc tot initial direct demand
    d_init = calc_cum_demand_route_set(route_to_mutate, mx_demand) 
    candidates = [] # list to keep the candidates

    for i in range(len(route_to_mutate)):
        
        # front terminal vertex eval
        route_copy = copy.deepcopy(route_to_mutate)
        route_copy[i] = route_copy[i][1:]
        # test feasibility
        if test_all_four_constraints(route_copy, main_problem):
            d = calc_cum_demand_route_set(route_copy, mx_demand)
            d_cont = d_init - d # calc direct demand contribution
            c = mx_dist[route_to_mutate[i][0], route_copy[i][0]] # get edge cost
            candidates.append({'route_nr': i, 'front':True, 
                               'dem_contribution':d_cont, 'cost':c}) 
        
        # end terminal vertex eval
        route_copy = copy.deepcopy(route_to_mutate)
        route_copy[i] = route_copy[i][:-1]
        # test feasibility
        if test_all_four_constraints(route_copy, main_problem):
            d = calc_cum_demand_route_set(route_copy, mx_demand)
            d_cont = d_init - d # calc direct demand contribution
            c = mx_dist[route_to_mutate[i][-1], route_copy[i][-1]] # get edge cost
            candidates.append({'route_nr': i, 'front':False, 
                               'dem_contribution':d_cont, 'cost':c})
    
    # if no candidates, return initial route
    if len(candidates) == 0: 
        return route_to_mutate
        
    # find terminals with lowest demand or cost
    criteria = 'cost'
    demands = [x[criteria] for x in candidates]    
    candidates_min = [x for x in candidates if x[criteria] == min(demands)]
    
    candidate = random.choice(candidates_min)
    
    # extract candidate details and return mutated route
    mut_route = copy.deepcopy(route_to_mutate)
    if candidate['front']:
        mut_route[candidate['route_nr']] = mut_route[candidate['route_nr']][1:]
    else:
        mut_route[candidate['route_nr']] = mut_route[candidate['route_nr']][:-1]
    
    return mut_route

def mut_remove_largest_cost_per_dem_terminal(route_to_mutate, main_problem):
    '''Mutation function that removes an overlapping terminal vertex that 
    has the largest route cost per demand associated'''

    mx_demand = main_problem.problem_data.mx_demand
    mx_dist = main_problem.problem_data.mx_dist

    # calc tot initial direct demand
    d_init = calc_cum_demand_route_set(route_to_mutate, mx_demand) 
    candidates = [] # list to keep the candidates

    for i in range(len(route_to_mutate)):
        
        # front terminal vertex eval
        route_copy = copy.deepcopy(route_to_mutate)
        route_copy[i] = route_copy[i][1:]
        # test feasibility
        if test_all_four_constraints(route_copy, main_problem):
            d = calc_cum_demand_route_set(route_copy, mx_demand)
            d_cont = d_init - d # calc direct demand contribution
            if d_cont==0: d_cont = 0.001
            c = mx_dist[route_to_mutate[i][0], route_copy[i][0]] # get edge cost
            
            candidates.append({'route_nr': i, 'front':True, 
                               'dem_contribution':d_cont, 'cost':c, 'cost_per_dem':c/d_cont}) 
        
        # end terminal vertex eval
        route_copy = copy.deepcopy(route_to_mutate)
        route_copy[i] = route_copy[i][:-1]
        # test feasibility
        if test_all_four_constraints(route_copy, main_problem):
            d = calc_cum_demand_route_set(route_copy, mx_demand)
            d_cont = d_init - d # calc direct demand contribution
            if d_cont==0: d_cont = 0.001
            c = mx_dist[route_to_mutate[i][-1], route_copy[i][-1]] # get edge cost
            candidates.append({'route_nr': i, 'front':False, 
                               'dem_contribution':d_cont, 'cost':c, 'cost_per_dem':c/d_cont})
    
    # if no candidates, return initial route
    if len(candidates) == 0: 
        return route_to_mutate
        
    # find terminals with lowest demand or cost
    criteria = 'cost_per_dem'
    demands = np.array([x[criteria] for x in candidates])
    if sum(demands)!=0: 
        dem_proportions = demands/sum(demands)
    else:
        dem_proportions = [1/len(demands) for _ in demands]
        
    candidate = random.choices(candidates, weights=dem_proportions, k=1)[0]
    
    # extract candidate details and return mutated route
    mut_route = copy.deepcopy(route_to_mutate)
    if candidate['front']:
        mut_route[candidate['route_nr']] = mut_route[candidate['route_nr']][1:]
    else:
        mut_route[candidate['route_nr']] = mut_route[candidate['route_nr']][:-1]
    
    return mut_route

def mut_replace_high_sim_routes(routes_R, main_problem):
    '''Mutation function where the routes with the maximum overlap is identified
    and then replaced by a better route'''
    debug = False
    R_copy = copy.deepcopy(routes_R)
    max_sim_list, max_sim = calc_path_similarity_matrix_for_mut(R_copy)
    pair = random.choice(max_sim_list)
    P_1 = routes_R[pair[0]]
    P_2 = routes_R[pair[1]]
        
    #NOTE: Only most similar would be replaced, and not pure subsets 
    if len(P_1) < len(P_2):
        repl_P_1 = True
    else:
        repl_P_1 = False
    
    if max_sim > 0.5:
        if debug: print(f"{P_1} \n{P_2} \n MAX:{max_sim} LIST:{max_sim_list}")

        if repl_P_1:
            del R_copy[pair[0]]
            R_copy = add_path_prob_unmet_demand_limited_len(R_copy, main_problem, routes_R[pair[0]], routes_to_search=False)

        else:
            del R_copy[pair[1]]
            R_copy = add_path_prob_unmet_demand_limited_len(R_copy, main_problem, routes_R[pair[1]], routes_to_search=False)
        
        return R_copy
    
    else:
        return routes_R
    
def are_lists_sublists(list_A, list_B):
    # using intersection() to 
    # check subset of list 
    if len(list_A) < len(list_B):
        sub_list = list_A
        test_list = list_B
    else:
        sub_list = list_B
        test_list = list_A   

    flag = False
    if((set(sub_list) & set(test_list))== set(sub_list)):
        flag = True
    return flag    


def test_list_order_and_subset(A, B):
    # Python3 program to Remove elements of 
    # list that repeated less than k times
    # A should be a shorter list than B
    n = len(A)
    return any(A == B[i:i + n] for i in range(len(B)-n + 1))
    
def mut_replace_path_subsets(routes_R, main_problem, routes_to_search=False, limit_len=True):
    '''Á function that mutates a route set by replacing subsets of lists
    subject to the same ordering in a given route set routes_R. Returns the
    mutated route as output'''
    # Identify shortest subset that should be replaced
    debug = False
    mut_R = copy.deepcopy(routes_R)
    ksp = main_problem.k_short_paths.paths
    
    for i, R_i in enumerate(mut_R):
        for j, R_j in enumerate(mut_R):
            if i < j:
                if are_lists_sublists(R_i, R_j):
                    if len(R_i) < len(R_j):
                        sub_list = R_i
                        sub_list_index = i
                        test_list = R_j
                    else:
                        sub_list = R_j
                        sub_list_index = j
                        test_list = R_i 
    
                    if test_list_order_and_subset(sub_list, test_list):
                        if debug: print(sub_list_index)
                        mut_R = replace_path_prob_demand_per_cost(mut_R, main_problem, sub_list_index, routes_to_search)
                        # mut_R = replace_path_prob_unmet_demand_limited_len(mut_R, main_problem, sub_list_index, routes_to_search, limit_len=limit_len)
                    else:
                        sub_list.reverse()
                        if test_list_order_and_subset(sub_list, test_list):
                            if debug: print(sub_list_index)
                            mut_R = replace_path_prob_demand_per_cost(mut_R, main_problem, sub_list_index, routes_to_search)
                            # mut_R = replace_path_prob_unmet_demand_limited_len(mut_R, main_problem, sub_list_index, routes_to_search, limit_len=limit_len)
                
                    while is_route_sublist_in_set(mut_R, sub_list_index):
                        mut_R = replace_path_prob_demand_per_cost(mut_R, main_problem, sub_list_index, ksp)
                
                return mut_R

def is_route_sublist_in_set(routes_R, index_i):
    '''Function to test whether the route at index i is a sublist of the 
    routeset'''
    
    R_i = routes_R[index_i]
    routes_copy = copy.deepcopy(routes_R)
    del routes_copy[index_i]
    
    for R_j in routes_copy:
        if are_lists_sublists(R_i, R_j):
            return True   
    return False

def test_adj_feasible(P_x, v_i, main_problem):
    """A function to check if the two vertices adjacent to vertex v_i has an 
    edge connecting them in path P_x"""
    i = P_x.index(v_i) # index of vertex v_i
    
    if i == 0:
        if not set(main_problem.mapping_adjacent[v_i]).intersection(set([P_x[i+1]])):
            return False
    elif i == len(P_x)-1:
        if not set(main_problem.mapping_adjacent[v_i]).intersection(set([P_x[i-1]])):
            return False
    else:
        if len(set(main_problem.mapping_adjacent[v_i]).intersection(set([P_x[i-1], P_x[i+1]]))) != 2:
            return False
    
    return True

# P_x = [5, 7, 9, 2, 1, 3, 11]
# v_i = 9
# v_j = 5
# main_problem = UTNDP_problem_1
# test_adj_feasible(P_x, v_i, main_problem)

def test_two_adj_feasible(P_x, v_i, v_j, main_problem):
    if not test_adj_feasible(P_x, v_i, main_problem):   
        return False
    if not test_adj_feasible(P_x, v_j, main_problem):   
        return False
    return True

def mut_invert_route_vertices_old(routes_R, main_problem):
    '''A mutation function for inverting a randomly selected vertex and a 
    potential inversion counter-part'''
    R_1 = routes_R   
    search_order = random.sample(range(len(R_1)), k=len(R_1))
    
    for i in search_order:
        if len(R_1[i]) > 2:
            i_start_overall = random.randint(0, len(R_1[i])-1)
            vertex_start = R_1[i][i_start_overall]
            
            # Find potential inversions
            vertex_neighbours = main_problem.mapping_adjacent[vertex_start]
            potential_inverts = list(set(vertex_neighbours) & set(R_1[i]))
            
            if len(potential_inverts) != 0:
                vertex_end = random.choice(potential_inverts)
                i_end = R_1[i].index(vertex_end)       
                i_start = i_start_overall
                # i_start must be smaller than i_end
                if i_start > i_end:
                    temp = i_start
                    i_start = i_end
                    i_end = temp
                    
                R_mut = copy.deepcopy(R_1)
                
                # Reverse the part of route list from index i_start to i_end
                if i_start == 0:
                    reversed_path = R_mut[i][i_end:i_start:-1] + R_mut[i][0:i_start+1] + R_mut[i][i_end+1:]
                elif i_end == (len(R_1[i])-1):
                    reversed_path = R_mut[i][0:i_start] + R_mut[i][i_end:i_start-1:-1]
                else:
                    reversed_path = R_mut[i][0:i_start] + R_mut[i][i_end:i_start-1:-1] + R_mut[i][i_end+1:]
                
                R_mut[i] = reversed_path
                    
                
                if False: # Debug
                    print(f'Start: {i_start} End:{i_end}')
                    x = [ a==b for a,b in zip(R_1[i],R_mut[i])]
                    print(R_1[i])
                    print(R_mut[i])
                    print(len(R_mut[i]) - sum(x))
                return R_mut
        
    return routes_R

def mut_invert_route_vertices(routes_R, main_problem):
    '''A mutation function for inverting a randomly selected vertex and a 
    potential inversion counter-part'''   
    debug = False
    search_order = random.sample(range(len(routes_R)), k=len(routes_R))
    if debug: print(f"Search order: {search_order}")
    
    for i in search_order:
        P_i = routes_R[i].copy()
        if debug: print(f"\n\nP_i: {P_i}")
        if len(P_i) > 2:
            shuffled_vertices = random.sample(range(len(P_i)), k=len(P_i))
            for i_start in shuffled_vertices:
                vertex_start = P_i[i_start]
                if debug: print(f"Vertex START: {vertex_start} \t(loc: {i_start})")
                # Find potential inversions
                vertex_neighbours = main_problem.mapping_adjacent[vertex_start]
                potential_inverts = list(set(vertex_neighbours) & set(P_i)) # ensures the end node is adjacent to at least one other vertex after replaced
                len_pot = len(potential_inverts)
                
                if len_pot != 0:
                    potential_inverts = random.sample(potential_inverts, k=len_pot)
                else:
                    continue
                
                for vertex_end in potential_inverts:
                    i_end = P_i.index(vertex_end) 
                    i_start_temp = i_start
                    
                    if debug: print(f"Vertex END: {vertex_end} \t(loc: {i_end})")
    
                    # i_start_temp must be smaller than i_end
                    if i_start_temp > i_end:
                        temp = i_start_temp
                        i_start_temp = i_end
                        i_end = temp
                    
                        
                    # Reverse the part of route list from index i_start_temp to i_end
                    if i_start_temp == 0:
                        if i_end != (len(P_i)-1):
                            inverted_path = P_i[i_end:i_start_temp:-1] + P_i[0:i_start_temp+1] + P_i[i_end+1:]
                        else:
                            continue
    
                    elif i_end == (len(P_i)-1):
                        if i_start_temp != 0:
                            inverted_path = P_i[0:i_start_temp] + P_i[i_end:i_start_temp-1:-1]
                        else:
                            continue
       
                    else:
                        inverted_path = P_i[0:i_start_temp] + P_i[i_end:i_start_temp-1:-1] + P_i[i_end+1:]
    
                    
                    if test_two_adj_feasible(inverted_path, vertex_start, vertex_end, main_problem):
                        
                        mut_R = routes_R # NB, changes the input route
                        mut_R[i] = inverted_path
                        # if ev.evaluateTotalRouteLength(mut_R,main_problem.problem_data.mx_dist)>10000:   
                        #     debug = True
                        
                        if debug:
                            print(f'Start: {i_start_temp} End:{i_end}')
                            x = [ a==b for a,b in zip(routes_R[i],inverted_path)]
                            print(routes_R[i])
                            print(inverted_path)
                            print(len(inverted_path) - sum(x))
                            #print(fn_obj_2(R_mut,main_problem))
                            
                        return mut_R
                        
    return routes_R

#routes_R = offspring_variables[10]
#main_problem = UTNDP_problem_1
#mut_R = mut_invert_route_vertices(routes_R, main_problem)
#assert routes_R!=mut_R

def mut_add_vertex_inside_route_not_working(routes_R, main_problem):
    '''A mutation function for adding a randomly selected vertex into a randomly
    selected route in a route set if feasible.'''   
    debug = False
    con_max_v = main_problem.problem_constraints.con_maxNodes
    neigh = main_problem.mapping_adjacent
    search_order = random.sample(range(len(routes_R)), k=len(routes_R))
    if debug: print(f"Search order: {search_order}")
    
    for i in search_order:
        P_i = routes_R[i].copy()
        if debug: print(f"\n\nP_i: {P_i}")
        if len(P_i) < con_max_v:
            shuffled_vertices = random.sample(range(len(P_i)), k=len(P_i))
            for s in shuffled_vertices:
                v_s = P_i[s] # select a random vertex
                if debug: print(f"Vertex START: {v_s} \t(loc: {s})")
                
                # Find all the selected vertex's neighbours
                neigh_v_s = neigh[v_s]                
                
                # Find potential inserts
                v_potentials = list(set(neigh_v_s).difference(set(P_i)))
                if len(v_potentials) != 0:
                    if debug: print(f"Potential inserts: {v_potentials}")
                    v_potentials = random.sample(v_potentials, k=len(v_potentials)) # Shuffle potentials
                else:
                    continue
                
                # Search through potential vertices to add
                for v_pot in v_potentials:
                    if debug: print(f"Eval insert: {v_pot} with Neigh: {neigh[v_pot]}")
                    neigh_v_pot = neigh[v_pot] 
                    
                    v_add_feasibles = list(set(neigh_v_pot).intersection(set(neigh_v_s), set(P_i)))                 
                    if len(v_add_feasibles) != 0:
                        #v_add_feasibles = random.sample(v_add_feasibles, k=len(v_add_feasibles)) # Shuffle potentials 
                        v_add = random.choice(v_add_feasibles)
                        if debug: print(f"Eval to add vertex: {v_add}")
                    
                        if debug: print(f"Added {v_pot} between {v_s} and {v_add} at index position {max(P_i.index(v_s), P_i.index(v_add))}")
                        P_i.insert(max(P_i.index(v_s), P_i.index(v_add)), v_pot)
                        mut_R = copy.deepcopy(routes_R)
                        mut_R[i] = P_i
                        return mut_R
                    
    return routes_R

def mut_add_vertex_inside_route(routes_R, main_problem):
    '''A mutation function for adding a randomly selected vertex into a randomly
    selected route in a route set if feasible.'''   
    con_max_v = main_problem.problem_constraints.con_maxNodes
    neigh = main_problem.mapping_adjacent
    search_order = random.sample(range(len(routes_R)), k=len(routes_R))
    
    for i in search_order:
        P_i = routes_R[i].copy()
        if len(P_i) < con_max_v:
            shuffled_vertices = random.sample(range(len(P_i))[:-1], k=len(P_i)-1) # ensures the last position not chosen
            for s in shuffled_vertices:
                
                # Find all the selected vertex's neighbours
                e_pot_rep = [P_i[s], P_i[s+1]] # potential edge to replace when vertex is added
                pot_adds = set(neigh[e_pot_rep[0]]).intersection(set(neigh[e_pot_rep[1]])).difference(set(P_i))
                if pot_adds:
                    v_pot = random.choice(list(pot_adds))
                    P_i.insert(s+1, v_pot)

                    mut_R = copy.deepcopy(routes_R)
                    mut_R[i] = P_i

                    return mut_R
                    
    return routes_R

def mut_add_vertex_inside_route_debug(routes_R, main_problem):
    '''A mutation function for adding a randomly selected vertex into a randomly
    selected route in a route set if feasible.'''   
    debug = False
    con_max_v = main_problem.problem_constraints.con_maxNodes
    neigh = main_problem.mapping_adjacent
    search_order = random.sample(range(len(routes_R)), k=len(routes_R))
    if debug: print(f"Search order: {search_order}")
    
    for i in search_order:
        P_i = routes_R[i].copy()
        if debug: print(f"\n\nP_i: {P_i}")
        if len(P_i) < con_max_v:
            shuffled_vertices = random.sample(range(len(P_i))[:-1], k=len(P_i)-1) # ensures the last position not chosen
            for s in shuffled_vertices:
                v_s = P_i[s] # select a random vertex
                if debug: print(f"Vertex START: {v_s} \t(loc: {s})")
                
                # Find all the selected vertex's neighbours
                e_pot_rep = [P_i[s], P_i[s+1]] # potential edge to replace when vertex is added
                print(f"Potential edge replacement: \n{e_pot_rep}")
                pot_adds = set(neigh[e_pot_rep[0]]).intersection(set(neigh[e_pot_rep[1]])).difference(set(P_i))
                if pot_adds:
                    print(f"Potential additions: \n{pot_adds}")
                    v_pot = random.choice(list(pot_adds))
                    P_i.insert(s+1, v_pot)

                    mut_R = copy.deepcopy(routes_R)
                    mut_R[i] = P_i


                    if debug: print(f"Adding vertex {v_pot} at position {s}\nP_i: {mut_R[i]} [NEW]")
                    return mut_R
                    
    return routes_R

#routes_R = offspring_variables[10]
#main_problem = UTNDP_problem_1
#mut_R = mut_add_vertex_inside_route(routes_R, main_problem)
#assert routes_R!=mut_R

def mut_delete_vertex_inside_route(routes_R, main_problem):
    '''A mutation function for deleting a randomly selected vertex from a randomly
    selected route in a route set if feasible.'''   
    debug = True
    con_min_v = main_problem.problem_constraints.con_minNodes
    neigh = main_problem.mapping_adjacent
    search_order = random.sample(range(len(routes_R)), k=len(routes_R))
    if debug: print(f"Search order: {search_order}")
    
    for i in search_order:
        P_i = routes_R[i].copy()
        if debug: print(f"\n\nP_i: {P_i}")
        if len(P_i) > con_min_v:
            # ensures that no terminals may be selected
            shuffled_vertex_indices = random.sample(range(len(P_i))[1:-1], k=len(P_i)-2)
            for s in shuffled_vertex_indices:
                v_s = P_i[s] # select a random vertex
                if debug: print(f"Vertex START: {v_s} \t(loc: {s})")
                
                # Find all the selected vertex's neighbours
                e_pot_add = [P_i[s-1], P_i[s+1]] # potential edge to add when v_s removed
                if set(neigh[e_pot_add[0]]).intersection(set([e_pot_add[1]])):
                    mut_R = copy.deepcopy(routes_R)
                    del mut_R[i][s]
                    if debug: print(f"Deleting vertex {v_s}\nP_i: {mut_R[i]} [NEW]")
                    return mut_R
                
    return routes_R

#routes_R = offspring_variables[11]
#main_problem = UTNDP_problem_1
#mut_R = gf.mut_delete_vertex_inside_route(routes_R, main_problem)
#assert routes_R!=mut_R

def mut_relocate_vertex_inside_route(routes_R, main_problem):
    '''A mutation function for relocating a randomly selected vertex from a 
    randomly selected route in a route set if feasible.'''   
    mut_R = copy.deepcopy(routes_R)
    
    debug = True
    neigh = main_problem.mapping_adjacent
    search_order = random.sample(range(len(routes_R)), k=len(routes_R))
    if debug: print(f"Search order: {search_order}")
    
    for i in search_order:
        P_i = routes_R[i].copy()
        if debug: print(f"\n\nP_i: {P_i}")

        # ensures that no terminal may be selected
        shuffled_vertex_indices = random.sample(range(len(P_i))[1:-1], k=len(P_i)-2)
        for s in shuffled_vertex_indices:
            v_s = P_i[s] # select a random vertex
            if debug: print(f"Vertex START: {v_s} \t(loc: {s})")
            
            # Find all the selected vertex's neighbours
            e_pot_add = [P_i[s-1], P_i[s+1]] # potential edge to add when v_s removed
            # Determine whether it is safe to remove the vertex by checking the neighbours' neighbours
            if set(neigh[e_pot_add[0]]).intersection(set([e_pot_add[1]])):
                
                del mut_R[i][s] # tentatively removes the vertex
                if debug: print(f"Vertex REMOVED: {v_s} \t(loc: {s})")
                  
                shuffled_vertex_indices_t = random.sample(range(len(mut_R[i]))[:-1], k=len(mut_R[i])-1)
                shuffled_vertex_indices_t.remove(s-1) # ensures the vertex is not placed back into same position with t+1
                
                for t in shuffled_vertex_indices_t:
                    # Find all the selected vertex's neighbours
                    e_pot_rep = [P_i[t], P_i[t+1]] # potential edge to replace when vertex is added
                    if debug: print(f"Pot replace: {e_pot_rep}")
                    # pot_adds = set(neigh[e_pot_rep[0]]).intersection(set(neigh[e_pot_rep[1]])).difference(set(P_i))
                    # pot_adds = v_s
                    
                    
                    #if debug: print(f"Pot adds: {pot_adds}")
                    # ensure both vertices in path are neighbours to v_s
                    if len(set(neigh[v_s]).intersection(set(e_pot_rep)))==2:
                        #v_pot = random.choice(list(pot_adds))
                        v_pot = v_s
                        mut_R[i].insert(t+1, v_pot)
                        if debug: print(f"Vertex INSERTED: {v_pot} \t(loc: {t+1})\nP_i: {mut_R[i]} [NEW]")
                
                        return mut_R
                
                mut_R[i] = P_i.copy() # reset the route
                
             
    return routes_R

#routes_R = offspring_variables[11]
#main_problem = UTNDP_problem_1
#mut_R = mut_relocate_vertex_inside_route(routes_R, main_problem)
#assert routes_R!=mut_R

def mut_replace_vertex_inside_route(routes_R, main_problem):
    '''A mutation function for raplacing a randomly selected vertex with 
    another random feasible vertex from a randomly selected route in a route 
    set if feasible.'''   
    debug = True
    neigh = main_problem.mapping_adjacent
    search_order = random.sample(range(len(routes_R)), k=len(routes_R))
    if debug: print(f"Search order: {search_order}")
    
    for i in search_order:
        P_i = routes_R[i].copy()
        if debug: print(f"\n\nP_i: {P_i}")

        # ensures that no terminals may be selected
        shuffled_vertex_indices = random.sample(range(len(P_i))[1:-1], k=len(P_i)-2)
        for s in shuffled_vertex_indices:
            v_s = P_i[s] # select a random vertex
            if debug: print(f"Vertex START: {v_s} \t(loc: {s})")
            
            # Find all the selected vertex's neighbours
            e_pot_add = [P_i[s-1], P_i[s+1]] # two neighbours of v_s
            
            # see whether the two neighbours have another vertex in common that is not in the route set
            pot_replace = list(set(neigh[e_pot_add[0]]).intersection(set(neigh[e_pot_add[1]])).difference(set(P_i))) 
            
            if debug: print(f"Edge: {e_pot_add} with potential replacements: {pot_replace}")
            
            if pot_replace:
                v_replace = random.choice(pot_replace)
                mut_R = copy.deepcopy(routes_R)
                mut_R[i][s] = v_replace
                if debug: print(f"Replacing vertex {v_s} with {v_replace}\nP_i: {mut_R[i]} [NEW]")
                return mut_R
                
    return routes_R

#routes_R = offspring_variables[11]
#main_problem = UTNDP_problem_1
#mut_R = mut_replace_vertex_inside_route(routes_R, main_problem)
#assert routes_R!=mut_R

def mut_swap_vertices_between_routes(routes_R, main_problem):
    '''A mutation function for swaspping a randomly selected vertex with 
    another random feasible vertex from a randomly selected route in a route 
    set if feasible.'''   
    debug = True
    neigh = main_problem.mapping_adjacent
    search_order = random.sample(range(len(routes_R)), k=len(routes_R))
    if debug: print(f"Search order: {search_order}")
    
    for i in search_order:
        P_i = routes_R[i].copy()
        if debug: print(f"\n\nP_i: {P_i}")

        # ensures that no terminals may be selected
        shuffled_vertex_indices = random.sample(range(len(P_i))[1:-1], k=len(P_i)-2)
        for s in shuffled_vertex_indices:
            v_s = P_i[s] # select a random vertex
            if debug: print(f"Vertex START v_s: {v_s} \t(loc: {s})")
            
            # Find all the selected vertex's neighbours
            e_pot_i = [P_i[s-1], P_i[s+1]] # two neighbours of v_s
            if debug: print(f"Considering neighbours of v_s: {e_pot_i}")
            
            # see whether the two neighbours have another vertex in common that is not in the route set
            pot_replace_i = list(set(neigh[e_pot_i[0]]).intersection(set(neigh[e_pot_i[1]])).difference(set(P_i))) 
            if debug: print(f"Pot replace i: {pot_replace_i}")
            
            search_order_j = random.sample(range(len(routes_R)), k=len(routes_R))
            for j in search_order_j:
                P_j = routes_R[j].copy()
                if debug: print(f"\n\nP_j: {P_j}")
                
                # ensures that no terminals may be selected
                shuffled_vertex_indices_t = random.sample(range(len(P_j))[1:-1], k=len(P_j)-2)
                    
                for t in shuffled_vertex_indices_t:
                    if debug: print(f"Vertex START v_s: {v_s} \t(loc: {s})")
                    if debug: print(f"Considering neighbours of v_s: {e_pot_i}")
                    if debug: print(f"Pot replace i: {pot_replace_i}")
                    
                    v_t = P_j[t] # select a random vertex
                    if debug: print(f"Vertex START v_t: {v_t} \t(loc: {t})")
                    # Find all the selected vertex's neighbours
                    e_pot_j = [P_j[t-1], P_j[t+1]] # two neighbours of v_t
                    if debug: print(f"Considering neighbours of v_t: {e_pot_j}")
            
                    # see whether the two neighbours have another vertex in common that is not in the route set
                    pot_replace_j = list(set(neigh[e_pot_j[0]]).intersection(set(neigh[e_pot_j[1]])).difference(set(P_j))) 
                    if debug: print(f"Pot replace j: {pot_replace_j}\n")

                    bool_v_s_into_P_j = (len(set([v_s]).intersection(set(pot_replace_j))) == 1) # test if v_s can go into P_j
                    bool_v_t_into_P_i = (len(set([v_t]).intersection(set(pot_replace_i))) == 1) # test if v_t can go into P_i


                    if bool_v_s_into_P_j & bool_v_t_into_P_i: # if both true, then swap
                        if debug: print(f"P_i: {routes_R[i]} [OLD]\nP_j: {routes_R[j]} [OLD]")
                        mut_R = copy.deepcopy(routes_R)
                        mut_R[i][s] = v_t
                        mut_R[j][t] = v_s
                        if debug: print(f"Replacing vertex {v_s} with {v_t} (index {s} with {t})\nP_i: {mut_R[i]} [NEW]\nP_j: {mut_R[j]} [NEW]")
                        return mut_R
                
    return routes_R

#routes_R = offspring_variables[11]
#main_problem = UTNDP_problem_1
#mut_R = mut_swap_vertices_between_routes(routes_R, main_problem)
#assert routes_R!=mut_R

def mut_donate_vertex_between_routes(routes_R, main_problem):
    '''A mutation function for donating a randomly selected vertex from a randomly
    selected route to another randomly selected route in a route set if feasible.'''   
    debug = True
    con_min_v = main_problem.problem_constraints.con_minNodes
    con_max_v = main_problem.problem_constraints.con_maxNodes
    neigh = main_problem.mapping_adjacent
    search_order = random.sample(range(len(routes_R)), k=len(routes_R))
    if debug: print(f"Search order: {search_order}")
    
    for i in search_order:
        P_i = routes_R[i].copy()
        if debug: print(f"\n\nP_i: {P_i}")
        if len(P_i) > con_min_v:
            # ensures that no terminals may be selected
            shuffled_vertex_indices = random.sample(range(len(P_i))[1:-1], k=len(P_i)-2)
            for s in shuffled_vertex_indices:
                v_s = P_i[s] # select a random vertex
                if debug: print(f"Vertex START: {v_s} \t(loc: {s})")
                
                # Find all the selected vertex's neighbours
                e_pot_add = [P_i[s-1], P_i[s+1]] # potential edge to add when v_s removed
                # Determine whether it is safe to remove the vertex by checking the neighbours' neighbours
                if set(neigh[e_pot_add[0]]).intersection(set([e_pot_add[1]])):
                    
                    search_order_j = random.sample(range(len(routes_R)), k=len(routes_R))
                    for j in search_order_j:
                        if i != j:
                            P_j = routes_R[j].copy()
                            if debug: print(f"\n\nP_j: {P_j}")
                            # ensure the vertex is not in P_j and that it has at least two neighbours in P_j
                            
                            bool_v_s_in_P_j = (len(set([v_s]).difference(set(P_j)))!=0)
                            bool_neigh_v_s_in_P_j = (len(set(neigh[v_s]).intersection(set(P_j)))>1)
                            bool_P_j_len_con = (len(P_j) < con_max_v)
                            
                            
                            if  bool_v_s_in_P_j & bool_neigh_v_s_in_P_j & bool_P_j_len_con:
                                
                                shuffled_vertex_indices_t = random.sample(range(len(P_j))[:-1], k=len(P_j)-1)
                                
                                for t in shuffled_vertex_indices_t:
                                    # Find all the selected vertex's neighbours
                                    e_pot_rep = [P_j[t], P_j[t+1]] # potential edge to replace when vertex is added
                                    if debug: print(f"Pot replace: {e_pot_rep}")                                   
                                    
                                    # ensure both vertices in path are neighbours to v_s
                                    if len(set(neigh[v_s]).intersection(set(e_pot_rep)))==2:
                                        mut_R = copy.deepcopy(routes_R)
                                        del mut_R[i][s] # remove vertex v_s from P_i
                                        if debug: print(f"Vertex REMOVED: {v_s} \t(loc: {s})\nP_i: {routes_R[i]} [OLD]\nP_i: {mut_R[i]} [NEW]")
                                        
                                        v_pot = v_s
                                        P_j.insert(t+1, v_pot) # insert vertex v_s into P_j
                                        mut_R[j] = P_j
                                        if debug: print(f"Vertex INSERTED: {v_pot} \t(loc: {t+1})\nP_j: {routes_R[j]} [OLD]\nP_j: {mut_R[j]} [NEW]")
                                
                                        return mut_R

    return routes_R

#routes_R = offspring_variables[11]
#main_problem = UTNDP_problem_1
#mut_R = mut_donate_vertex_inside_route(routes_R, main_problem)
#assert routes_R!=mut_R

def mut_remove_largest_cost_per_dem_terminal_from_path(route_to_mutate, main_problem, path_index):
    '''Mutation function that removes a terminal vertex from a path that 
    has the highest cost per route demand'''

    mx_demand = main_problem.problem_data.mx_demand
    mx_dist = main_problem.problem_data.mx_dist

    # calc tot initial direct demand
    d_init = calc_cum_demand_route_set(route_to_mutate, mx_demand) 
    candidates = [] # list to keep the candidates

    i = path_index
    
    # front terminal vertex eval
    route_copy = copy.deepcopy(route_to_mutate)
    route_copy[i] = route_copy[i][1:]
    
    # test feasibility
    if test_all_four_constraints(route_copy, main_problem):
        d = calc_cum_demand_route_set(route_copy, mx_demand)
        d_cont = d_init - d # calc direct demand contribution
        if d_cont==0: d_cont = 0.001
        c = mx_dist[route_to_mutate[i][0], route_copy[i][0]] # get edge cost
        candidates.append({'route_nr': i, 'front':True, 
                           'dem_contribution':d_cont, 'cost':c, 'cost_per_dem':c/d_cont}) 
    
    # end terminal vertex eval
    route_copy = copy.deepcopy(route_to_mutate)
    route_copy[i] = route_copy[i][:-1]
    
    # test feasibility
    if test_all_four_constraints(route_copy, main_problem):
        d = calc_cum_demand_route_set(route_copy, mx_demand)
        d_cont = d_init - d # calc direct demand contribution
        if d_cont==0: d_cont = 0.001
        c = mx_dist[route_to_mutate[i][-1], route_copy[i][-1]] # get edge cost
        candidates.append({'route_nr': i, 'front':False, 
                           'dem_contribution':d_cont, 'cost':c, 'cost_per_dem':c/d_cont})
    
    # if no candidates, return initial route
    if len(candidates) == 0: 
        return route_to_mutate
        
    # find terminals with lowest demand or cost
    criteria = 'cost_per_dem'
    demands = np.array([x[criteria] for x in candidates])
    if sum(demands)!=0: 
        dem_proportions = demands/sum(demands)
    else:
        dem_proportions = [1/len(demands) for _ in demands]
        
    # candidates_min = [x for x in candidates if x[criteria] == min(demands)]
    # candidate = random.choice(candidates_min)

    candidate = random.choices(candidates, weights=dem_proportions, k=1)[0] 
    
    # extract candidate details and return mutated route
    mut_route = copy.deepcopy(route_to_mutate)
    if candidate['front']:
        mut_route[candidate['route_nr']] = mut_route[candidate['route_nr']][1:]
    else:
        mut_route[candidate['route_nr']] = mut_route[candidate['route_nr']][:-1]
    
    return mut_route

def mut_trim_one_path_random_cb(routes_R, main_problem):
    '''A mutation function that removes k_i vertices from path i in a route set
    with with k_i being an integer number between 1 and |V(path_i)| - con_min_vertices
    for one random path i chosen from the route set.
    Include a probabilistic element where the demand met per route cost is 
    considered.'''
    
    route_copy = copy.deepcopy(routes_R)
    len_routes = len(routes_R)
    con_min_v = main_problem.problem_constraints.con_minNodes
        
    # Get the random route index for mutation 
    list_I = [random.choice(range(len_routes))]
            
    for i in list_I:
        r_i = route_copy[i]        
        if len(r_i) > con_min_v:
            k = random.randint(1, len(r_i) - con_min_v) # number of vertices to remove
        else:
            k = 0
        while k > 0: 
            route_copy = mut_remove_largest_cost_per_dem_terminal_from_path(route_copy, main_problem, i)
            k -= 1
    
    return route_copy

def mut_trim_routes_random_cb(routes_R, main_problem):
    '''A mutation function that removes k_i vertices from path i in a route set
    with with k_i being an integer number between 1 and |V(path_i)| - con_min_vertices
    for each random path i chosen from the route set, where I random paths are
    chosen with I being between 1 and |route set|.
    Include a probabilistic element where the demand met per route cost is 
    considered.'''
    
    route_copy = copy.deepcopy(routes_R)
    len_routes = len(routes_R)
    con_min_v = main_problem.problem_constraints.con_minNodes
        
    # Get random route indices for mutations    
    I = random.randint(1,len_routes) # the number of routes to mutate
    list_I = random.sample(range(len_routes), I)
            
    for i in list_I:
        r_i = route_copy[i]        
        if len(r_i) > con_min_v:
            k = random.randint(1, len(r_i) - con_min_v) # number of vertices to remove
        else:
            k = 0
        while k > 0: 
            route_copy = mut_remove_largest_cost_per_dem_terminal_from_path(route_copy, main_problem, i)
            k -= 1
    
    return route_copy

def mut_trim_all_paths_random_cb(routes_R, main_problem):
    '''A mutation function that removes k_i vertices from path i in a route set
    with with k_i being an integer number between 1 and |V(path_i)| - con_min_vertices
    for each random path i chosen from the route set, where I random paths are
    chosen with I being between 1 and |route set|.
    Include a probabilistic element where the demand met per route cost is 
    considered.'''
    
    route_copy = copy.deepcopy(routes_R)
    len_routes = len(routes_R)
    con_min_v = main_problem.problem_constraints.con_minNodes
        
    # Get random route indices for mutations    
    I = len_routes # the number of routes to mutate
    list_I = random.sample(range(len_routes), I)
            
    for i in list_I:
        r_i = route_copy[i]        
        if len(r_i) > con_min_v:
            k = random.randint(1, len(r_i) - con_min_v) # number of vertices to remove
        else:
            k = 0
        while k > 0: 
            route_copy = mut_remove_largest_cost_per_dem_terminal_from_path(route_copy, main_problem, i)
            k -= 1
    
    return route_copy

def mut_trim_full_overall_cb(routes_R, main_problem):
    '''A mutation function that removes k_i vertices from path i in a route set
    with with k_i being an integer number between 1 and |V(path_i)| - con_min_vertices
    for each random path i chosen from the route set, where I random paths are
    chosen with I being between 1 and |route set|.
    Include a probabilistic element where the demand met per route cost is 
    considered.'''
    
    route_copy = copy.deepcopy(routes_R)
    route_to_mut = copy.deepcopy(routes_R)
                  
    route_copy = mut_remove_largest_cost_per_dem_terminal(route_to_mut, main_problem)
    
    while route_to_mut != route_copy:
        route_to_mut = copy.deepcopy(route_copy)
        route_copy = mut_remove_largest_cost_per_dem_terminal(route_to_mut, main_problem)

    return route_copy

#R_x = ld_mut_temp[0]['Route']
#R_mut = mut_full_trim_overall(routes_R=R_x, main_problem=UTNDP_problem_1)
#mut_full_trim(routes_R=R_x, main_problem=UTNDP_problem_1)
#print(R_mut)

def mut_add_terminal_highest_demand_per_cost(routes_R, main_problem):
    '''A mutation function that adds 1 vertex to path i in a route set
    for 1 random path i chosen from the route set.
    Include a probabilistic element where the highest demand contribution per 
    route cost is considered.'''
    
    route_copy = copy.deepcopy(routes_R)
    len_routes = len(routes_R)
    mx_demand = main_problem.problem_data.mx_demand
    mx_dist = main_problem.problem_data.mx_dist
    con_max_v = main_problem.problem_constraints.con_maxNodes
    mapping_adjacent = main_problem.mapping_adjacent 
    candidates = []   

    d_init = calc_cum_demand_route_set(route_copy, mx_demand)

    for i in range(len_routes):
        r_i = route_copy[i].copy() # keep a copy of the route that is evaluated
        
        if len(r_i) < con_max_v: # ensures feasible solution
            # test for best additional vertex to include based on demand per cost

    
            # Evaluate the front of the route set first
            adjs_front = list(set(mapping_adjacent[r_i[0]]).difference(set(r_i)))                
            for adj in adjs_front:
                route_copy[i] = r_i.copy() # replace the route in index i
                
                # front terminal vertex eval
                route_copy[i].insert(0,adj)
                
                # test feasibility
                if test_all_four_constraints(route_copy, main_problem):
                    d = calc_cum_demand_route_set(route_copy, mx_demand)
                    d_cont = d - d_init # calc direct demand contribution
                    c = mx_dist[r_i[0], route_copy[i][0]] # get edge cost
                    candidates.append({'route_nr': i, 'front':True, 'adj':adj,
                                       'dem_contribution':d_cont, 'cost':c, 'dem_per_cost':d_cont/c}) 
                
                
            # Evaluate the end of the route set second
            adjs_end = list(set(mapping_adjacent[r_i[-1]]).difference(set(r_i)))    
            for adj in adjs_end:   
                route_copy[i] = r_i.copy() # replace the route in index i

                # end terminal vertex eval
                route_copy[i].append(adj)
             
                # test feasibility
                if test_all_four_constraints(route_copy, main_problem):
                    d = calc_cum_demand_route_set(route_copy, mx_demand)
                    d_cont = d - d_init # calc direct demand contribution
                    c = mx_dist[r_i[-1], route_copy[i][-1]] # get edge cost
                    candidates.append({'route_nr': i, 'front':False, 'adj':adj,
                                       'dem_contribution':d_cont, 'cost':c, 'dem_per_cost':d_cont/c})
        
        # Replace the route in index i to reset route
        route_copy[i] = r_i.copy() 

    # if no candidates, return initial route
    if len(candidates) == 0: 
        return routes_R
        
    # find terminals with lowest demand per cost
    criteria = 'dem_per_cost'
    demands = np.array([x[criteria] for x in candidates])
    if np.sum(demands)!=0: 
        dem_proportions = demands/np.sum(demands)
    else:
        dem_proportions = [1/len(demands) for _ in demands]

    candidate = random.choices(candidates, weights=dem_proportions, k=1)[0] 

    # extract candidate details and return mutated route
    if candidate['front']:
        route_copy[candidate['route_nr']].insert(0,candidate['adj'])
    else:
        route_copy[candidate['route_nr']].append(candidate['adj'])

    return route_copy


def mut_grow_one_path_random_cb(routes_R, main_problem):
    '''A mutation function that adds k_i vertices to path i in a route set
    with with k_i being an integer number between 1 and con_max_vertices - |V(path_i)|
    for each random path i chosen from the route set, where I random paths are
    chosen with I being 1.
    Include a probabilistic element where the highest demand contribution per route cost is 
    considered.'''
    
    route_copy = copy.deepcopy(routes_R)
    len_routes = len(routes_R)
    mx_demand = main_problem.problem_data.mx_demand
    mx_dist = main_problem.problem_data.mx_dist
    con_max_v = main_problem.problem_constraints.con_maxNodes
    mapping_adjacent = main_problem.mapping_adjacent 
       
    # Get random route indices for mutations    
    I = 1 # the number of routes to mutate
    list_I = random.sample(range(len_routes), I)
        
    for i in list_I:
        r_i = route_copy[i]
        if len(r_i) < con_max_v:
            k = random.randint(1, con_max_v - len(r_i)) # number of vertices to add
        else:
            k = 0

        while k > 0:
            candidates = []   
            r_i = route_copy[i]
            # test for best additional vertex to include based on demand per cost
            d_init = calc_cum_demand_route_set(route_copy, mx_demand)

            # Evaluate the front of the route set first
            adjs_front = list(set(mapping_adjacent[r_i[0]]).difference(set(r_i)))                
            for adj in adjs_front:
                
                # front terminal vertex eval
                route_temp = copy.deepcopy(route_copy)
                route_temp[i].insert(0,adj)
                
                # test feasibility
                if test_all_four_constraints(route_temp, main_problem):
                    d = calc_cum_demand_route_set(route_temp, mx_demand)
                    d_cont = d - d_init # calc direct demand contribution
                    c = mx_dist[route_copy[i][0], route_temp[i][0]] # get edge cost
                    candidates.append({'route_nr': i, 'front':True, 'adj':adj,
                                       'dem_contribution':d_cont, 'cost':c, 'dem_per_cost':d_cont/c}) 
                
                
            # Evaluate the end of the route set second
            adjs_end = list(set(mapping_adjacent[r_i[-1]]).difference(set(r_i)))    
            for adj in adjs_end:   
                
            # end terminal vertex eval
                route_temp = copy.deepcopy(route_copy)
                route_temp[i].append(adj)
             
                # test feasibility
                if test_all_four_constraints(route_temp, main_problem):
                    d = calc_cum_demand_route_set(route_temp, mx_demand)
                    d_cont = d - d_init # calc direct demand contribution
                    c = mx_dist[route_copy[i][-1], route_temp[i][-1]] # get edge cost
                    candidates.append({'route_nr': i, 'front':False, 'adj':adj,
                                       'dem_contribution':d_cont, 'cost':c, 'dem_per_cost':d_cont/c})
                
            # if no candidates, return initial route
            if len(candidates) == 0: 
                break
                
            # find terminals with lowest demand per cost
            criteria = 'dem_per_cost'
            demands = np.array([x[criteria] for x in candidates])
            if sum(demands)!=0: 
                dem_proportions = demands/sum(demands)
            else:
                dem_proportions = [1/len(demands) for _ in demands]
        
            candidate = random.choices(candidates, weights=dem_proportions, k=1)[0] 
    
            # extract candidate details and return mutated route
            mut_route = copy.deepcopy(route_copy)
            if candidate['front']:
                mut_route[candidate['route_nr']].insert(0,candidate['adj'])
            else:
                mut_route[candidate['route_nr']].append(candidate['adj'])
    
            
            route_copy = copy.deepcopy(mut_route)
            k -= 1
            
    if test_all_four_constraints(route_copy, main_problem):
        return route_copy 
    else:
        #print("mut_grow_one_path_random_cb funct did not produce feasible solution")
        return routes_R

def mut_grow_routes_random_cb(routes_R, main_problem):
    '''A mutation function that adds k_i vertices to path i in a route set
    with with k_i being an integer number between 1 and con_max_vertices - |V(path_i)|
    for each random path i chosen from the route set, where I random paths are
    chosen with I being between 1 and |route set|.
    Include a probabilistic element where the highest demand contribution per route cost is 
    considered.'''
    
    route_copy = copy.deepcopy(routes_R)
    len_routes = len(routes_R)
    mx_demand = main_problem.problem_data.mx_demand
    mx_dist = main_problem.problem_data.mx_dist
    con_max_v = main_problem.problem_constraints.con_maxNodes
    mapping_adjacent = main_problem.mapping_adjacent 
       
    # Get random route indices for mutations    
    I = random.randint(1,len_routes) # the number of routes to mutate
    list_I = random.sample(range(len_routes), I)
        
    for i in list_I:
        r_i = route_copy[i]
        if len(r_i) < con_max_v:
            k = random.randint(1, con_max_v - len(r_i)) # number of vertices to add
        else:
            k = 0

        while k > 0:
            candidates = []   
            r_i = route_copy[i]
            # test for best additional vertex to include based on demand per cost
            d_init = calc_cum_demand_route_set(route_copy, mx_demand)

            # Evaluate the front of the route set first
            adjs_front = list(set(mapping_adjacent[r_i[0]]).difference(set(r_i)))                
            for adj in adjs_front:
                
                # front terminal vertex eval
                route_temp = copy.deepcopy(route_copy)
                route_temp[i].insert(0,adj)
                
                # test feasibility
                if test_all_four_constraints(route_temp, main_problem):
                    d = calc_cum_demand_route_set(route_temp, mx_demand)
                    d_cont = d - d_init # calc direct demand contribution
                    c = mx_dist[route_copy[i][0], route_temp[i][0]] # get edge cost
                    candidates.append({'route_nr': i, 'front':True, 'adj':adj,
                                       'dem_contribution':d_cont, 'cost':c, 'dem_per_cost':d_cont/c}) 
                
                
            # Evaluate the end of the route set second
            adjs_end = list(set(mapping_adjacent[r_i[-1]]).difference(set(r_i)))    
            for adj in adjs_end:   
                
            # end terminal vertex eval
                route_temp = copy.deepcopy(route_copy)
                route_temp[i].append(adj)
             
                # test feasibility
                if test_all_four_constraints(route_temp, main_problem):
                    d = calc_cum_demand_route_set(route_temp, mx_demand)
                    d_cont = d - d_init # calc direct demand contribution
                    c = mx_dist[route_copy[i][-1], route_temp[i][-1]] # get edge cost
                    candidates.append({'route_nr': i, 'front':False, 'adj':adj,
                                       'dem_contribution':d_cont, 'cost':c, 'dem_per_cost':d_cont/c})
                
            # if no candidates, return initial route
            if len(candidates) == 0: 
                break
                
            # find terminals with lowest demand per cost
            criteria = 'dem_per_cost'
            demands = np.array([x[criteria] for x in candidates])
            if sum(demands)!=0: 
                dem_proportions = demands/sum(demands)
            else:
                dem_proportions = [1/len(demands) for _ in demands]
        
            candidate = random.choices(candidates, weights=dem_proportions, k=1)[0] 
    
            # extract candidate details and return mutated route
            mut_route = copy.deepcopy(route_copy)
            if candidate['front']:
                mut_route[candidate['route_nr']].insert(0,candidate['adj'])
            else:
                mut_route[candidate['route_nr']].append(candidate['adj'])
    
            
            route_copy = copy.deepcopy(mut_route)
            k -= 1
            
    if test_all_four_constraints(route_copy, main_problem):
        return route_copy 
    else:
        #print("mut_grow_routes_random_cb funct did not produce feasible solution")
        return routes_R
    
    
def mut_grow_all_paths_random_cb(routes_R, main_problem):
    '''A mutation function that adds k_i vertices to path i in a route set
    with with k_i being an integer number con_max_vertices - |V(path_i)|
    for each all paths i chosen from the route set, where I random paths are
    chosen with I being |route set|.
    Include a probabilistic element where the highest demand contribution per route cost is 
    considered.'''
    
    route_copy = copy.deepcopy(routes_R)
    len_routes = len(routes_R)
    mx_demand = main_problem.problem_data.mx_demand
    mx_dist = main_problem.problem_data.mx_dist
    con_max_v = main_problem.problem_constraints.con_maxNodes
    mapping_adjacent = main_problem.mapping_adjacent 
       
    # Get random route indices for mutations    
    I = len_routes # the number of routes to mutate
    list_I = random.sample(range(len_routes), I)
        
    for i in list_I:
        r_i = route_copy[i]
        if len(r_i) < con_max_v:
            k = random.randint(1, con_max_v - len(r_i)) # number of vertices to add
        else:
            k = 0

        while k > 0:
            candidates = []   
            r_i = route_copy[i]
            # test for best additional vertex to include based on demand per cost
            d_init = calc_cum_demand_route_set(route_copy, mx_demand)

            # Evaluate the front of the route set first
            adjs_front = list(set(mapping_adjacent[r_i[0]]).difference(set(r_i)))                
            for adj in adjs_front:
                
                # front terminal vertex eval
                route_temp = copy.deepcopy(route_copy)
                route_temp[i].insert(0,adj)
                
                # test feasibility
                if test_all_four_constraints(route_temp, main_problem):
                    d = calc_cum_demand_route_set(route_temp, mx_demand)
                    d_cont = d - d_init # calc direct demand contribution
                    c = mx_dist[route_copy[i][0], route_temp[i][0]] # get edge cost
                    candidates.append({'route_nr': i, 'front':True, 'adj':adj,
                                       'dem_contribution':d_cont, 'cost':c, 'dem_per_cost':d_cont/c}) 
                
                
            # Evaluate the end of the route set second
            adjs_end = list(set(mapping_adjacent[r_i[-1]]).difference(set(r_i)))    
            for adj in adjs_end:   
                
            # end terminal vertex eval
                route_temp = copy.deepcopy(route_copy)
                route_temp[i].append(adj)
             
                # test feasibility
                if test_all_four_constraints(route_temp, main_problem):
                    d = calc_cum_demand_route_set(route_temp, mx_demand)
                    d_cont = d - d_init # calc direct demand contribution
                    c = mx_dist[route_copy[i][-1], route_temp[i][-1]] # get edge cost
                    candidates.append({'route_nr': i, 'front':False, 'adj':adj,
                                       'dem_contribution':d_cont, 'cost':c, 'dem_per_cost':d_cont/c})
                
            # if no candidates, return initial route
            if len(candidates) == 0: 
                break
                
            # find terminals with lowest demand per cost
            criteria = 'dem_per_cost'
            demands = np.array([x[criteria] for x in candidates])
            if sum(demands)!=0: 
                dem_proportions = demands/sum(demands)
            else:
                dem_proportions = [1/len(demands) for _ in demands]
        
            candidate = random.choices(candidates, weights=dem_proportions, k=1)[0] 
    
            # extract candidate details and return mutated route
            mut_route = copy.deepcopy(route_copy)
            if candidate['front']:
                mut_route[candidate['route_nr']].insert(0,candidate['adj'])
            else:
                mut_route[candidate['route_nr']].append(candidate['adj'])
    
            
            route_copy = copy.deepcopy(mut_route)
            k -= 1
            
    if test_all_four_constraints(route_copy, main_problem):
        return route_copy 
    else:
        #print("mut_grow funct did not produce feasible solution")
        return routes_R
 
    
def mut_grow_full_overall_cb(routes_R, main_problem):
    '''A mutation function that adds vertices to a path in the route set until
    all paths are the maximum number of nodes.
    Include a probabilistic element where the highest demand contribution per route cost is 
    considered.'''
    
    route_copy = copy.deepcopy(routes_R)
    route_to_mut = copy.deepcopy(routes_R)
                  
    route_copy = mut_add_terminal_highest_demand_per_cost(route_to_mut, main_problem)
    
    while route_to_mut != route_copy:
        route_to_mut = copy.deepcopy(route_copy)
        route_copy = mut_add_terminal_highest_demand_per_cost(route_to_mut, main_problem)

    return route_copy
    
#R_x = ld_mut_temp[0]['Route']
#R_mut = mut_grow(routes_R=R_x, main_problem=UTNDP_problem_1)
#R_mut_2 = mut_grow_full(routes_R=R_x, main_problem=UTNDP_problem_1)
#R_mut_3 = mut_add_terminal_highest_demand_per_cost(routes_R=R_x, main_problem=UTNDP_problem_1)
# print(R_mut_3)
#R_mut_4 = mut_grow_full_overall(routes_R=R_x, main_problem=UTNDP_problem_1)
#mut_grow(routes_R=R_x, main_problem=UTNDP_problem_1)
#print(R_mut)

       
def no_mutation(routes_R, main_problem):   
    return routes_R 


def mutate_overall_routes(routes_R, main_problem, mutation_probability):
    """This is a function that helps with the overall random choosing of any of 
    the predefined mutations, and can be appended easily"""
    
    p_rand = random.random()
    
    if p_rand < mutation_probability:
    
        if random.random() < main_problem.problem_GA_parameters.mutation_ratio: # 1st Mutation: Two routes intertwine
            candidate_routes_R = mutate_routes_two_intertwine(routes_R, main_problem)
            
            if test_all_four_constraints(candidate_routes_R, main_problem):
                return candidate_routes_R
            else:
                candidate_routes_R = repair_add_missing_from_terminal(candidate_routes_R, main_problem)
                
                if test_all_four_constraints(candidate_routes_R, main_problem):
                    return candidate_routes_R
                else:
                    return routes_R
        
        else: # 2nd Mutation: Add or remove node at terminal
            candidate_routes_R = perturb_make_small_change(routes_R, 
                                                    main_problem.problem_constraints.con_r, 
                                                    main_problem.mapping_adjacent)
            
            if test_all_four_constraints(candidate_routes_R, main_problem):
                return candidate_routes_R
            else:
                candidate_routes_R = repair_add_missing_from_terminal(candidate_routes_R, main_problem)
                
                if test_all_four_constraints(candidate_routes_R, main_problem):
                    return candidate_routes_R
                else:
                    return routes_R
                        
    else:
        return routes_R
    
def mutate_overall_routes_all(routes_R, main_problem):
    """This is a function that helps with the overall random choosing of any of 
    the predefined mutations, and can be appended easily"""
    mut_prob = main_problem.problem_GA_parameters.mutation_probability # mutation probability
    mut_ratio = main_problem.problem_GA_parameters.mutation_ratio
    mut_ratio = [0.2, 0.4, 0.2, 0.2] # mutation ratio list  
    
    p_rand = random.random()
    
    if random.random() < mut_prob:
    
        if p_rand < sum(mut_ratio[:1]): # 1st Mutation: Two routes intertwine
            candidate_routes_R = mutate_routes_two_intertwine(routes_R, main_problem) 
            
            if test_all_four_constraints(candidate_routes_R, main_problem):
                return candidate_routes_R
            else:
                candidate_routes_R = repair_add_missing_from_terminal(candidate_routes_R, main_problem)
                
                if test_all_four_constraints(candidate_routes_R, main_problem):
                    return candidate_routes_R
                else:
                    return routes_R
        
        elif p_rand < sum(mut_ratio[:2]): # 2nd Mutation: Add node at terminal
            candidate_routes_R = add_vertex_to_terminal(routes_R, main_problem)
            
            if test_all_four_constraints(candidate_routes_R, main_problem):
                return candidate_routes_R
            else:
                candidate_routes_R = repair_add_missing_from_terminal(candidate_routes_R, main_problem)
                
                if test_all_four_constraints(candidate_routes_R, main_problem):
                    return candidate_routes_R
                else:
                    return routes_R
                
        elif p_rand < sum(mut_ratio[:3]): # 3rd Mutation: Remove node at terminal
            candidate_routes_R = remove_vertex_from_terminal(routes_R, main_problem)
            
            if test_all_four_constraints(candidate_routes_R, main_problem):
                return candidate_routes_R
            else:
                candidate_routes_R = repair_add_missing_from_terminal(candidate_routes_R, main_problem)
                
                if test_all_four_constraints(candidate_routes_R, main_problem):
                    return candidate_routes_R
                else:
                    return routes_R
            
        elif p_rand < sum(mut_ratio[:4]): # 4th Mutation: Merge paths at terminals
            candidate_routes_R = mutate_merge_routes_at_common_terminal(routes_R, main_problem)
            return candidate_routes_R
    else:
        return routes_R
    
def mutate_overall_routes_all_smart(routes_R, main_problem):
    """This is a function that helps with the overall random choosing of any of 
    the predefined mutations, and can be appended easily"""
    try:
        mut_prob = main_problem.problem_GA_parameters.mutation_probability # mutation probability
    except:
        mut_prob = 1
    mut_ratio = main_problem.mutation_ratio
    #mut_ratio = [0.2, 0.4, 0.2, 0.2] # mutation ratio list
    mut_functions = main_problem.mutation_functions
    
    mut_nr = 0 # sets the mutation number to return, 0 is default with no mutation
    mut_successful = 0
    mut_repaired = 0
    copied_route_R = lol_copy(routes_R)
    output_list = {"Route":copied_route_R, "Mut_nr":mut_nr,
                   "Mut_successful":mut_successful, "Mut_repaired":mut_repaired}
    
    p_rand = random.random() # generate random number
    
    if random.random() < mut_prob:
        for mut_i in range(len(mut_functions)):
            
            # test if cumulative probability true
            if p_rand < sum(mut_ratio[:mut_i+1]):
                # mutate route set
                output_list["Mut_nr"] = mut_i+1
                candidate_routes_R = mut_functions[mut_i](copied_route_R, main_problem) 
                
                
                if mut_functions[mut_i].__name__ == "no_mutation":
                    output_list["Mut_successful"] = 1
                    output_list["Route"] = candidate_routes_R
                    return output_list
                
                if candidate_routes_R == routes_R:
                    return output_list
                
                # test feasibility
                if test_all_four_constraints(candidate_routes_R, main_problem):
                    output_list["Mut_successful"] = 1
                    output_list["Route"] = candidate_routes_R
                    return output_list
                else:
                    # attempt repair
                    candidate_routes_R = repair_add_missing_from_terminal(candidate_routes_R, main_problem)
                    
                    # test feasibility
                    if test_all_four_constraints(candidate_routes_R, main_problem):
                        output_list["Mut_repaired"] = 1
                        output_list["Route"] = candidate_routes_R
                        return output_list
                    else:
                        return output_list
            

        return output_list # if for loop completes and no mutation was performed
    
    else:
        return output_list 
    
def mutate_overall_routes_all_smart_SA(routes_R, main_problem):
    """This is a function that helps with the overall random choosing of any of 
    the predefined mutations, and can be appended easily
    For SA algorithm, with BIG change that no repair is attempted"""
    try:
        mut_prob = main_problem.problem_GA_parameters.mutation_probability # mutation probability
    except:
        mut_prob = 1
    mut_ratio = main_problem.mutation_ratio
    #mut_ratio = [0.2, 0.4, 0.2, 0.2] # mutation ratio list
    mut_functions = main_problem.mutation_functions
    
    mut_nr = 0 # sets the mutation number to return, 0 is default with no mutation
    mut_successful = 0
    mut_repaired = 0
    output_list = {"Route":routes_R, "Mut_nr":mut_nr,
                   "Mut_successful":mut_successful, "Mut_repaired":mut_repaired}
    
    p_rand = random.random() # generate random number
    
    if random.random() < mut_prob:
        for mut_i in range(len(mut_functions)):
            
            # test if cumulative probability true
            if p_rand < sum(mut_ratio[:mut_i+1]):
                # mutate route set
                output_list["Mut_nr"] = mut_i+1
                candidate_routes_R = mut_functions[mut_i](routes_R, main_problem) 
                
                
                if mut_functions[mut_i].__name__ == "no_mutation":
                    output_list["Mut_successful"] = 1
                    output_list["Route"] = candidate_routes_R
                    return output_list
                
                if candidate_routes_R == routes_R:
                    return output_list
                
                # test feasibility
                if test_all_four_constraints(candidate_routes_R, main_problem):
                    output_list["Mut_successful"] = 1
                    output_list["Route"] = candidate_routes_R
                    return output_list
                else:
                    # attempt repair
                    #candidate_routes_R = repair_add_missing_from_terminal(candidate_routes_R, main_problem)
                    
                    # test feasibility
                    # if test_all_four_constraints(candidate_routes_R, main_problem):
                    #     output_list["Mut_repaired"] = 1
                    #     output_list["Route"] = candidate_routes_R
                    #     return output_list
                    # else:
                    return output_list
            

        return output_list # if for loop completes and no mutation was performed
    
    else:
        return output_list 
       
def mutate_overall_routes_all_smart_debug(routes_R, main_problem):
    """This is a function that helps with the overall random choosing of any of 
    the predefined mutations, and can be appended easily"""
    mut_prob = main_problem.problem_GA_parameters.mutation_probability # mutation probability
    mut_ratio = main_problem.mutation_ratio
    #mut_ratio = [0.2, 0.4, 0.2, 0.2] # mutation ratio list
    mut_functions = main_problem.mutation_functions
    
    mut_nr = 0 # sets the mutation number to return, 0 is default with no mutation
    mut_successful = 0
    mut_repaired = 0
    output_list = {"Route":routes_R, "Mut_nr":mut_nr,
                   "Mut_successful":mut_successful, "Mut_repaired":mut_repaired}
    
    p_rand = random.random() # generate random number
    
    if random.random() < mut_prob:
        for mut_i in range(len(mut_functions)):
            
            # test if cumulative probability true
            if p_rand < sum(mut_ratio[:mut_i+1]):
                # mutate route set
                output_list["Mut_nr"] = mut_i+1
                candidate_routes_R = mut_functions[mut_i](routes_R, main_problem) 
                
                
                if mut_functions[mut_i].__name__ == "no_mutation":
                    output_list["Mut_successful"] = 1
                    output_list["Route"] = candidate_routes_R
                    return output_list
                
                if candidate_routes_R == routes_R:
                    return output_list
                
                # test feasibility
                if test_all_four_constraints(candidate_routes_R, main_problem):
                    output_list["Mut_successful"] = 1
                    output_list["Route"] = candidate_routes_R
                    _, max_sim = calc_path_similarity_matrix_for_mut(candidate_routes_R)
                    if max_sim == 1: print(f"DUPLICATE: Mutation {main_problem.mutation_names[mut_i]}")
                    return output_list
                else:
                    # attempt repair
                    candidate_routes_R = repair_add_missing_from_terminal(candidate_routes_R, main_problem)
                
                    # test feasibility
                    if test_all_four_constraints(candidate_routes_R, main_problem):
                        output_list["Mut_repaired"] = 1
                        output_list["Route"] = candidate_routes_R
                        _, max_sim = calc_path_similarity_matrix_for_mut(candidate_routes_R)
                        if max_sim == 1: print(f"DUPLICATE: Mutation Repair {main_problem.mutation_names[mut_i]}")
                        return output_list
                    else:
                        return output_list
            

        return output_list # if for loop completes and no mutation was performed
    
    else:
        return output_list 
    
    
def mutate_route_population(pop_variables_routes, main_problem):
    """A function to mutate over the entire population"""
    pop_mutated_variables = copy.deepcopy(pop_variables_routes)
    for i in range(len(pop_mutated_variables)):
        mut_output = mutate_overall_routes_all_smart(pop_mutated_variables[i], main_problem)
        pop_mutated_variables[i] = mut_output["Route"]
    return pop_mutated_variables

def mutate_route_population_detailed(pop_variables_routes, main_problem):
    """A function to mutate over the entire population"""
    pop_mutated_variables = copy.deepcopy(pop_variables_routes)
    df_mut_details = pd.DataFrame(columns=(["Mut_nr", "Mut_successful", "Mut_repaired"]))
    for i in range(len(pop_mutated_variables)):
        mut_output = mutate_overall_routes_all_smart(pop_mutated_variables[i], main_problem)
        df_mut_details.loc[i] = [np.float32(mut_output["Mut_nr"]), mut_output["Mut_successful"], mut_output["Mut_repaired"]]
        pop_mutated_variables[i] = mut_output["Route"]
        
    
    return pop_mutated_variables, df_mut_details

def mutate_route_population_detailed_ld(pop_variables_routes, main_problem):
    """A function to mutate over the entire population, 
    returns a list of dictionaries"""
    pop_mutated_variables = copy.deepcopy(pop_variables_routes)
    ld_mut_details = []
    for i in range(len(pop_mutated_variables)):
        mut_output = mutate_overall_routes_all_smart(pop_mutated_variables[i], main_problem)
        ld_mut_details.append(mut_output)
                    
    return ld_mut_details

def mutate_route_population_UTRFSP(pop_variables_routes, main_problem):
    """A function to mutate over the entire population"""
    pop_mutated_variables = copy.deepcopy(pop_variables_routes)
    for i in range(len(pop_mutated_variables)):
         pop_mutated_variables[i] = mutate_overall_routes(pop_mutated_variables[i], main_problem, 
                          main_problem.problem_GA_parameters.mutation_probability_routes)
    return pop_mutated_variables


# %% Objective functions
""" Define the Objective UTNDP functions """
def fn_obj_2(routes, UTNDP_problem_input):
    return (ev.evalObjs(routes, 
            UTNDP_problem_input.problem_data.mx_dist, 
            UTNDP_problem_input.problem_data.mx_demand, 
            UTNDP_problem_input.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)

def fn_obj_3(routes, UTNDP_problem_input):
    '''An onbjective function for calculating Average Travel Time and Route set
    similarity to an initial solution'''
    R_compare = UTNDP_problem_input.route_compare
    
    ATT = ev.evalATT(routes, 
            UTNDP_problem_input.problem_data.mx_dist, 
            UTNDP_problem_input.problem_data.mx_demand, 
            UTNDP_problem_input.problem_inputs.__dict__)
    RD = calc_route_set_disruption(R_compare, routes)
    return (ATT, RD) # returns (f1_ATT, f3_RD)

# %% Similarity functions
def return_all_route_set_edges(R_x):
    R_x_edges = [(P_x[i], P_x[i+1]) for P_x in R_x for i in range(len(P_x)-1)]
    return R_x_edges

def return_all_path_edges(P_x):
    P_x_edges = [(P_x[i], P_x[i+1]) for i in range(len(P_x)-1)]
    return P_x_edges

def calc_similarity_from_ms(R_1_ms, R_2_ms):
    sim = 2*(len(R_1_ms.intersection(R_2_ms)))/(len(R_1_ms) + len(R_2_ms))
    return sim

def calc_route_set_similarity(R_1, R_2):
    '''Takes as input two route sets, each being a list of lists,
    for example:
        R_1 = [[1,2,3,4], [5,2,6]]
        R_2 = [[1,2,3,4], [5,2,3,6]]
    and returns the percentage similarity in terms of edges'''
    
    # Get all the edges
    R_1_edges = return_all_route_set_edges(R_1)
    R_2_edges = return_all_route_set_edges(R_2)

    # Working with the multisets
    R_1_ms = multiset.Multiset(R_1_edges)
    R_2_ms = multiset.Multiset(R_2_edges)
    
    similarity = calc_similarity_from_ms(R_1_ms, R_2_ms)
    return similarity

def calc_route_set_disruption(R_1, R_2):
    '''Takes as input two route sets, each being a list of lists,
    for example:
        R_1 = [[1,2,3,4], [5,2,6]]
        R_2 = [[1,2,3,4], [5,2,3,6]]
    and returns the percentage disruption in terms of edges'''
    
    disruption = 1 - calc_route_set_similarity(R_1, R_2)
    return disruption

def calc_path_similarity(R_1, R_2):
    '''Takes as input two paths, each being a list of vertices,
    for example:
        R_1 = [5,2,6]
        R_2 = [5,2,3,6]
    and returns the percentage similarity in terms of edges'''
    
    # Get all the edges
    R_1_edges = return_all_path_edges(R_1)
    R_2_edges = return_all_path_edges(R_2)

    # Working with the multisets
    R_1_ms = multiset.Multiset(R_1_edges)
    R_2_ms = multiset.Multiset(R_2_edges)
    
    similarity = calc_similarity_from_ms(R_1_ms, R_2_ms)
    return similarity

def calc_route_set_similarity_matrix(R_list):
    '''Takes as input a list of numerous routes and calculates the similarity
    between all of the routes'''
    n = len(R_list) # length of routes list
    
    R_ms = [multiset.Multiset(return_all_route_set_edges(r)) for r in R_list]
    
    mx_sim = np.ones((n,n))
    for i in range(n):
        for j in range(n):
            if i < j: 
                sim = calc_similarity_from_ms(R_ms[i], R_ms[j])
                mx_sim[i,j] = sim
                mx_sim[j,i] = sim
    return mx_sim

def calc_path_similarity_matrix(P_list):
    '''Takes as input a list of numerous paths and calculates the similarity
    between all of the paths'''
    n = len(P_list) # length of routes list
    
    R_ms = [multiset.Multiset(return_all_path_edges(p)) for p in P_list]
    
    mx_sim = np.ones((n,n))
    for i in range(n):
        for j in range(n):
            if i < j: 
                sim = calc_similarity_from_ms(R_ms[i], R_ms[j])
                mx_sim[i,j] = sim
                mx_sim[j,i] = sim
    return mx_sim

def calc_path_similarity_matrix_for_mut(P_list):
    '''Takes as input a list of numerous paths and calculates the similarity
    between all of the paths, returning the max similarity and the respective route indices'''
    n = len(P_list) # length of routes list
    
    R_ms = [multiset.Multiset(return_all_path_edges(p)) for p in P_list]
    
    max_sim = 0
    max_sim_list = []
    
    for i in range(n):
        for j in range(n):
            if i < j: 
                sim = calc_similarity_from_ms(R_ms[i], R_ms[j])
                
                if sim == max_sim:
                    max_sim_list.append((i,j))
                    
                if sim > max_sim:
                    max_sim = sim
                    max_sim_list = [(i,j)]
                    
    return max_sim_list, max_sim

def calc_avg_route_set_diversity(R_list):
    '''Takes as input a list of numerous routes and calculates the diversity
    between all of the routes'''
    n = len(R_list) # length of routes list
    
    R_ms = [multiset.Multiset(return_all_route_set_edges(r)) for r in R_list]
    
    mx_sim = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i < j: 
                sim = calc_similarity_from_ms(R_ms[i], R_ms[j])
                mx_sim[i,j] = sim

    avg_div = 1 - np.sum(mx_sim)/((n*n-n)/2)
    return avg_div

# %% Non-dominated set creation functions
def create_non_dom_set_from_dataframe(df_data_for_analysis, obj_1_name='F_3', obj_2_name='F_4'):
    df_non_dominated_set = copy.deepcopy(df_data_for_analysis.loc[df_data_for_analysis['Rank'] == 0]) # create df for non-dominated set
    df_non_dominated_set = df_non_dominated_set[is_pareto_efficient(df_non_dominated_set[[obj_1_name,obj_2_name]].values, True)]
    df_non_dominated_set = df_non_dominated_set.sort_values(by=obj_1_name, ascending=True) # sort
    return df_non_dominated_set

def determine_extreme_sols_and_objs_from_pareto_set(df_overall_pareto_set):
    routes_R_op_str = df_overall_pareto_set['R_x'].iloc[-1]
    routes_R_op = convert_routes_str2list(routes_R_op_str)
    op_objs = (df_overall_pareto_set['f_1'].iloc[-1], df_overall_pareto_set['f_2'].iloc[-1])
    
    routes_R_pas_str = df_overall_pareto_set['R_x'].iloc[0]
    routes_R_pas = convert_routes_str2list(routes_R_pas_str)
    pas_objs = (df_overall_pareto_set['f_1'].iloc[0], df_overall_pareto_set['f_2'].iloc[0])
    
    return {'R_op':routes_R_op, 'R_op_str':routes_R_op_str, 'Objs_op':op_objs, 'R_pas':routes_R_pas, 'R_pas_str':routes_R_pas_str, 'Objs_pas':pas_objs}


def create_strings_extreme_solutions(routes_R_op, routes_R_pas, HV, HV_BM, name_input_data, main_problem, op_objs=False, pas_objs=False):
    
    # Get the main parameters
    mx_dist = main_problem.problem_data.mx_dist
    mx_demand = main_problem.problem_data.mx_demand
    parameters_input = main_problem.problem_inputs.__dict__
    
    # OPERATOR EVALUATION
    if op_objs: # saves time if the objectives are provided
        objs_op = op_objs
    else:
        objs_op = ev.evalObjs(routes_R_op,mx_dist,mx_demand,parameters_input)
    evaluation_op = ev.fullPassengerEvaluation(routes_R_op, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
        
    f1_op, f2_op = objs_op
    d0_op, d1_op, d2_op, d_un_op  = evaluation_op[1:5] 
    
    # Generate string for route set
    routes_op= "["
    for route in routes_R_op:  
        routes_op = routes_op + f"{route},\n"
    routes_op = routes_op[:-2] + "]"
    
    # PASSENGER EVALUATION
    if pas_objs: # saves time if the objectives are provided
        objs_pas = pas_objs
    else:
        objs_pas = ev.evalObjs(routes_R_pas,mx_dist,mx_demand,parameters_input)
    evaluation_pas = ev.fullPassengerEvaluation(routes_R_pas, mx_dist, mx_demand, parameters_input['total_demand'],parameters_input['n'],parameters_input['tp'],parameters_input['wt'])
    
    f1_pas, f2_pas = objs_pas
    d0_pas, d1_pas, d2_pas, d_un_pas  = evaluation_pas[1:5] 
    
    # Generate string for route set
    routes_pas= "["
    for route in routes_R_pas:  
        routes_pas = routes_pas + f"{route},\n"
    routes_pas = routes_pas[:-2] + "]"  
    
    summary_str = f"{name_input_data} RESULTS EVALUATION:\n\
HV:{HV:.4f} | BM:{HV_BM:.4f}\n\n\
\tOperator\tPassenger\n\
f1_ATT\t{f1_op:.4f}\t\t{f1_pas:.4f}\n\
f2_TRT\t{f2_op:.0f}\t\t{f2_pas:.0f}\n\
d_0\t{d0_op:.4f}\t\t{d0_pas:.4f}\n\
d_1\t{d1_op:.4f}\t\t{d1_pas:.4f}\n\
d_2\t{d2_op:.4f}\t\t{d2_pas:.4f}\n\
d_un\t{d_un_op:.4f}\t\t{d_un_pas:.4f}\n"
                    
    final_str = f"{summary_str}\n\
Best Operator Route set:\n\
{routes_op}\n\n\
Best Passenger Route set:\n\
{routes_pas}\n\n\
Best Operator Route set string:\n\
{convert_routes_list2str(routes_R_op)}\n\n\
Best Passenger Route set sting:\n\
{convert_routes_list2str(routes_R_pas)}"
        
    return final_str
    
    
def print_extreme_solutions(df_overall_pareto_set, HV, HV_BM, name_input_data, main_problem, path_to_save):
    
    # Create dictionary for extreme solutions
    EX = determine_extreme_sols_and_objs_from_pareto_set(df_overall_pareto_set)
    
    final_str = create_strings_extreme_solutions(EX['R_op'], EX['R_pas'], HV, HV_BM, name_input_data, main_problem, op_objs=EX['Objs_op'], pas_objs=EX['Objs_pas'])
    try:
        with open(path_to_save / "Extreme solutions.txt", "w") as text_file:
            text_file.write(final_str)
    except:
        print("Could not print the text file of extreme solutions!")
        
def get_equispaced_indices(n_solutions, objs_sorted):
    seed_indices = np.percentile(range(0,len(objs_sorted)),np.linspace(0, 100, num=n_solutions))
    #seeding_route_choices = seeding_route_set.iloc[seed_indices,:]
    return seed_indices


# %% Update mutation ratios

# AMALGAM
def update_mutation_ratio_amalgam(df_mut_summary, UTNDP_problem_1):
    nr_of_mutations = len(UTNDP_problem_1.mutation_functions)
    mutation_threshold = UTNDP_problem_1.problem_GA_parameters.mutation_threshold
    success_ratio = df_mut_summary["Inc_over_Tot"].iloc[-nr_of_mutations:].values
    
    # reset the success ratios if all have falied
    if sum(success_ratio) != 0:
        success_proportion = (success_ratio / sum(success_ratio))*(1-nr_of_mutations*mutation_threshold)      
        updated_ratio = mutation_threshold + success_proportion
        UTNDP_problem_1.mutation_ratio = updated_ratio
    else:
        # success_ratio = np.array([1/len(success_ratio) for _ in success_ratio])
        # print("Mutation ratios reset")  
        pass
    
def update_mutation_ratio_amalgam_for_SA(df_mut_summary, UTNDP_problem_1):
    nr_of_mutations = len(UTNDP_problem_1.mutation_functions)
    mutation_threshold = UTNDP_problem_1.problem_SA_parameters.mutation_threshold
    success_ratio = df_mut_summary["Inc_over_Tot"].iloc[-nr_of_mutations:].values
    
    # reset the success ratios if all have falied
    if sum(success_ratio) != 0:
        success_proportion = (success_ratio / sum(success_ratio))*(1-nr_of_mutations*mutation_threshold)      
        updated_ratio = mutation_threshold + success_proportion
        UTNDP_problem_1.mutation_ratio = updated_ratio
    else:
        # success_ratio = np.array([1/len(success_ratio) for _ in success_ratio])
        # print("Mutation ratios reset")  
        pass
