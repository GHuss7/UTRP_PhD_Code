# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:44:36 2019

@author: 17832020
"""
# %% Import libraries
import re
import string
import numpy as np
import pandas as pd
import igraph as ig
import math
import random
from itertools import compress
import networkx as nx
import copy
import pygmo as pg
from math import inf

import DSS_UTNDP_Functions as gf
import DSS_Visualisation as gv

#%% Formatting functions

def format_routes_with_letters(R_routes):
    # format routes with letters
    R_routes_named = copy.deepcopy(R_routes)
    for i in range(len(R_routes)):
        if i <= 25:
            for j in range(len(R_routes[i])):
                R_routes_named[i][j] = ''.join([char*(i//26 + 1) for char in string.ascii_uppercase[i%26]])+str(R_routes[i][j])         
    return R_routes_named

def get_names_of_routes_as_list(R_routes_named):
    names_of_transit_routes = []
    for sublist in R_routes_named: 
        for val in sublist: 
            names_of_transit_routes.append(val) 
    return names_of_transit_routes

def f_i_u_i_test(fi, ui, alpha_val):
    if fi == 0 and ui == inf:
        return alpha_val
    else:
        return fi*ui
    
#%% Objective function calculations
        
def calc_route_lengths(mx_dist, R_routes):
    # returns a vector of the lengths of the individual routes
    len_r = len(R_routes)
    routeLengths = np.zeros(len_r)
    
    for i in range(len_r):
        path = R_routes[i]
        dist = 0
    
        for j in range(len(path)-1):
            dist = dist + mx_dist[path[j] , path[j+1]]
    
        routeLengths[i] = dist

    return routeLengths

def generate_random_vehicle_assignments(len_r, num_fleet_size):
    # returns a vector with vehicles assigned to routes randomly
    Veh_per_route = np.ones(len_r) # assign one vehicle to each route
    Remaining_veh = num_fleet_size - len_r

    for i in np.random.randint(len_r, size = Remaining_veh):
        Veh_per_route[i] = Veh_per_route[i] + 1
    
    return Veh_per_route

# %% f3_ETT objective function [OLD]

def f3_ETT_old(R_routes, F_x, mx_dist, mx_demand, parameters_input):
    '''Create the transit network'''
    R_routes_named = format_routes_with_letters(R_routes)
    names_of_transit_routes = get_names_of_routes_as_list(R_routes_named)
    names_all_transit_nodes = list(map(str, range(parameters_input['n'])))+names_of_transit_routes
    n_transit_nodes = len(names_all_transit_nodes)
    
    vec_nodes_u_i_ALL_array = np.empty((n_transit_nodes,parameters_input['n'])) # creates array to contain all the final u_i values
    
    dict_all_nodes = dict() # creates a dictionary to map all the transit nodes to numbers
    for i in range(len(names_all_transit_nodes)):
        dict_all_nodes[names_all_transit_nodes[i]] = i
         
    mx_transit_network = np.zeros(shape=(n_transit_nodes, n_transit_nodes))
    mx_C_a = parameters_input['large_dist']*np.ones(shape=(n_transit_nodes, n_transit_nodes))
    mx_f_a = np.zeros(shape=(n_transit_nodes, n_transit_nodes))
    
    
    # Did not  name the numpy arrays
    
    '''Fill in the walk links that are present in the graph'''
    for i in range(parameters_input['n']):  
        for j in range(parameters_input['n']):
            if mx_dist[i,j] == parameters_input['large_dist']:
                mx_C_a[i,j] = mx_dist[i,j]
            else:
                if mx_dist[i,j] == 0:
                    mx_transit_network[i,j] = 0
                    mx_C_a[i,j] = parameters_input['walkFactor']*mx_dist[i,j]
                    mx_f_a[i,j] = 0
                else:
                    mx_transit_network[i,j] = 1
                    mx_C_a[i,j] = parameters_input['walkFactor']*mx_dist[i,j]
                    mx_f_a[i,j] = inf
                # else:
                #     """Test for weird results"""
                #     mx_transit_network[i,j] = 0
                #     mx_C_a[i,j] = parameters_input['large_dist']
                #     mx_f_a[i,j] = 0
    
    ''''Fill in the boarding links characteristics'''
    counter = 0
    
    for i in range(len(R_routes)): 
        for j in range(len(R_routes[i])):       
            i_index =  int(re.findall(r'\d+', names_of_transit_routes[counter])[0]) # number of the transit node
            j_index = names_all_transit_nodes.index(names_of_transit_routes[counter]) # position of the transit node in network graph
    
            mx_transit_network[i_index, j_index] = 1   
            mx_C_a[i_index, j_index] = parameters_input['boardingTime'] # sets the boarding
            mx_f_a[i_index, j_index] = F_x[i] # set the frequencies per transit line
      
            counter = counter + 1  
    
    '''Fill in the alighting links characteristics'''
    counter = 0
    
    for i in range(len(R_routes)): 
        for j in range(len(R_routes[i])):       
            i_index =  int(re.findall(r'\d+', names_of_transit_routes[counter])[0]) # number of the transit node
            j_index = names_all_transit_nodes.index(names_of_transit_routes[counter]) # position of the transit node in network graph
    
            mx_transit_network[j_index, i_index] = 1   
            mx_C_a[j_index, i_index] = parameters_input['alightingTime'] # sets the alighting
            mx_f_a[j_index, i_index] = inf # set the frequencies per transit line
      
            counter = counter + 1 
     
    '''Fill in the travel times using the transit lines / routes'''       
    
    for i in range(len(names_of_transit_routes) - 1):
        if " ".join(re.findall("[a-zA-Z]+", names_of_transit_routes[i]))==\
        " ".join(re.findall("[a-zA-Z]+", names_of_transit_routes[i+1])):
             
            i_index = names_all_transit_nodes.index(names_of_transit_routes[i])
            j_index = names_all_transit_nodes.index(names_of_transit_routes[i+1])
        
            mx_transit_network[i_index, j_index] =\
            mx_transit_network[j_index, i_index] = 1 
           
            mx_C_a[i_index, j_index] =\
            mx_C_a[j_index, i_index] =\
            mx_dist[int(re.findall(r'\d+', names_of_transit_routes[i])[0]),\
            int(re.findall(r'\d+', names_of_transit_routes[i+1])[0])]
        
            mx_f_a[i_index, j_index] =\
            mx_f_a[j_index, i_index] = inf
      
    '''Put all the links in one matrix'''  
    df_transit_links = pd.DataFrame(columns = ["I_i", "I_j", "c_a","f_a"])
    
    counter = 0
    for i in range(n_transit_nodes):
        for j in range(n_transit_nodes):
            if mx_transit_network[i,j]:
          
                df_transit_links.loc[counter] = [names_all_transit_nodes[i],\
                                     names_all_transit_nodes[j], mx_C_a[i,j], mx_f_a[i,j]]                 
                counter = counter + 1
    
    del counter, i, i_index, j, j_index    
    

    '''Optimal strategy algorithm (Spiess, 1989)'''    
    mx_volumes_nodes = np.zeros(n_transit_nodes) # create object to keep the node volumes
    mx_volumes_links = np.zeros(shape=(n_transit_nodes, n_transit_nodes)) # create object to keep the arc volumes
    names_main_nodes = names_all_transit_nodes[0:parameters_input['n']]
    
    u_i_times = np.zeros(shape=(parameters_input['n'],parameters_input['n']))
    
        # Overall loop to change the destinations
    for i_destination in range(parameters_input['n']): 
        
        # Create the data frames to keep the answers in
        df_opt_strat_alg = pd.DataFrame(columns = ["a=(i,","j)","f_a","u_j+c_a","a_in_A_bar"])
        vec_nodes_u_i = np.ones(n_transit_nodes)*inf
        vec_nodes_f_i = np.zeros(n_transit_nodes)
        
        # Set values of the first row
        r_destination = i_destination
        num_transit_links = len(df_transit_links)
        vec_nodes_u_i[r_destination] = 0 # set the destination expected time
        df_S_list = df_transit_links.copy() # creates a copy to work with
        
        for i in range(num_transit_links):
            df_S_list.iloc[i,0] = dict_all_nodes[df_S_list.iloc[i,0]]
            df_S_list.iloc[i,1] = dict_all_nodes[df_S_list.iloc[i,1]]
        
        mx_S_list = df_S_list.values # cast as numpy array for speed in calculations
        mx_S_list = np.hstack((mx_S_list, np.arange(num_transit_links).reshape(num_transit_links,1))) #adds indices
        
        df_A_bar_strategy_lines = pd.DataFrame(columns = ["I_i", "I_j", "c_a","f_a"])
        mx_A_bar_strategy_lines = np.empty(shape=(0,4))
        
        # repeats steps 6.2 and 6.3 until df_S_list is empty
        
        '''Get the next link'''
        
        for counter_S_list in range(num_transit_links-1, -1, -1):
            
            for i in range(counter_S_list+1): # loop through mx_S_list to find the minimum u_j + c_a
              
                if i == 0:
                    u_j = vec_nodes_u_i[int(mx_S_list[i,1])]
                    c_a = mx_S_list[i,2]
                    min_u_j_and_c_a = u_j + c_a
                    min_u_j_and_c_a_index = i
                
                else:
                    u_j = vec_nodes_u_i[int(mx_S_list[i,1])]
                    c_a = mx_S_list[i,2]
                    
                    if u_j + c_a <= min_u_j_and_c_a:
                      
                        min_u_j_and_c_a = u_j + c_a
                        min_u_j_and_c_a_index = i
                      
            '''Update the node label'''
            current_link = mx_S_list[min_u_j_and_c_a_index,:4] 
        
                
            col_index_i = int(current_link[0])
            u_i = vec_nodes_u_i[col_index_i]
            f_i = vec_nodes_f_i[col_index_i]
            f_a = current_link[3]
            
            ''''Test for optimal strategy'''
            if u_i >= min_u_j_and_c_a:
              
                if f_a == inf or f_i == inf: # for the case where the modification is needed in Spiess (1989) for no waiting time
                #if f_a == inf:
                    vec_nodes_u_i[col_index_i] = min_u_j_and_c_a 
                    vec_nodes_f_i[col_index_i] = inf
                    mx_A_bar_strategy_lines = np.vstack((mx_A_bar_strategy_lines, current_link))
                    df_opt_strat_alg.loc[num_transit_links-counter_S_list] = [mx_S_list[min_u_j_and_c_a_index,0],\
                                        mx_S_list[min_u_j_and_c_a_index,1],\
                                        f_a,\
                                        min_u_j_and_c_a,\
                                        True]
              
                else: # normal case when a link is added
                    vec_nodes_u_i[col_index_i] =\
                    (f_i_u_i_test(f_i,u_i, parameters_input['alpha_const_inter']) + f_a*(min_u_j_and_c_a))/(f_i+f_a)
                    vec_nodes_f_i[col_index_i] = f_i + f_a
                    mx_A_bar_strategy_lines = np.vstack((mx_A_bar_strategy_lines, current_link))
                    df_opt_strat_alg.loc[num_transit_links-counter_S_list] = [mx_S_list[min_u_j_and_c_a_index,0],\
                                mx_S_list[min_u_j_and_c_a_index,1],\
                                f_a,\
                                min_u_j_and_c_a,\
                                True]
                
            else:
                df_opt_strat_alg.loc[num_transit_links-counter_S_list] = [mx_S_list[min_u_j_and_c_a_index,0],\
                                mx_S_list[min_u_j_and_c_a_index,1],\
                                f_a,\
                                min_u_j_and_c_a,\
                                False]
                
            '''Remove the current link'''
            mx_S_list = np.delete(mx_S_list, min_u_j_and_c_a_index, axis = 0) # remove the current link from S_list
        
        '''Gets the expected travel time per passenger based on optimal strategies'''
        for origin_j in range(parameters_input['n']):
            if origin_j != i_destination:
                u_i_times[origin_j,i_destination] = df_opt_strat_alg[(df_opt_strat_alg.iloc[:,0] == origin_j) & df_opt_strat_alg.iloc[:,4]].iloc[0,3]
    
        
        #'''Assign demand according to optimal strategy'''
        # # Initialise the algorithm
        # # load the volumes of demand per node, called V_i
        
        # V_i = np.zeros(n_transit_nodes)
        # for i in range(parameters_input['n']):
        #     V_i[i] = mx_demand[i,r_destination]
        # V_i[r_destination] = - sum(V_i)
        
        # # NB this needs to hold to the conservation of flow requirements
        # # colnames(V_i) = names_all_nodes
        # # also the actual demand values can be input here
        
        # df_opt_strat_alg.insert(5, "v_a", np.zeros(len(df_opt_strat_alg)))
        
        #'''Load the links according to demand and frequencies'''
        # for i in range(len(df_opt_strat_alg)-1, -1, -1):  # for every link in decreasing order of u_j + c_a
        #     if df_opt_strat_alg.iloc[i,4]:
        
        #         if not int(df_opt_strat_alg.iloc[i, 0]) == r_destination: # this restricts the alg to assign negative demand to 
        #               # the outgoing nodes from the node that is being evaluated
        #               # also note, errors might come in when demand is wrongfully assigned out, and in.
                      
        #             # set the indices
        #             node_i_index = int(df_opt_strat_alg.iloc[i, 0])
        #             node_j_index = int(df_opt_strat_alg.iloc[i, 1])
                    
        #             # assign the v_a values
        #             if not df_opt_strat_alg.iloc[i,2] == inf :
        #                 df_opt_strat_alg.iloc[i, 5] = (df_opt_strat_alg.iloc[i,2]/\
        #                                                   vec_nodes_f_i[node_i_index])*V_i[node_i_index]
        #             else:
        #                 df_opt_strat_alg.iloc[i, 5] = V_i[node_i_index]
                                
        #             # assign the V_j values                                                
        #             V_i[node_j_index] = V_i[node_j_index] + df_opt_strat_alg.iloc[i, 5]                                                                                        
        
        # # Update the volumes overall
        # mx_volumes_nodes = mx_volumes_nodes + V_i
        
        # counter_link = 0  
        # while counter_link < len(df_opt_strat_alg):
        #     if df_opt_strat_alg.iloc[counter_link,4]:
        #         mx_volumes_links[int(df_opt_strat_alg.iloc[counter_link,0]), int(df_opt_strat_alg.iloc[counter_link,1])] =\
        #         mx_volumes_links[int(df_opt_strat_alg.iloc[counter_link,0]), int(df_opt_strat_alg.iloc[counter_link,1])] + df_opt_strat_alg.iloc[counter_link,5]
        
        #     counter_link = counter_link + 1 
    
        # end the overall destination change for loop spanning from 6.)
    
        """Adds all u_i end times to an array for final u_i times"""
        vec_nodes_u_i_ALL_array[:,i_destination] = vec_nodes_u_i
      
    #'''Add the volume per arc details to the list_transit_links object'''
    # df_transit_links.insert(4, "v_a", np.zeros(len(df_transit_links)))
    
    # for i in range(len(df_transit_links)):
    #     df_transit_links.iloc[i,4] = mx_volumes_links[int(dict_all_nodes[df_transit_links.iloc[i,0]]),\
    #                          int(dict_all_nodes[df_transit_links.iloc[i,1]])]

    # F3 Expected Travel Time
    return sum(sum(mx_demand*vec_nodes_u_i_ALL_array[:parameters_input['n'],:]))/(parameters_input['total_demand']*2)

# %% f3_ETT objective function 

def f3_ETT(R_routes, F_x, mx_dist, mx_demand, parameters_input, mx_walk=False):
    '''Create the transit network'''
    R_routes_named = format_routes_with_letters(R_routes)
    names_of_transit_routes = get_names_of_routes_as_list(R_routes_named)
    names_all_transit_nodes = list(map(str, range(parameters_input['n'])))+names_of_transit_routes
    n_transit_nodes = len(names_all_transit_nodes)
    
    vec_nodes_u_i_ALL_array = np.empty((n_transit_nodes,parameters_input['n'])) # creates array to contain all the final u_i values
    
    dict_all_nodes = dict() # creates a dictionary to map all the transit nodes to numbers
    for i in range(len(names_all_transit_nodes)):
        dict_all_nodes[names_all_transit_nodes[i]] = i
         
    mx_transit_network = np.zeros(shape=(n_transit_nodes, n_transit_nodes))
    mx_C_a = parameters_input['large_dist']*np.ones(shape=(n_transit_nodes, n_transit_nodes))
    mx_f_a = np.zeros(shape=(n_transit_nodes, n_transit_nodes))
    
    
    # Did not  name the numpy arrays
    
    '''Fill in the walk links that are present in the graph'''
    
    if hasattr(mx_walk, "__len__"):
        for i in range(parameters_input['n']):  
            for j in range(parameters_input['n']):
                if mx_dist[i,j] == parameters_input['large_dist']:
                    mx_C_a[i,j] = mx_walk[i,j]
                else:
                    if mx_dist[i,j] == 0:
                        mx_transit_network[i,j] = 0
                        mx_C_a[i,j] = mx_walk[i,j]
                        mx_f_a[i,j] = 0
                    else:
                        mx_transit_network[i,j] = 1
                        mx_C_a[i,j] = mx_walk[i,j]
                        mx_f_a[i,j] = inf
                        
    else:
        for i in range(parameters_input['n']):  
            for j in range(parameters_input['n']):
                if mx_dist[i,j] == parameters_input['large_dist']:
                    mx_C_a[i,j] = mx_dist[i,j]
                else:
                    if mx_dist[i,j] == 0:
                        mx_transit_network[i,j] = 0
                        mx_C_a[i,j] = parameters_input['walkFactor']*mx_dist[i,j]
                        mx_f_a[i,j] = 0
                    else:
                        mx_transit_network[i,j] = 1
                        mx_C_a[i,j] = parameters_input['walkFactor']*mx_dist[i,j]
                        mx_f_a[i,j] = inf
                    # else:
                    #     """Test for weird results"""
                    #     mx_transit_network[i,j] = 0
                    #     mx_C_a[i,j] = parameters_input['large_dist']
                    #     mx_f_a[i,j] = 0
    
    ''''Fill in the boarding links characteristics'''
    counter = 0
    
    for i in range(len(R_routes)): 
        for j in range(len(R_routes[i])):       
            i_index =  int(re.findall(r'\d+', names_of_transit_routes[counter])[0]) # number of the transit node
            j_index = names_all_transit_nodes.index(names_of_transit_routes[counter]) # position of the transit node in network graph
    
            mx_transit_network[i_index, j_index] = 1   
            mx_C_a[i_index, j_index] = parameters_input['boardingTime'] # sets the boarding
            mx_f_a[i_index, j_index] = F_x[i] # set the frequencies per transit line
      
            counter = counter + 1  
    
    '''Fill in the alighting links characteristics'''
    counter = 0
    
    for i in range(len(R_routes)): 
        for j in range(len(R_routes[i])):       
            i_index =  int(re.findall(r'\d+', names_of_transit_routes[counter])[0]) # number of the transit node
            j_index = names_all_transit_nodes.index(names_of_transit_routes[counter]) # position of the transit node in network graph
    
            mx_transit_network[j_index, i_index] = 1   
            mx_C_a[j_index, i_index] = parameters_input['alightingTime'] # sets the alighting
            mx_f_a[j_index, i_index] = inf # set the frequencies per transit line
      
            counter = counter + 1 
     
    '''Fill in the travel times using the transit lines / routes'''       
    
    for i in range(len(names_of_transit_routes) - 1):
        if " ".join(re.findall("[a-zA-Z]+", names_of_transit_routes[i]))==\
        " ".join(re.findall("[a-zA-Z]+", names_of_transit_routes[i+1])):
             
            i_index = names_all_transit_nodes.index(names_of_transit_routes[i])
            j_index = names_all_transit_nodes.index(names_of_transit_routes[i+1])
        
            mx_transit_network[i_index, j_index] =\
            mx_transit_network[j_index, i_index] = 1 
           
            mx_C_a[i_index, j_index] =\
            mx_C_a[j_index, i_index] =\
            mx_dist[int(re.findall(r'\d+', names_of_transit_routes[i])[0]),\
            int(re.findall(r'\d+', names_of_transit_routes[i+1])[0])]
        
            mx_f_a[i_index, j_index] =\
            mx_f_a[j_index, i_index] = inf
      
    '''Put all the links in one matrix'''  
    df_transit_links = pd.DataFrame(columns = ["I_i", "I_j", "c_a","f_a"])
    
    counter = 0
    for i in range(n_transit_nodes):
        for j in range(n_transit_nodes):
            if mx_transit_network[i,j]:
          
                df_transit_links.loc[counter] = [names_all_transit_nodes[i],\
                                     names_all_transit_nodes[j], mx_C_a[i,j], mx_f_a[i,j]]                 
                counter = counter + 1
    
    del counter, i, i_index, j, j_index    
    

    '''Optimal strategy algorithm (Spiess, 1989)'''    
    mx_volumes_nodes = np.zeros(n_transit_nodes) # create object to keep the node volumes
    mx_volumes_links = np.zeros(shape=(n_transit_nodes, n_transit_nodes)) # create object to keep the arc volumes
    names_main_nodes = names_all_transit_nodes[0:parameters_input['n']]
    
    u_i_times = np.zeros(shape=(parameters_input['n'],parameters_input['n']))
    
        # Overall loop to change the destinations
    for i_destination in range(parameters_input['n']): 
        
        # Create the data frames to keep the answers in
        df_opt_strat_alg = pd.DataFrame(columns = ["a=(i,","j)","f_a","u_j+c_a","a_in_A_bar"])
        vec_nodes_u_i = np.ones(n_transit_nodes)*inf
        vec_nodes_f_i = np.zeros(n_transit_nodes)
        
        # Set values of the first row
        r_destination = i_destination
        num_transit_links = len(df_transit_links)
        vec_nodes_u_i[r_destination] = 0 # set the destination expected time
        df_S_list = df_transit_links.copy() # creates a copy to work with
        
        for i in range(num_transit_links):
            df_S_list.iloc[i,0] = dict_all_nodes[df_S_list.iloc[i,0]]
            df_S_list.iloc[i,1] = dict_all_nodes[df_S_list.iloc[i,1]]
        
        mx_S_list = df_S_list.values # cast as numpy array for speed in calculations
        mx_S_list = np.hstack((mx_S_list, np.arange(num_transit_links).reshape(num_transit_links,1))) #adds indices
        
        df_A_bar_strategy_lines = pd.DataFrame(columns = ["I_i", "I_j", "c_a","f_a"])
        mx_A_bar_strategy_lines = np.empty(shape=(0,4))
        
        # repeats steps 6.2 and 6.3 until df_S_list is empty
        
        '''Get the next link'''
        
        for counter_S_list in range(num_transit_links-1, -1, -1):
            
            for i in range(counter_S_list+1): # loop through mx_S_list to find the minimum u_j + c_a
              
                if i == 0:
                    u_j = vec_nodes_u_i[int(mx_S_list[i,1])]
                    c_a = mx_S_list[i,2]
                    min_u_j_and_c_a = u_j + c_a
                    min_u_j_and_c_a_index = i
                
                else:
                    u_j = vec_nodes_u_i[int(mx_S_list[i,1])]
                    c_a = mx_S_list[i,2]
                    
                    if u_j + c_a <= min_u_j_and_c_a:
                      
                        min_u_j_and_c_a = u_j + c_a
                        min_u_j_and_c_a_index = i
                      
            '''Update the node label'''
            current_link = mx_S_list[min_u_j_and_c_a_index,:4] 
        
                
            col_index_i = int(current_link[0])
            u_i = vec_nodes_u_i[col_index_i]
            f_i = vec_nodes_f_i[col_index_i]
            f_a = current_link[3]
            
            ''''Test for optimal strategy'''
            if u_i >= min_u_j_and_c_a:
              
                if f_a == inf or f_i == inf: # for the case where the modification is needed in Spiess (1989) for no waiting time
                #if f_a == inf:
                    vec_nodes_u_i[col_index_i] = min_u_j_and_c_a 
                    vec_nodes_f_i[col_index_i] = inf
                    mx_A_bar_strategy_lines = np.vstack((mx_A_bar_strategy_lines, current_link))
                    df_opt_strat_alg.loc[num_transit_links-counter_S_list] = [mx_S_list[min_u_j_and_c_a_index,0],\
                                        mx_S_list[min_u_j_and_c_a_index,1],\
                                        f_a,\
                                        min_u_j_and_c_a,\
                                        True]
              
                else: # normal case when a link is added
                    vec_nodes_u_i[col_index_i] =\
                    (f_i_u_i_test(f_i,u_i, parameters_input['alpha_const_inter']) + f_a*(min_u_j_and_c_a))/(f_i+f_a)
                    vec_nodes_f_i[col_index_i] = f_i + f_a
                    mx_A_bar_strategy_lines = np.vstack((mx_A_bar_strategy_lines, current_link))
                    df_opt_strat_alg.loc[num_transit_links-counter_S_list] = [mx_S_list[min_u_j_and_c_a_index,0],\
                                mx_S_list[min_u_j_and_c_a_index,1],\
                                f_a,\
                                min_u_j_and_c_a,\
                                True]
                
            else:
                df_opt_strat_alg.loc[num_transit_links-counter_S_list] = [mx_S_list[min_u_j_and_c_a_index,0],\
                                mx_S_list[min_u_j_and_c_a_index,1],\
                                f_a,\
                                min_u_j_and_c_a,\
                                False]
                
            '''Remove the current link'''
            mx_S_list = np.delete(mx_S_list, min_u_j_and_c_a_index, axis = 0) # remove the current link from S_list
        
        '''Gets the expected travel time per passenger based on optimal strategies'''
        for origin_j in range(parameters_input['n']):
            if origin_j != i_destination:
                u_i_times[origin_j,i_destination] = df_opt_strat_alg[(df_opt_strat_alg.iloc[:,0] == origin_j) & df_opt_strat_alg.iloc[:,4]].iloc[0,3]
    
        
        #'''Assign demand according to optimal strategy'''
        # # Initialise the algorithm
        # # load the volumes of demand per node, called V_i
        
        # V_i = np.zeros(n_transit_nodes)
        # for i in range(parameters_input['n']):
        #     V_i[i] = mx_demand[i,r_destination]
        # V_i[r_destination] = - sum(V_i)
        
        # # NB this needs to hold to the conservation of flow requirements
        # # colnames(V_i) = names_all_nodes
        # # also the actual demand values can be input here
        
        # df_opt_strat_alg.insert(5, "v_a", np.zeros(len(df_opt_strat_alg)))
        
        #'''Load the links according to demand and frequencies'''
        # for i in range(len(df_opt_strat_alg)-1, -1, -1):  # for every link in decreasing order of u_j + c_a
        #     if df_opt_strat_alg.iloc[i,4]:
        
        #         if not int(df_opt_strat_alg.iloc[i, 0]) == r_destination: # this restricts the alg to assign negative demand to 
        #               # the outgoing nodes from the node that is being evaluated
        #               # also note, errors might come in when demand is wrongfully assigned out, and in.
                      
        #             # set the indices
        #             node_i_index = int(df_opt_strat_alg.iloc[i, 0])
        #             node_j_index = int(df_opt_strat_alg.iloc[i, 1])
                    
        #             # assign the v_a values
        #             if not df_opt_strat_alg.iloc[i,2] == inf :
        #                 df_opt_strat_alg.iloc[i, 5] = (df_opt_strat_alg.iloc[i,2]/\
        #                                                   vec_nodes_f_i[node_i_index])*V_i[node_i_index]
        #             else:
        #                 df_opt_strat_alg.iloc[i, 5] = V_i[node_i_index]
                                
        #             # assign the V_j values                                                
        #             V_i[node_j_index] = V_i[node_j_index] + df_opt_strat_alg.iloc[i, 5]                                                                                        
        
        # # Update the volumes overall
        # mx_volumes_nodes = mx_volumes_nodes + V_i
        
        # counter_link = 0  
        # while counter_link < len(df_opt_strat_alg):
        #     if df_opt_strat_alg.iloc[counter_link,4]:
        #         mx_volumes_links[int(df_opt_strat_alg.iloc[counter_link,0]), int(df_opt_strat_alg.iloc[counter_link,1])] =\
        #         mx_volumes_links[int(df_opt_strat_alg.iloc[counter_link,0]), int(df_opt_strat_alg.iloc[counter_link,1])] + df_opt_strat_alg.iloc[counter_link,5]
        
        #     counter_link = counter_link + 1 
    
        # end the overall destination change for loop spanning from 6.)
    
        """Adds all u_i end times to an array for final u_i times"""
        vec_nodes_u_i_ALL_array[:,i_destination] = vec_nodes_u_i
      
    #'''Add the volume per arc details to the list_transit_links object'''
    # df_transit_links.insert(4, "v_a", np.zeros(len(df_transit_links)))
    
    # for i in range(len(df_transit_links)):
    #     df_transit_links.iloc[i,4] = mx_volumes_links[int(dict_all_nodes[df_transit_links.iloc[i,0]]),\
    #                          int(dict_all_nodes[df_transit_links.iloc[i,1]])]

    # F3 Expected Travel Time
    return sum(sum(mx_demand*vec_nodes_u_i_ALL_array[:parameters_input['n'],:]))/(parameters_input['total_demand']*2)

#%% f4_TBR objective function

def f4_TBR(R_routes, F_x, mx_dist, sum_boolean=True):
    # returns a vector of the lengths of the individual routes
    len_r = len(R_routes)
    routeLengths = np.zeros(len_r)
    
    for i in range(len_r):
        path = R_routes[i]
        dist = 0
    
        for j in range(len(path)-1):
            dist = dist + mx_dist[path[j] , path[j+1]]
    
        routeLengths[i] = dist
    
    #F4 Total Number of Buses Required
    if sum_boolean:
        return sum(2*F_x*routeLengths)
    else:
        return 2*F_x*routeLengths

def calc_fn_obj_for_np_array(fn_obj_row, variables):
    """A function for calculating the obj function for all the rows in an array"""
    objectives = np.empty((0,2)) # second entry is the number of objectives
    for variable in variables:  
        objectives = np.vstack([objectives, fn_obj_row(variable)])
    
    return objectives

#%% Class: Transit_network 

class Transit_network():
    """A class containing all the information about the transit network""" 
    def __init__(self, R_routes, F_x, mx_dist, mx_demand, parameters_input):
        
        # Define the adjacent mapping of each node
        self.mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes

        self.df_opt_strat_alg = []
        self.df_opt_strat_alg_named = []
        self.mx_A_bar_strategy_lines = []
        self.df_A_bar_strategy_lines = []
        
        R_routes_named = format_routes_with_letters(R_routes)
        names_of_transit_routes = get_names_of_routes_as_list(R_routes_named)
        names_all_transit_nodes = list(map(str, range(parameters_input['n'])))+names_of_transit_routes
        n_transit_nodes = len(names_all_transit_nodes)

        dict_all_nodes = dict() # creates a dictionary to map all the transit nodes to numbers
        for i in range(len(names_all_transit_nodes)):
            dict_all_nodes[names_all_transit_nodes[i]] = i
                
        mx_transit_network = np.zeros(shape=(n_transit_nodes, n_transit_nodes))
        mx_C_a = parameters_input['large_dist']*np.ones(shape=(n_transit_nodes, n_transit_nodes))
        mx_f_a = np.zeros(shape=(n_transit_nodes, n_transit_nodes))
        
        
        '''Fill in the walk links that are present in the graph'''
        for i in range(parameters_input['n']):  
            for j in range(parameters_input['n']):
                if mx_dist[i,j] == parameters_input['large_dist']:
                    mx_C_a[i,j] = mx_dist[i,j]
                else:
                    if mx_dist[i,j] == 0:
                        mx_transit_network[i,j] = 0
                        mx_C_a[i,j] = parameters_input['walkFactor']*mx_dist[i,j]
                        mx_f_a[i,j] = 0
                    else:
                        mx_transit_network[i,j] = 1
                        mx_C_a[i,j] = parameters_input['walkFactor']*mx_dist[i,j]
                        mx_f_a[i,j] = inf
        
        
        ''''Fill in the boarding links characteristics'''
        counter = 0
        
        for i in range(len(R_routes)): 
            for j in range(len(R_routes[i])):       
                i_index =  int(re.findall(r'\d+', names_of_transit_routes[counter])[0]) # number of the transit node
                j_index = names_all_transit_nodes.index(names_of_transit_routes[counter]) # position of the transit node in network graph
        
                mx_transit_network[i_index, j_index] = 1   
                mx_C_a[i_index, j_index] = parameters_input['boardingTime'] # sets the boarding
                mx_f_a[i_index, j_index] = F_x[i] # set the frequencies per transit line
          
                counter = counter + 1  
        
        '''Fill in the alighting links characteristics'''
        counter = 0
        
        for i in range(len(R_routes)): 
            for j in range(len(R_routes[i])):       
                i_index =  int(re.findall(r'\d+', names_of_transit_routes[counter])[0]) # number of the transit node
                j_index = names_all_transit_nodes.index(names_of_transit_routes[counter]) # position of the transit node in network graph
        
                mx_transit_network[j_index, i_index] = 1   
                mx_C_a[j_index, i_index] = parameters_input['alightingTime'] # sets the alighting
                mx_f_a[j_index, i_index] = inf # set the frequencies per transit line
          
                counter = counter + 1 
         
        '''Fill in the travel times using the transit lines / routes'''       
        
        for i in range(len(names_of_transit_routes) - 1):
            if " ".join(re.findall("[a-zA-Z]+", names_of_transit_routes[i]))==\
            " ".join(re.findall("[a-zA-Z]+", names_of_transit_routes[i+1])):
                 
                i_index = names_all_transit_nodes.index(names_of_transit_routes[i])
                j_index = names_all_transit_nodes.index(names_of_transit_routes[i+1])
            
                mx_transit_network[i_index, j_index] =\
                mx_transit_network[j_index, i_index] = 1 
               
                mx_C_a[i_index, j_index] =\
                mx_C_a[j_index, i_index] =\
                mx_dist[int(re.findall(r'\d+', names_of_transit_routes[i])[0]),\
                int(re.findall(r'\d+', names_of_transit_routes[i+1])[0])]
            
                mx_f_a[i_index, j_index] =\
                mx_f_a[j_index, i_index] = inf
          
        '''Put all the links in one matrix'''  
        df_transit_links = pd.DataFrame(columns = ["I_i", "I_j", "c_a","f_a"])
        
        counter = 0
        for i in range(n_transit_nodes):
            for j in range(n_transit_nodes):
                if mx_transit_network[i,j]:
              
                    df_transit_links.loc[counter] = [names_all_transit_nodes[i],\
                                         names_all_transit_nodes[j], mx_C_a[i,j], mx_f_a[i,j]]                 
                    counter = counter + 1
        
        self.R_routes = R_routes
        self.R_routes_named = R_routes_named
        self.names_of_transit_routes = names_of_transit_routes
        self.names_all_transit_nodes = names_all_transit_nodes
        self.n_transit_nodes = n_transit_nodes
        self.dict_all_nodes = dict_all_nodes
        self.mx_transit_network = mx_transit_network
        self.mx_C_a = mx_C_a
        self.mx_f_a = mx_f_a
        

        # %% Optimal strategy algorithm (Spiess, 1989)
        '''Optimal strategy algorithm (Spiess, 1989)'''
        
        mx_volumes_nodes = np.zeros(n_transit_nodes) # create object to keep the node volumes
        mx_volumes_links = np.zeros(shape=(n_transit_nodes, n_transit_nodes)) # create object to keep the arc volumes
        names_main_nodes = names_all_transit_nodes[0:parameters_input['n']]
        
            # Overall loop to change the destinations
        for i_destination in range(parameters_input['n']): 
            
            # Create the data frames to keep the answers in
            df_opt_strat_alg = pd.DataFrame(columns = ["a=(i,","j)","f_a","u_j+c_a","a_in_A_bar"])
            vec_nodes_u_i = np.ones(n_transit_nodes)*inf
            vec_nodes_f_i = np.zeros(n_transit_nodes)
            
            # Set values of the first row
            r_destination = i_destination
            num_transit_links = len(df_transit_links)
            vec_nodes_u_i[r_destination] = 0 # set the destination expected time
            df_S_list = df_transit_links.copy() # creates a copy to work with
            
            for i in range(num_transit_links):
                df_S_list.iloc[i,0] = dict_all_nodes[df_S_list.iloc[i,0]]
                df_S_list.iloc[i,1] = dict_all_nodes[df_S_list.iloc[i,1]]
            
            mx_S_list = df_S_list.values # cast as numpy array for speed in calculations
            mx_S_list = np.hstack((mx_S_list, np.arange(num_transit_links).reshape(num_transit_links,1))) #adds indices
            
            df_A_bar_strategy_lines = pd.DataFrame(columns = ["I_i", "I_j", "c_a","f_a"])
            mx_A_bar_strategy_lines = np.empty(shape=(0,4))
            
            # repeats steps 6.2 and 6.3 until df_S_list is empty
            
            '''Get the next link'''
            
            for counter_S_list in range(num_transit_links-1, -1, -1):
                
                for i in range(counter_S_list+1): # loop through mx_S_list to find the minimum u_j + c_a
                  
                    if i == 0:
                        u_j = vec_nodes_u_i[int(mx_S_list[i,1])]
                        c_a = mx_S_list[i,2]
                        min_u_j_and_c_a = u_j + c_a
                        min_u_j_and_c_a_index = i
                    
                    else:
                        u_j = vec_nodes_u_i[int(mx_S_list[i,1])]
                        c_a = mx_S_list[i,2]
                        
                        if u_j + c_a <= min_u_j_and_c_a:
                          
                            min_u_j_and_c_a = u_j + c_a
                            min_u_j_and_c_a_index = i
                          
                '''Update the node label'''
                current_link = mx_S_list[min_u_j_and_c_a_index,:4] 
            
                    
                col_index_i = int(current_link[0])
                u_i = vec_nodes_u_i[col_index_i]
                f_i = vec_nodes_f_i[col_index_i]
                f_a = current_link[3]
                
                ''''Test for optimal strategy'''
                if u_i >= min_u_j_and_c_a:
                  
                    if f_a == inf or f_i == inf: # for the case where the modification is needed in Spiess (1989) for no waiting time
                    #if f_a == inf:
                        vec_nodes_u_i[col_index_i] = min_u_j_and_c_a 
                        vec_nodes_f_i[col_index_i] = inf
                        mx_A_bar_strategy_lines = np.vstack((mx_A_bar_strategy_lines, current_link))
                        df_opt_strat_alg.loc[num_transit_links-counter_S_list] = [mx_S_list[min_u_j_and_c_a_index,0],\
                                            mx_S_list[min_u_j_and_c_a_index,1],\
                                            f_a,\
                                            min_u_j_and_c_a,\
                                            True]
                  
                    else: # normal case when a link is added
                        vec_nodes_u_i[col_index_i] =\
                        (f_i_u_i_test(f_i,u_i, parameters_input['alpha_const_inter']) + f_a*(min_u_j_and_c_a))/(f_i+f_a)
                        vec_nodes_f_i[col_index_i] = f_i + f_a
                        mx_A_bar_strategy_lines = np.vstack((mx_A_bar_strategy_lines, current_link))
                        df_opt_strat_alg.loc[num_transit_links-counter_S_list] = [mx_S_list[min_u_j_and_c_a_index,0],\
                                    mx_S_list[min_u_j_and_c_a_index,1],\
                                    f_a,\
                                    min_u_j_and_c_a,\
                                    True]
                    
                else:
                    df_opt_strat_alg.loc[num_transit_links-counter_S_list] = [mx_S_list[min_u_j_and_c_a_index,0],\
                                    mx_S_list[min_u_j_and_c_a_index,1],\
                                    f_a,\
                                    min_u_j_and_c_a,\
                                    False]
                    
                '''Remove the current link'''
                mx_S_list = np.delete(mx_S_list, min_u_j_and_c_a_index, axis = 0) # remove the current link from S_list
                
            
            '''Assign demand according to optimal strategy'''
            # Initialise the algorithm
            # load the volumes of demand per node, called V_i
            
            V_i = np.zeros(n_transit_nodes)
            for i in range(parameters_input['n']):
                V_i[i] = mx_demand[i,r_destination]
            V_i[r_destination] = - sum(V_i)
            
            # NB this needs to hold to the conservation of flow requirements
            # colnames(V_i) = names_all_nodes
            # also the actual demand values can be input here
            
            df_opt_strat_alg.insert(5, "v_a", np.zeros(len(df_opt_strat_alg)))
            
            #df_opt_strat_alg = copy.deepcopy(df_opt_strat_alg) # an extra as used in R
            
            ''''TO DO'''
            # ctrl F and R : df_opt_strat_alg and df_opt_strat_alg
            
            '''Load the links according to demand and frequencies'''
            for i in range(len(df_opt_strat_alg)-1, -1, -1):  # for every link in decreasing order of u_j + c_a
                if df_opt_strat_alg.iloc[i,4]:
            
                    if not int(df_opt_strat_alg.iloc[i, 0]) == r_destination: # this restricts the alg to assign negative demand to 
                          # the outgoing nodes from the node that is being evaluated
                          # also note, errors might come in when demand is wrongfully assigned out, and in.
                          
                        # set the indices
                        node_i_index = int(df_opt_strat_alg.iloc[i, 0])
                        node_j_index = int(df_opt_strat_alg.iloc[i, 1])
                        
                        # assign the v_a values
                        if not df_opt_strat_alg.iloc[i,2] == inf :
                            df_opt_strat_alg.iloc[i, 5] = (df_opt_strat_alg.iloc[i,2]/\
                                                              vec_nodes_f_i[node_i_index])*V_i[node_i_index]
                        else:
                            df_opt_strat_alg.iloc[i, 5] = V_i[node_i_index]
                                    
                        # assign the V_j values                                                
                        V_i[node_j_index] = V_i[node_j_index] + df_opt_strat_alg.iloc[i, 5]                                                                                        
            
            # Update the volumes overall
            mx_volumes_nodes = mx_volumes_nodes + V_i
            
            counter_link = 0  
            while counter_link < len(df_opt_strat_alg):
                if df_opt_strat_alg.iloc[counter_link,4]:
                    mx_volumes_links[int(df_opt_strat_alg.iloc[counter_link,0]), int(df_opt_strat_alg.iloc[counter_link,1])] =\
                    mx_volumes_links[int(df_opt_strat_alg.iloc[counter_link,0]), int(df_opt_strat_alg.iloc[counter_link,1])] + df_opt_strat_alg.iloc[counter_link,5]
            
              
                counter_link = counter_link + 1 
         
            '''Name the nodes according to the routes'''
            for i in range(len(mx_A_bar_strategy_lines)):
                df_A_bar_strategy_lines.loc[i] = [names_all_transit_nodes[int(mx_A_bar_strategy_lines[i,0])],\
                                                   names_all_transit_nodes[int(mx_A_bar_strategy_lines[i,1])],\
                                                                           mx_A_bar_strategy_lines[i,2],\
                                                                           mx_A_bar_strategy_lines[i,3]]
                    
            df_opt_strat_alg_named = copy.deepcopy(df_opt_strat_alg)
            
            for i in range(len(df_opt_strat_alg)):
                df_opt_strat_alg_named.iloc[i,0] = names_all_transit_nodes[int(df_opt_strat_alg.iloc[i,0])]
                df_opt_strat_alg_named.iloc[i,1] = names_all_transit_nodes[int(df_opt_strat_alg.iloc[i,1])]        
            
            """Attributes relating to the optimal assignment strategy per destination"""
            self.df_opt_strat_alg.append(df_opt_strat_alg)
            self.df_opt_strat_alg_named.append(df_opt_strat_alg_named)
            self.mx_A_bar_strategy_lines.append(mx_A_bar_strategy_lines)
            self.df_A_bar_strategy_lines.append(df_A_bar_strategy_lines)
        
        # end the overall destination change for loop spanning from 6.)
              
        '''Add the volume per arc details to the list_transit_links object'''
        df_transit_links.insert(4, "v_a", np.zeros(len(df_transit_links)))
        
        for i in range(len(df_transit_links)):
            df_transit_links.iloc[i,4] = mx_volumes_links[int(dict_all_nodes[df_transit_links.iloc[i,0]]),\
                                 int(dict_all_nodes[df_transit_links.iloc[i,1]])]
                
       
        """Attributes relating to the entire Transit Network"""
        self.mx_transit_network = mx_transit_network
        self.names_main_nodes = names_main_nodes
        self.df_transit_links = df_transit_links
        self.mx_volumes_links = mx_volumes_links
        self.mx_volumes_nodes = mx_volumes_nodes
        
    def test_reachability(self):
        # Test reachability matrix operations with boolean matrices 
        # Graphs and Boolean matrices in computer programming, B. Marimont, 1960
        reachability_list = list()
        for i in range(len(self.mx_transit_network)):
            reachability_list.append(np.linalg.matrix_power(self.mx_transit_network, i+1))
        return reachability_list

#%% Class: Routes

class Routes():
    """A class containing all the information about the routes"""   
    def __init__(self, R_routes: list):
        self.number_of_routes = len(R_routes)
        self.routes = R_routes
        
    def __str__(self) -> str:
        return gf.convert_routes_list2str(self.routes)
    
    def calc_route_lengths(self, mx_dist):
        # returns a vector of the lengths of the individual routes
        routeLengths = np.zeros(self.number_of_routes)
        
        for i in range(self.number_of_routes):
            path = self.routes[i]
            dist = 0
        
            for j in range(len(path)-1):
                dist = dist + mx_dist[path[j] , path[j+1]]
        
            routeLengths[i] = dist
        
        self.route_lengths = routeLengths
        return routeLengths 
    
    def plot_routes(self, main_problem):
        """A function that plots the routes of a problem based on the problem defined"""
        gv.plotRouteSet2(main_problem.problem_data.mx_dist, self.routes, main_problem.problem_data.mx_coords) # using iGraph
        
#%% Class: Frequencies2
 
class Frequencies(): 
    """A class containing all the information about the frequencies""" 
    # the idea is to encode the decision variables as the argument that is to be input into the 
    # theta set so that manipulations can be easier to conduct
    theta_set = np.array([5,6,7,8,9,10,12,14,16,18,20,25,30])
    theta_set_len = len(theta_set)
    
    def __init__(self, number_of_frequencies):
        self.number_of_frequencies = number_of_frequencies
        
    def set_frequencies(self, F_frequencies: float):
        self.number_of_frequencies = len(F_frequencies)
        self.frequencies = F_frequencies
                    
    def set_random_frequencies(self):
        F_x_arg = np.random.randint(0, len(self.theta_set), self.number_of_frequencies)
        self.frequencies = 1/self.theta_set[F_x_arg]
        
    def return_random_theta_frequencies(self):
        F_x_arg = np.random.randint(0, len(self.theta_set), self.number_of_frequencies)
        return 1/self.theta_set[F_x_arg]
        
    def return_random_theta_args(self):
        F_x_arg = np.random.randint(0, len(self.theta_set), self.number_of_frequencies)
        return F_x_arg
        
    def set_frequencies_all_equal(self, freq):
        self.frequencies = np.full(self.number_of_frequencies, freq)
        
    def do_random_vehicle_assignments(self, R_routes, mx_dist, fleet_size):
        ''' Vehicle assignment to routes for determining frequencies''' 
        # Spreads the availabe vehicles randomly over the routes
        # Returns the frequencies of the accociated assignments
        self.vec_vehicle_assignments = generate_random_vehicle_assignments(self.number_of_frequencies, fleet_size)
        return self.vec_vehicle_assignments / (2 * calc_route_lengths(mx_dist, R_routes))  # calculate the frequencies on each route


    
#%% Class: UTFSP_problem

class Problem_data():
    """A class for storing the data of a generic problem"""
    def __init__(self, mx_dist, mx_demand, mx_coords: list, mx_walk=False):
        self.mx_dist = mx_dist
        self.mx_demand = mx_demand
        self.mx_coords = mx_coords
        self.mx_walk = mx_walk
        
class Problem_constraints():
    """A class for storing the constraints of a generic problem"""
    def __init__(self, parameters_constraints: dict, **kwargs):
        for k, v in parameters_constraints.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)        
            
class Problem_inputs():
    """A class for storing the input parameters of a generic problem"""
    def __init__(self, parameters_input: dict, **kwargs):
        for k, v in parameters_input.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)
            
class Problem_GA_inputs():
    """A class for storing the input parameters of the Genetic Algorithm"""
    def __init__(self, parameters_input: dict, **kwargs):
        for k, v in parameters_input.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

class UTFSP_problem():
    """A class for storing all the information pertaining to a problem"""
    pass

