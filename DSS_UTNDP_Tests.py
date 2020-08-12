# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:19:08 2019

@author: 17832020
"""
# %% Import Libraries
import os
import pandas as pd 
from timeit import default_timer as timer
import datetime

'''We can use dir() function to get a list containing all attributes of a module.'''
print(dir(datetime.datetime))

# %% Tests



# %% Create a graph and plot it

# The links for future use
links_list_dist_mx, links_list_distances = gf.get_links_list_and_distances(mx_dist)
         
# Create the transit network graph
g_tn = gf.create_igraph_from_dist_mx(mx_dist)
g_tn_layout = g_tn.layout("kk")

# %% Old way of doing SA Route Design
mx_dist_bus_network = gf.generate_bus_network_dist_mx(routes_R, mx_dist)
g_btn = gf.create_igraph_from_dist_mx(mx_dist_bus_network)  # generate the bus transit network
paths_shortest_bus_routes = gf.get_all_shortest_paths(g_btn)
mx_shortest_bus_distances = gf.calculate_shortest_dist_matrix(paths_shortest_bus_routes, mx_dist_bus_network)
paths_shortest_bus_routes = gf.remove_duplicate_routes(paths_shortest_bus_routes, len(mx_dist))
mx_transfer = gf.generate_transfer_mx(routes_R, paths_shortest_bus_routes, len(mx_dist))

# %% Other things to potentially use again

path_to_travel = [(8,14),(14,6),(6,9),(9,13),(13,12),(12,10),(10,11),(11,3),(3,4),(4,1),(1,0),(0,1),(1,2),(2,5),(5,14)] # input route to travel

gv.format_igraph_custom_1(g_tn)

g_n = copy.deepcopy(g_tn) # creates a copy to work with later

gv.add_route_to_igraph(g_tn,path_to_travel)

ig.plot(g_tn, inline = False, layout = g_tn_layout)

# %% Transfer matrix tests

from itertools import compress

t = [False, False, False, False, True, True, False, True, False, False, False, False, False, False, False, False]
list(compress(range(len(t)), t))
t = t*10
%timeit [i for i, x in enumerate(t) if x]

%timeit list(compress(range(len(t)), t))

# %% Test to validate
if False:
    x_Mandl = list()
    x_Mandl.append([0,1,2,5,7,9,10,12])
    x_Mandl.append([4,3,5,7,14,6])
    x_Mandl.append([11,3,5,14,8])
    x_Mandl.append([12,13,9])
    
    routes_R = x_Mandl
    # routes_R = routes

# %% Feasibility tests


# todo --- nx is faster than my own algorithm -- incorporate it
# todo --- get_graph_node_and_edge_connectivity(routes_R) could be used in guiding repairing strategies


#%% Append row tests

%timeit df_archive.loc[5] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)] # adds the new solution

dict1 = {'ATT':f_cur[0], 'TRT':f_cur[1], 'Routes':gf.convert_routes_list2str(routes_R)}
%timeit df_archive.append(dict1, ignore_index=True)

%timeit df_archive.append(pd.Series({'ATT':f_cur[0], 'TRT':f_cur[1], 'Routes':gf.convert_routes_list2str(routes_R)}), ignore_index=True)


%timeit df_archive.reset_index(); df_archive.loc[len(df_archive)] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)] # adds the new solution
                    