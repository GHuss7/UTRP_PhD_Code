# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:02:00 2019

@author: 17832020
"""

# %% Import Libraries
import os
import csv
import numpy as np
import pandas as pd 
import igraph as ig
import copy
import networkx as nx
import matplotlib as plt

#from timeit import default_timer as timer

# %% Import personal functions
import DSS_UTNDP_Functions as gf

# %% Format a graph's attributes

def format_igraph_custom_1(g_tn):  
    # Formats the graph according to some specifications
    g_tn.vs["size"] = [20]
    g_tn.vs["color"] = ["gray"]
    g_tn.vs["label"] = range(g_tn.ecount())
    g_tn.es["label"] = g_tn.es["distance"]
    return g_tn


# %% Create and plot a graph from a distance matrix

def plot_igraph_from_dist_mx(distance_matrix):
    
    # The links for future use
    links_list_dist_mx, links_list_distances = gf.get_links_list_and_distances(distance_matrix)
         
    # Create the transit network graph
    g_tn = gf.create_igraph_from_dist_mx(distance_matrix)
    
    # Set the layout
    g_tn_layout = g_tn.layout("kk")
    
    # Set the visual style dictionary for the plot
    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_color"] = "grey"
    visual_style["vertex_label"] = g_tn.vs["name"]
    # visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
    visual_style["layout"] = g_tn_layout
    visual_style["bbox"] = (600, 600)
    visual_style["margin"] = 20
    visual_style["edge_label"] = g_tn.es["distance"] # adds the distances to each edge
    # visual_style["edge_label"] = "" # uncomment and comment above to try and remove the edge weights
    
    return ig.plot(g_tn, **visual_style) # plot graph with visual style

# %% Plot an igraph 

def plot_igraph_custom_1(g_tn):
    
    # Set the layout
    g_tn_layout = g_tn.layout("kk")
    
    # Set the visual style dictionary for the plot
    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_color"] = "grey"
    visual_style["vertex_label"] = g_tn.vs["name"]
    # visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
    visual_style["layout"] = g_tn_layout
    visual_style["bbox"] = (600, 600)
    visual_style["margin"] = 20
    visual_style["edge_label"] = g_tn.es["distance"] # adds the distances to each edge
    # visual_style["edge_label"] = "" # uncomment and comment above to try and remove the edge weights
    
    return ig.plot(g_tn, **visual_style) # plot graph with visual style

# %% Plot an igraph with layout specified 

def plot_igraph_custom_2(g_tn, g_tn_layout):
    # g_tn is an iGraph object
    # g_tn_layout is a layout of g_tn
    
    # Set the visual style dictionary for the plot
    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_color"] = "grey"
    visual_style["vertex_label"] = g_tn.vs["name"]
    # visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
    visual_style["layout"] = g_tn_layout
    visual_style["bbox"] = (600, 600)
    visual_style["margin"] = 20
    visual_style["edge_label"] = g_tn.es["distance"] # adds the distances to each edge
    # visual_style["edge_label"] = "" # uncomment and comment above to try and remove the edge weights
    
    return ig.plot(g_tn, **visual_style) # plot graph with visual style

# %% Add and plot a route on an igraph

def plot_add_route_to_igraph_1(g_tn, route):

    # Set the layout
    g_tn_layout = g_tn.layout("kk")
    
    # Set the visual style dictionary for the plot
    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_color"] = "grey"
    visual_style["vertex_label"] = g_tn.vs["name"]
    # visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
    visual_style["layout"] = g_tn_layout
    visual_style["bbox"] = (600, 600)
    visual_style["margin"] = 20
    visual_style["edge_label"] = g_tn.es["distance"] # adds the distances to each edge
    # visual_style["edge_label"] = "" # uncomment and comment above to try and remove the edge weights
    
    g_tn.add_edges(route) # "edge_label" = range(len(route)), "edge_color" = "red"

    g_tn.es[g_tn.ecount():]["color"] = "red" # format the route to travel as red
    g_tn.es[g_tn.ecount():]["weight"] = np.arange(len(route)).tolist() # still not working properly, but try to assign the sequence of the route
    # as the edge labels for the route to travel


    return ig.plot(g_tn, **visual_style)
    
# %% Add and plot a route on an igraph with a specified layout

def plot_add_route_to_igraph_2(g_tn, route, g_tn_layout):
    # g_tn is an iGraph object
    # route is a list with tuple entries denoting the edges of the route to travel
    # g_tn_layout is a layout of g_tn
    
    # Set the visual style dictionary for the plot
    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_color"] = "grey"
    visual_style["vertex_label"] = g_tn.vs["name"]
    # visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
    visual_style["layout"] = g_tn_layout
    visual_style["bbox"] = (600, 600)
    visual_style["margin"] = 20
    visual_style["edge_label"] = g_tn.es["distance"] # adds the distances to each edge
    # visual_style["edge_label"] = "" # uncomment and comment above to try and remove the edge weights
    
    g_tn.add_edges(route) # "edge_label" = range(len(route)), "edge_color" = "red"

    g_tn.es[g_tn.ecount():]["color"] = "red" # format the route to travel as red
    g_tn.es[g_tn.ecount():]["weight"] = np.arange(len(route)).tolist() # still not working properly, but try to assign the sequence of the route
    # as the edge labels for the route to travel


    return ig.plot(g_tn, **visual_style)   

# %% Add route to an iGraph

def add_one_route_to_igraph(g_tn, route):
    # route should be in the form of a list of tuples
    g_tn.add_edges(route) # "edge_label" = range(len(route)), "edge_color" = "red"
    g_tn.es[g_tn.ecount()-len(route):]["color"] = "red" # format the route to travel as red
    g_tn.es[g_tn.ecount()-len(route):]["weight"] = np.arange(len(route)).tolist()
    
    return g_tn

# %% Save graph as SVG in current directory

def save_svg_igraph_1(g_tn, route, g_tn_layout, file_name):
    # g_tn is an iGraph object
    # route is a list with tuple entries denoting the edges of the route to travel
    # g_tn_layout is a layout of g_tn
    
    # Set the visual style dictionary for the plot
    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_color"] = "grey"
    visual_style["vertex_label"] = g_tn.vs["name"]
    # visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
    visual_style["layout"] = g_tn_layout
    visual_style["bbox"] = (600, 600)
    visual_style["margin"] = 20
    visual_style["edge_label"] = g_tn.es["distance"] # adds the distances to each edge
    # visual_style["edge_label"] = "" # uncomment and comment above to try and remove the edge weights
    
    g_tn.add_edges(route) # "edge_label" = range(len(route)), "edge_color" = "red"

    g_tn.es[g_tn.ecount():]["color"] = "red" # format the route to travel as red
    g_tn.es[g_tn.ecount():]["weight"] = np.arange(len(route)).tolist() # still not working properly, but try to assign the sequence of the route
    # as the edge labels for the route to travel


    return g_tn.write_svg(file_name + ".svg", **visual_style)

# %% Add routes to igraph
    
def add_route_edges_to_igraph(g_tn, routes_R):
    # adds the edges with different colours (up to 10 routes) and returns the graph
    
    links_list_route_R = list()
    for i in  range(len(routes_R)):
        links_list_route_R_i = list()
        for j in  range(len(routes_R[i])-1):
            links_list_route_R_i.append((routes_R[i][j],routes_R[i][j+1]))
        links_list_route_R.append(links_list_route_R_i)
        links_list_route_R_i = list()
    del links_list_route_R_i
    
    colours = ["red","lime","blue","gold","darkorange","magenta","cyan","brown","gray","black"]
    
    for i in range(len(routes_R)):
        g_tn.add_edges(links_list_route_R[i]) # "edge_label" = range(len(route)), "edge_color" = "red"
        g_tn.es[g_tn.ecount()-len(links_list_route_R[i]):]["color"] = colours[i] # format the route to travel as red
        # g_tn.es[g_tn.ecount():]["weight"] = could_add_later     
    return g_tn

# %% Print graph in seperate window
if False:
    g_tn_layout = g_tn.layout("kk") 
    format_igraph_custom_1(g_tn)
    ig.plot(g_tn, inline = False, layout = g_tn_layout)

# %% Extra notes
    
#from igraph import *
#g = Graph.Tree(7, 2)
#layout = g.layout_reingold_tilford(mode="in", root=[0])
#g.vs["size"] = [60]
#g.vs["color"] = ["green", "red", "blue", "yellow"]
#g.vs["label"] = ["’Matthew’", "’John’", "’Luke’", "’Mark’","’Paul’", "’James’", "’Eliah’", "’Emanuel’"]
#g.write_svg("first.svg", layout=layout, vertex_size=20)
#plot(g, layout=layout,bbox = (500, 300), margin = 50) 
#
#plot(g)
    
# %% Plots graph in seperate window
def plotRouteSet(mx_dist,routes_R):
    # this function takes as input the distance mx and route set and plots it
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
    g_tn_layout = g_tn.layout("kk")
    format_igraph_custom_1(g_tn)    
    add_route_edges_to_igraph(g_tn, routes_R)
    ig.plot(g_tn, inline=False, layout=g_tn_layout)  # switch inline to false if you want to print inline   


def plotRouteSet2(mx_dist,routes_R, mx_coords):
    # this function takes as input the distance mx and route set and plots it
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
    g_tn_layout = g_tn.layout("kk")
    format_igraph_custom_1(g_tn)    
    add_route_edges_to_igraph(g_tn, routes_R)
    ig.plot(g_tn, inline=False, layout=mx_coords)  # switch inline to false if you want to print inline   

def plotRouteSet3(mx_dist,routes_R, mx_coords):
    # this function takes as input the distance mx and route set and plots it
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
    g_tn_layout = g_tn.layout("kk")
    format_igraph_custom_1(g_tn)    
    add_route_edges_to_igraph(g_tn, routes_R)
    ig.plot(g_tn, inline=True, layout=mx_coords)  # switch inline to false if you want to print inline   

# %% Graph visualisation
'''Create networkx graph'''
def create_nx_graph_from_dist_mx(mx_dist, mx_coords):
    links_list_dist_mx, links_list_distances = gf.get_links_list_and_distances(mx_dist)
    G = nx.Graph()
    for i in range(len(mx_dist)): # add nodes
        G.add_node(i, pos = mx_coords[i])
    
    for i in range(len(links_list_dist_mx)): # add edges
        G.add_edge(links_list_dist_mx[i][0], links_list_dist_mx[i][1], weight = links_list_distances[i])

    return G

def plot_nx_graph_with_labels(G):
    pos = nx.get_node_attributes(G,'pos')
    labels = nx.get_edge_attributes(G,'weight')
    
    nx.draw(G, with_labels=True, pos=pos)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show() # display

''' Create minimum spanning tree from networkx graph'''
#def create_min_spanning_tree_from_nx_graph(G):
#    G2 = nx.minimum_spanning_tree(G)
#    labels = nx.get_edge_attributes(G2,'weight')
#    nx.draw(G2, with_labels=True, pos=pos)
#    nx.draw_networkx_edge_labels(G2,pos,edge_labels=labels)
#    plt.show() # display
#    G2.size(weight='weight')

#%% Function: Visualisation of generations in UTRP GA
def plot_generations_objectives(pop_generations):
    # function to visualise the different populations per generation
    # gets difficult to visualise and plot when generations reach more than 10
    
    df_to_plot = pd.DataFrame()
    df_to_plot = df_to_plot.assign(f_1 = pop_generations[:,0],
                      f_2 = pop_generations[:,1],
                      generation = pop_generations[:,3])
    
    plt.style.use('seaborn-whitegrid')
    
    groups = df_to_plot.groupby("generation")
    for name, group in groups:
        plt.plot(group["f_1"], group["f_2"], marker="o", linestyle="", label=name)
    plt.legend()
