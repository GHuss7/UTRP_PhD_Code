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
import matplotlib.pyplot as plt
import matplotlib as mplt
import seaborn as sns

mplt.rcParams['font.family'] = 'serif'
try:
    cmfont = mplt.font_manager.FontProperties(fname=mplt.get_data_path() + '/fonts/ttf/cmunrm.ttf')
    mplt.rcParams['font.serif'] = cmfont.get_name()
except:
    print('File not found: ' + mplt.get_data_path() + '/fonts/ttf/cmunrm.ttf')
mplt.rcParams['mathtext.fontset']= 'cm'
mplt.rcParams['font.size']= 11
mplt.rcParams['axes.unicode_minus']= False

#from timeit import default_timer as timer

# %% Import personal functions
import DSS_UTNDP_Functions as gf
import DSS_Admin as ga

# %% Format a graph's attributes

def format_igraph_custom_1(g_tn):  
    # Formats the graph according to some specifications
    g_tn.vs["size"] = [20]
    g_tn.vs["color"] = ["gray"]
    g_tn.vs["label"] = range(g_tn.ecount())
    g_tn.es["label"] = g_tn.es["distance"]
    return g_tn

def format_igraph_custom_2(g_tn):   # for transit network
    # Formats the graph according to some specifications
    g_tn.vs["size"] = [30]
    g_tn.vs["label_size"] = [20]
    g_tn.vs["color"] = ["gray"]
    g_tn.vs["label"] = range(g_tn.ecount())
    g_tn.es["label"] = g_tn.es["distance"]
    g_tn.es["label_size"] = [20]
    return g_tn

def format_igraph_custom_thesis(g_tn):  
    # Formats the graph according to some specifications
    g_tn.vs["size"] = [30]
    g_tn.vs["label_size"] = [20]
    g_tn.vs["color"] = ["gray"]
    g_tn.vs["label"] = range(g_tn.ecount())
    g_tn.es["label"] = g_tn.es["distance"]
    # g_tn.es["size"] = [30]
    g_tn.es["label_size"] = [15]
    return g_tn

def format_igraph_custom_experiment(g_tn):  
    # Formats the graph according to some specifications
    g_tn.vs["size"] = [30]
    g_tn.vs["label_size"] = [20]
    g_tn.vs["color"] = ["gray"]
    g_tn.vs["label"] = range(g_tn.ecount())
    g_tn.es["label"] = g_tn.es["distance"]
    # g_tn.es["size"] = [30]
    g_tn.es["label_size"] = [15]
    g_tn.es["curved"] = [0]
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
    
    colours = ["red","lime","blue","darkorange","magenta","cyan","green","gold",
               "brown","gray",'yellow', 'purple', 'pink', 'orange', 'teal', 
               'coral', 'lightblue', 'lavender', 'turquoise', 'darkgreen', 'tan', 
               'salmon', 'lightpurple', 'darkred', 'darkblue']
    
    for i in range(len(routes_R)):
        g_tn.add_edges(links_list_route_R[i]) # "edge_label" = range(len(route)), "edge_color" = "red"
        g_tn.es[g_tn.ecount()-len(links_list_route_R[i]):]["color"] = colours[i] # format the route to travel as red
        # g_tn.es[g_tn.ecount():]["weight"] = could_add_later     
    return g_tn


def add_route_edges_to_igraph_experiment(g_tn, routes_R):
    # adds the edges with different colours (up to 10 routes) and returns the graph
    
    links_list_route_R = list()
    for i in  range(len(routes_R)):
        links_list_route_R_i = list()
        for j in  range(len(routes_R[i])-1):
            links_list_route_R_i.append((routes_R[i][j],routes_R[i][j+1]))
        links_list_route_R.append(links_list_route_R_i)
        links_list_route_R_i = list()
    del links_list_route_R_i
    
    colours = ["red","lime","blue","darkorange","magenta","cyan","green","gold",
               "brown","gray",'yellow', 'purple', 'pink', 'orange', 'teal', 
               'coral', 'lightblue', 'lavender', 'turquoise', 'darkgreen', 'tan', 
               'salmon', 'lightpurple', 'darkred', 'darkblue']
    
      
    
    multiple_edges_mx = np.zeros(shape=(len(g_tn.vs()), len(g_tn.vs())))
    
    for edge in g_tn.es:
        multiple_edges_mx[edge.tuple[0], edge.tuple[1]] += 1
        multiple_edges_mx[edge.tuple[1], edge.tuple[0]] += 1
    
    #TODO: finish the specific edge's additions and set curved appropriately
    # https://igraph.org/python/doc/igraph.EdgeSeq-class.html
    
    # g_tn.es.is_multiple() a function that migth be of use
    
    
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
def plotRouteSet(mx_dist,routes_R,layout_style="kk"):
    # this function takes as input the distance mx and route set and plots it
    # different layout styles: https://igraph.org/python/doc/tutorial/tutorial.html#layouts-and-plotting
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
    g_tn_layout = g_tn.layout(layout_style)
    format_igraph_custom_1(g_tn)    
    add_route_edges_to_igraph(g_tn, routes_R)
    ig.plot(g_tn, inline=False, layout=g_tn_layout)  # switch inline to false if you want to print inline   


def plotRouteSet2(mx_dist,routes_R, mx_coords):
    # this function takes as input the distance mx and route set and plots it
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
    format_igraph_custom_1(g_tn)    
    add_route_edges_to_igraph(g_tn, routes_R)
    ig.plot(g_tn, inline=False, layout=mx_coords)  # switch inline to false if you want to print inline   

def plotRouteSetAndSavePDF_road_network(mx_dist,routes_R, mx_coords, name):
    # this function takes as input the distance mx and route set and plots it
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
    format_igraph_custom_2(g_tn)    
    add_route_edges_to_igraph(g_tn, routes_R)
    ig.plot(g_tn, f"Plots/{name}_plot.pdf", inline=False, layout=mx_coords)  # switch inline to false if you want to print inline   
    return g_tn

def plotRouteSetAndSavePDF(mx_dist,routes_R, mx_coords, name):
    # this function takes as input the distance mx and route set and plots it
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
    format_igraph_custom_thesis(g_tn)    
    add_route_edges_to_igraph(g_tn, routes_R)
    ig.plot(g_tn, f"Plots/{name}_plot.pdf", inline=False, layout=mx_coords)  # switch inline to false if you want to print inline   
    return g_tn

def plotUTFSPAndSavePDF(mx_dist,routes_R, mx_coords, name):
    # this function takes as input the distance mx and route set and plots it
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
    format_igraph_custom_thesis(g_tn)    
    add_route_edges_to_igraph(g_tn, routes_R)
    ig.plot(g_tn, f"Plots/{name}_plot.pdf", inline=False, layout=mx_coords)  # switch inline to false if you want to print inline   
    return g_tn

def plotRouteSet3(mx_dist,routes_R, mx_coords):
    # this function takes as input the distance mx and route set and plots it
    g_tn = gf.create_igraph_from_dist_mx(mx_dist)
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
    mplt.show() # display

''' Create minimum spanning tree from networkx graph'''
#def create_min_spanning_tree_from_nx_graph(G):
#    G2 = nx.minimum_spanning_tree(G)
#    labels = nx.get_edge_attributes(G2,'weight')
#    nx.draw(G2, with_labels=True, pos=pos)
#    nx.draw_networkx_edge_labels(G2,pos,edge_labels=labels)
#    mplt.show() # display
#    G2.size(weight='weight')

#%% Function: Visualisation of generations in UTRP GA
def plot_generations_objectives(pop_generations, every_n_gen=False, path=False):
    # function to visualise the different populations per generation
    # gets difficult to visualise and plot when generations reach more than 10
    if every_n_gen:
        df_to_plot = pd.DataFrame()
        df_to_plot = df_to_plot.assign(f_1 = pop_generations[:,0],
                      f_2 = pop_generations[:,1],
                      Generation = pop_generations[:,3]) 
        df_to_plot = df_to_plot.iloc[::every_n_gen, :]
        
    else:
        df_to_plot = pd.DataFrame()
        df_to_plot = df_to_plot.assign(f_1 = pop_generations[:,0],
                      f_2 = pop_generations[:,1],
                      Generation = pop_generations[:,3])
    
    # mplt.style.use('seaborn-whitegrid')
    
    try:
        groups = df_to_plot.groupby("Generation")
        for name, group in groups:
            mplt.plot(group["f_1"], group["f_2"], marker="o", linestyle="", label=name)
        mplt.legend()
    
        if path: 
            plt.ioff()
            plt.savefig(path /"Fronts_over_gens.pdf", bbox_inches='tight')
            plt.close()
    except:
        pass
    
def plot_generations_objectives_UTRP(df_pop_generations, every_n_gen=False, path=False):
    # function to visualise the different populations per generation
    # gets difficult to visualise and plot when generations reach more than 10
    if every_n_gen:
        df_to_plot = pd.DataFrame()
        df_to_plot = df_pop_generations.drop(['R_x', 'Rank'], axis=1)
        df_to_plot = df_pop_generations[df_pop_generations['Generation']%every_n_gen==0]
        
    else:
        df_to_plot = pd.DataFrame()
        df_to_plot = df_pop_generations.drop(['R_x', 'Rank'], axis=1)
    
    # Seaborn settings:    
    # mplt.style.use('seaborn-whitegrid')
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    
    try:
        sns.lmplot(x='f_1', y='f_2', data=df_to_plot, hue='Generation', fit_reg=False)
    
        if path: 
            # plt.show()
            plt.ioff()
            plt.savefig(path /"Fronts_over_gens.pdf", bbox_inches='tight')
            plt.close()
    except:
        pass   
    

#%% Print summary figures
def save_results_analysis_fig_interim_UTRP(initial_set, df_non_dominated_set, validation_data, df_data_generations, name_input_data, path_results_per_run, validation_line=False):
    '''Print Objective functions over time, all solutions and pareto set obtained'''
    f_1_col_name, f_2_col_name, f_1_label, f_2_label = "f_1", "f_2", "F_1_ATT", "F_2_TRT"
    
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(7.5)
    fig.set_figwidth(20)
    
    
    axs[0].plot(df_data_generations["Generation"], df_data_generations["HV"], marker="o", label='HV obtained')
    #axs[0, 1].scatter(range(len(df_SA_analysis)), np.ones(len(df_SA_analysis))*gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], UTRFSP_problem_1.max_objs, UTRFSP_problem_1.min_objs),\
    #   s=1, marker="o", label='HV Mumford (2013)')
    axs[0].set_title('HV over all generations')
    axs[0].set(xlabel='Generations', ylabel='%')
    axs[0].legend(loc="upper right")
    if validation_line:
        axs[0].plot(range(len(df_data_generations["Generation"])), np.ones(len(df_data_generations["Generation"]))*validation_line,\
                        c='black', label='Benchmark')
    
    axs[1].scatter(initial_set[f_1_col_name], initial_set[f_2_col_name], s=10, marker="o", label='Initial set')
    axs[1].scatter(df_non_dominated_set[f_1_col_name], df_non_dominated_set[f_2_col_name], s=10, marker="o", label='Non-dom set obtained')
    
    if isinstance(validation_data, pd.DataFrame):        
        for approach_name in validation_data.Approach.unique():
            df_temp = validation_data[validation_data["Approach"]==approach_name]
            axs[1].scatter(df_temp.iloc[:,0], df_temp.iloc[:,1], s=10, marker="o", label=approach_name)
            
    axs[1].set_title('Non-dominated set obtained vs benchmark results')
    axs[1].set(xlabel=f_1_label, ylabel=f_2_label)
    axs[1].legend(loc="upper right")
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()
    plt.savefig(path_results_per_run / "Results_summary_interim.pdf", bbox_inches='tight')

    manager.window.close()
    
def save_results_analysis_fig_interim(initial_set, df_non_dominated_set, validation_data, df_data_generations, name_input_data, path_results_per_run, validation_line=False):
    '''Print Objective functions over time, all solutions and pareto set obtained'''
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(7.5)
    fig.set_figwidth(20)
    
    
    axs[0].plot(df_data_generations["Generation"], df_data_generations["HV"], c='r', marker="o", label='HV obtained')
    axs[0].set_title('HV over all generations')
    axs[0].set(xlabel='Generations', ylabel='%')
    axs[0].legend(loc="upper right")
    if validation_line:
        axs[0].plot(range(len(df_data_generations["Generation"])), np.ones(len(df_data_generations["Generation"]))*validation_line,\
                        c='black', marker=".", label='Benchmark')
    
    axs[1].scatter(initial_set['F_3'], initial_set['F_4'], s=10, marker="o", label='Initial set')
    axs[1].scatter(df_non_dominated_set["F_3"], df_non_dominated_set["F_4"], s=10, marker="o", label='Non-dom set obtained')
    
    if isinstance(validation_data, pd.DataFrame):        
        for approach_name in validation_data.Approach.unique():
            df_temp = validation_data[validation_data["Approach"]==approach_name]
            axs[1].scatter(df_temp.iloc[:,0], df_temp.iloc[:,1], s=10, marker="o", label=approach_name)
            
    axs[1].set_title('Non-dominated set obtained vs benchmark results')
    axs[1].set(xlabel='F_3_AETT', ylabel='F_4_TBR')
    axs[1].legend(loc="upper right")
    
    try:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
        plt.savefig(path_results_per_run / "Results_summary_interim.pdf", bbox_inches='tight')
        manager.window.close()

    except:
        plt.show()
        plt.savefig(path_results_per_run / "Results_summary_interim.pdf", bbox_inches='tight')
        plt.close(fig)
        
def save_results_analysis_fig_interim_save_all(initial_set, df_non_dominated_set, validation_data, df_data_generations, name_input_data, path_results_per_run, add_text="",
                                               labels = ["f_1", "f_2", "f1_ATT", "f2_TRT"], validation_line=False):
    '''Print Objective functions over time, all solutions and pareto set obtained
    If the value of the validation HV line is given, it is printed'''
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(7.5)
    fig.set_figwidth(20)
    
    
    axs[0].plot(df_data_generations["Generation"], df_data_generations["HV"], c='r', marker="o", label='HV obtained')
    axs[0].set_title('HV over all generations')
    axs[0].set(xlabel='Generations', ylabel='%')
    axs[0].legend(loc="upper right")
    if validation_line:
        axs[0].plot(range(len(df_data_generations["Generation"])), np.ones(len(df_data_generations["Generation"]))*validation_line,\
                        c='black', marker=".", label='Benchmark')
    
    axs[1].scatter(initial_set[labels[0]], initial_set[labels[1]], s=10, marker="o", label='Initial set')
    axs[1].scatter(df_non_dominated_set[labels[0]], df_non_dominated_set[labels[1]], s=10, marker="o", label='Non-dom set obtained')
    
    if isinstance(validation_data, pd.DataFrame):        
        for approach_name in validation_data.Approach.unique():
            df_temp = validation_data[validation_data["Approach"]==approach_name]
            axs[1].scatter(df_temp.iloc[:,0], df_temp.iloc[:,1], s=10, marker="o", label=approach_name)
            
    axs[1].set_title('Non-dominated set obtained vs benchmark results')
    axs[1].set(xlabel=labels[2], ylabel=labels[3])
    axs[1].legend(loc="upper right")
    
    if not (path_results_per_run /"Interim").exists():
        os.makedirs(path_results_per_run /"Interim")
    
    if False:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
        plt.savefig(path_results_per_run /"Interim"/f"Results_summary_interim_{str(add_text)}.pdf", bbox_inches='tight')
        manager.window.close()

    else:
        plt.ioff()
        plt.savefig(path_results_per_run /"Interim"/f"Results_summary_interim_{str(add_text)}.pdf", bbox_inches='tight')
        plt.close()



def save_results_analysis_fig(initial_set, df_non_dominated_set, validation_data, df_data_generations, name_input_data, path_results_per_run, labels, validation_line=False):
    '''Print Objective functions over time, all solutions and pareto set obtained'''
    '''labels = ["f_1", "f_2", "f1_AETT", "f2_TBR"] # names labels for the visualisations format
    If the value of the validation HV line is given, it is printed'''
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    
    
    axs[0, 0].plot(df_data_generations["Generation"], df_data_generations["mean_f_1"], c='r', marker="o", label=labels[2])
    axs[0, 0].set_title(f'Mean {labels[2]} over all generations')
    axs[0, 0].set(xlabel='Generations', ylabel=labels[2])
    axs[0, 0].legend(loc="upper right")
    
    axs[1, 0].plot(df_data_generations["Generation"], df_data_generations["mean_f_2"], c='b', marker="o", label=labels[3])
    axs[1, 0].set_title(f'Mean {labels[3]} over all generations')
    axs[1, 0].set(xlabel='Generations', ylabel=labels[3])
    axs[1, 0].legend(loc="upper right") 
    
    axs[0, 1].plot(df_data_generations["Generation"], df_data_generations["HV"], c='r', marker="o", label='HV obtained')
    axs[0, 1].set_title('HV over all generations')
    axs[0, 1].set(xlabel='Generations', ylabel='%')
    axs[0, 1].legend(loc="upper right")
    if validation_line:
        axs[0, 1].plot(range(len(df_data_generations["Generation"])), np.ones(len(df_data_generations["Generation"]))*validation_line,\
                        c='black', marker=".", label='Benchmark')
    
    axs[1, 1].scatter(initial_set[labels[0]], initial_set[labels[1]], s=10, marker="o", label='Initial set')
    axs[1, 1].scatter(df_non_dominated_set[labels[0]], df_non_dominated_set[labels[1]], s=10, c='r', marker="o", label='Non-dom set obtained')
    
    if isinstance(validation_data, pd.DataFrame):        
        for approach_name in validation_data.Approach.unique():
            df_temp = validation_data[validation_data["Approach"]==approach_name]
            axs[1, 1].scatter(df_temp.iloc[:,0], df_temp.iloc[:,1], s=10, marker="o", label=approach_name)
    
    axs[1, 1].set_title('Non-dominated set obtained vs benchmark results')
    axs[1, 1].set(xlabel=labels[2], ylabel=labels[3])
    axs[1, 1].legend(loc="upper right")
    
    plt.ioff()
    plt.savefig(path_results_per_run / "Results_summary.pdf", bbox_inches='tight')
    plt.close()
        
def save_results_analysis_mut_fig(initial_set, df_non_dominated_set, validation_data, df_data_generations, df_mut_ratios, name_input_data, path_results_per_run, labels, validation_line=False, type_mut="line"):
    '''Print Objective functions over time, all solutions and pareto set obtained'''
    '''labels = ["f_1", "f_2", "f1_AETT", "f2_TBR"] # names labels for the visualisations format
    If the value of the validation HV line is given, it is printed'''
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    
    
    axs[0, 0].set_title('Mean objectives over all generations')
    axs[0, 0].plot(df_data_generations["Generation"], df_data_generations["mean_f_1"], c='r', label=labels[2])
    axs[0, 0].set(xlabel='Generations', ylabel=labels[2])
    axs[0, 0].legend(loc="lower left") 
    ax_twin = axs[0, 0].twinx()
    ax_twin.plot(df_data_generations["Generation"], df_data_generations["mean_f_2"], c='b', label=labels[3])
    ax_twin.set(ylabel=labels[3])
    ax_twin.legend(loc=0) 
    
    # Mutation plots
    if type_mut == 'line':
        df_mut_smooth = ga.exp_smooth_df(df_mut_ratios, alpha=0.1, beta=0.1)
        for mut_nr in range(1,len(df_mut_smooth.columns)):
            axs[1, 0].plot(df_mut_smooth["Generation"], df_mut_smooth[df_mut_smooth.columns[mut_nr]], label=df_mut_smooth.columns[mut_nr])
    
    elif type_mut == 'stacked':
        mut_names = df_mut_ratios.columns[1:]
        spread_mutation_values = [df_mut_ratios[y] for y in mut_names]
        axs[1, 0].stackplot(df_mut_ratios["Generation"], spread_mutation_values, labels=mut_names, linewidth=0)
    
    elif type_mut == 'stacked_smooth':
        df_mut_smooth = ga.exp_smooth_df(df_mut_ratios, alpha=0.1, beta=0.1)
        mut_names = df_mut_smooth.columns[1:]
        spread_mutation_values = [df_mut_smooth[y] for y in mut_names]
        axs[1, 0].stackplot(df_mut_smooth["Generation"], spread_mutation_values, labels=mut_names, linewidth=0)
    else:
        print(f"Error: {type_mut} is not a valid option for argument 'type_mut'") 
        
    axs[1, 0].set_title('Mutation ratios over all generations')
    axs[1, 0].set(xlabel='Generations', ylabel='Mutation ratio')
    axs[1, 0].legend(loc='upper left') 
    
    axs[0, 1].plot(df_data_generations["Generation"], df_data_generations["HV"], c='r', label='HV obtained')
    axs[0, 1].set_title('HV over all generations')
    axs[0, 1].set(xlabel='Generations', ylabel='HV [%]')
    axs[0, 1].legend(loc=0)
    if validation_line:
        axs[0, 1].plot(range(len(df_data_generations["Generation"])), np.ones(len(df_data_generations["Generation"]))*validation_line,\
                        c='black', label='Benchmark')
    ax_twin = axs[0, 1].twinx()
    ax_twin.plot(df_data_generations["Generation"], df_data_generations["APD"], c='b', label='APD')
    ax_twin.set(ylabel='Average population diversity [%]')
    ax_twin.legend(loc=0)
            
    axs[1, 1].scatter(initial_set[labels[0]], initial_set[labels[1]], s=10, marker="o", label='Initial set')
    axs[1, 1].scatter(df_non_dominated_set[labels[0]], df_non_dominated_set[labels[1]], s=10, marker="o", label='Non-dom set obtained')

    if isinstance(validation_data, pd.DataFrame):        
        for approach_name in validation_data.Approach.unique():
            df_temp = validation_data[validation_data["Approach"]==approach_name]
            axs[1, 1].scatter(df_temp.iloc[:,0], df_temp.iloc[:,1], s=10, marker="o", label=approach_name)
            
    axs[1, 1].set_title('Non-dominated set obtained vs benchmark results')
    axs[1, 1].set(xlabel=labels[2], ylabel=labels[3])
    axs[1, 1].legend(loc=0)
    try:
        plt.ioff()
        plt.savefig(path_results_per_run / ("Results_summary_"+ type_mut +".pdf"), bbox_inches='tight')
        plt.close()
    except:
        pass
    
def save_results_analysis_mut_fig_UTRP_SA(initial_set, df_non_dominated_set, validation_data, df_data, df_mut_ratios, name_input_data, path_results_per_run, labels, validation_line=False, type_mut = 'line'):
    '''Print Objective functions over time, all solutions and pareto set obtained'''
    '''labels = ["f_1", "f_2", "f1_ATT", "f2_TRT"] # names labels for the visualisations format
    If the value of the validation HV line is given, it is printed'''
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    
    if len(df_data) < 1000:
        n_th = 1
    elif len(df_data) < 10000:
        n_th = 10
    else:
        n_th = 100
    
    axs[0, 0].set_title(f'Objectives for every {n_th}_th iteration')
    axs[0, 0].plot(range(len(df_data))[::n_th], df_data[labels[0]][::n_th], c='r', label=labels[2])
    axs[0, 0].set(xlabel='Iteration', ylabel=labels[2])
    axs[0, 0].legend(loc=0) 
    ax_twin = axs[0, 0].twinx()
    ax_twin.plot(range(len(df_data))[::n_th], df_data[labels[1]][::n_th], c='b', label=labels[3])
    ax_twin.set(ylabel=labels[3])
    ax_twin.legend(loc=0) 

    
    # Mutation plots
    if type_mut == 'line':
        df_mut_smooth = ga.exp_smooth_df(df_mut_ratios, alpha=0.1, beta=0.1)
        for mut_nr in range(1,len(df_mut_smooth.columns)):
            axs[1, 0].plot(range(len(df_mut_smooth))[::n_th], df_mut_smooth[df_mut_smooth.columns[mut_nr]][::n_th], label=df_mut_smooth.columns[mut_nr])
    
    elif type_mut == 'stacked':
        mut_names = df_mut_ratios.columns[1:]
        spread_mutation_values = [df_mut_ratios[y][::n_th] for y in mut_names]
        axs[1, 0].stackplot(range(len(df_mut_ratios))[::n_th], spread_mutation_values, labels=mut_names, linewidth=0)
    
    elif type_mut == 'stacked_smooth':
        df_mut_smooth = ga.exp_smooth_df(df_mut_ratios, alpha=0.1, beta=0.1)
        mut_names = df_mut_smooth.columns[1:]
        spread_mutation_values = [df_mut_smooth[y][::n_th] for y in mut_names]
        axs[1, 0].stackplot(range(len(df_mut_smooth))[::n_th], spread_mutation_values, labels=mut_names, linewidth=0)
    else:
        print(f"Error: {type_mut} is not a valid option for argument 'type_mut'")
    
    axs[1, 0].set_title('Mutation ratios over all iterations')
    axs[1, 0].set(xlabel='Iteration', ylabel='Mutation ratio')
    axs[1, 0].legend(loc='upper left') 
    
    axs[0, 1].plot(range(len(df_data))[::n_th], df_data["HV"][::n_th], c='r', label='HV obtained')
    axs[0, 1].set_title(f'HV for every {n_th}_th iteration')
    axs[0, 1].set(xlabel='Iteration', ylabel='HV [%]')
    axs[0, 1].legend(loc=0)
    if validation_line:
        axs[0, 1].plot(range(len(df_data))[::n_th], np.ones(len(df_data))[::n_th]*validation_line,\
                        c='black', label='Benchmark')
    ax_twin = axs[0, 1].twinx()
    ax_twin.plot(range(len(df_data))[::n_th], df_data["Temperature"][::n_th], c='b', label='Temperature')
    ax_twin.set(ylabel='Temperature for search')
    ax_twin.legend(loc=0)
            
    axs[1, 1].scatter(initial_set[labels[0]], initial_set[labels[1]], s=10, marker="o", label='Initial set')
    axs[1, 1].scatter(df_non_dominated_set[labels[0]], df_non_dominated_set[labels[1]], s=10, marker="o", label='Non-dom set obtained')

    if isinstance(validation_data, pd.DataFrame):        
        for approach_name in validation_data.Approach.unique():
            df_temp = validation_data[validation_data["Approach"]==approach_name]
            axs[1, 1].scatter(df_temp.iloc[:,0], df_temp.iloc[:,1], s=10, marker="o", label=approach_name)
            
    axs[1, 1].set_title('Non-dominated set obtained vs benchmark results')
    axs[1, 1].set(xlabel=labels[2], ylabel=labels[3])
    axs[1, 1].legend(loc=0)
    try:
        plt.ioff()
        plt.savefig(path_results_per_run / ("Results_summary_"+ type_mut +".pdf"), bbox_inches='tight')
        plt.close()
    except:
        pass

def save_final_avgd_results_analysis(initial_set, df_non_dominated_set, validation_data, df_data_generations, df_mut_ratios, name_input_data, path_results, labels, validation_line=False, type_mut = 'line'):
    '''Print Objective functions over time, all solutions and pareto set obtained'''
    '''labels = ["f_1", "f_2", "f1_AETT", "f2_TBR"] # names labels for the visualisations format
    If the value of the validation HV line is given, it is printed'''
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    
    axs[0, 0].set_title('Averaged mean objectives over all generations')
    axs[0, 0].plot(df_data_generations["Generation"], df_data_generations["mean_f_1"], c='r', label=labels[2])
    axs[0, 0].set(xlabel='Generations', ylabel=labels[2])
    axs[0, 0].legend(loc="lower left") 
    ax_twin = axs[0, 0].twinx()
    ax_twin.plot(df_data_generations["Generation"], df_data_generations["mean_f_2"], c='b', label=labels[3])
    ax_twin.set(ylabel=labels[3])
    ax_twin.legend(loc=0) 
    
    
    # Mutation plots
    if type_mut == 'line':
        df_mut_smooth = ga.exp_smooth_df(df_mut_ratios, alpha=0.1, beta=0.1)
        for mut_nr in range(3,len(df_mut_smooth.columns)): # 3 accomodates for the added columns for the read in df
            axs[1, 0].plot(df_mut_smooth["Generation"], df_mut_smooth[df_mut_smooth.columns[mut_nr]], label=df_mut_smooth.columns[mut_nr])
    
    elif type_mut == 'stacked':
        mut_names = df_mut_ratios.columns[3:]
        spread_mutation_values = [df_mut_ratios[y] for y in mut_names]
        axs[1, 0].stackplot(df_mut_ratios["Generation"], spread_mutation_values, labels=mut_names, linewidth=0)
    
    elif type_mut == 'stacked_smooth':
        df_mut_smooth = ga.exp_smooth_df(df_mut_ratios, alpha=0.1, beta=0.1)
        mut_names = df_mut_smooth.columns[3:]
        spread_mutation_values = [df_mut_smooth[y] for y in mut_names]
        axs[1, 0].stackplot(df_mut_smooth["Generation"], spread_mutation_values, labels=mut_names, linewidth=0)
    else:
        print(f"Error: {type_mut} is not a valid option for argument 'type_mut'")
    
    axs[1, 0].set_title('Averaged mutation ratios over all generations')
    axs[1, 0].set(xlabel='Generations', ylabel='Mutation ratio')
    axs[1, 0].legend(loc='upper left') 
    
    axs[0, 1].plot(df_data_generations["Generation"], df_data_generations["HV"], c='r', label='HV obtained')
    axs[0, 1].set_title('Average HV over all generations')
    axs[0, 1].set(xlabel='Generations', ylabel='HV [%]')
    axs[0, 1].legend(loc=0)
    if validation_line:
        axs[0, 1].plot(range(len(df_data_generations["Generation"])), np.ones(len(df_data_generations["Generation"]))*validation_line,\
                        c='black', label='Benchmark')
    ax_twin = axs[0, 1].twinx()
    ax_twin.plot(df_data_generations["Generation"], df_data_generations["APD"], c='b', label='APD')
    ax_twin.set(ylabel='Average population diversity [%]')
    ax_twin.legend(loc=0)
            
    axs[1, 1].scatter(initial_set[labels[0]], initial_set[labels[1]], s=10, marker="o", label='Last initial set')
    axs[1, 1].scatter(df_non_dominated_set[labels[0]], df_non_dominated_set[labels[1]], s=10, marker="o", label='Non-dom set obtained')
    if isinstance(validation_data, pd.DataFrame):        
        for approach_name in validation_data.Approach.unique():
            df_temp = validation_data[validation_data["Approach"]==approach_name]
            axs[1, 1].scatter(df_temp.iloc[:,0], df_temp.iloc[:,1], s=10, marker="o", label=approach_name)
            
    axs[1, 1].set_title('Overall non-dominated set obtained over all runs')
    axs[1, 1].set(xlabel=labels[2], ylabel=labels[3])
    axs[1, 1].legend(loc=0)
    try:
        plt.ioff()
        plt.savefig(path_results / ("Averaged_summary_"+ type_mut +".pdf"), bbox_inches='tight')
        plt.close()
    except:
        pass

def save_final_avgd_results_analysis_SA(initial_set, df_non_dominated_set, validation_data, df_data_generations, df_mut_ratios, name_input_data, path_results, labels, validation_line=False, type_mut = 'line'):
    '''Print Objective functions over time, all solutions and pareto set obtained'''
    '''labels = ["f_1", "f_2", "f1_AETT", "f2_TBR"] # names labels for the visualisations format
    If the value of the validation HV line is given, it is printed'''
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    
    if len(df_mut_ratios) < 1000:
        n_th = 1
    elif len(df_mut_ratios) < 10000:
        n_th = 10
    else:
        n_th = 100
    
    axs[0, 0].set_title('Averaged mean objectives over all iterations')
    axs[0, 0].plot(range(len(df_data_generations))[::n_th], df_data_generations["f_1"][::n_th], c='r', label=labels[2])
    axs[0, 0].set(xlabel='Generations', ylabel=labels[2])
    axs[0, 0].legend(loc=0) 
    ax_twin = axs[0, 0].twinx()
    ax_twin.plot(range(len(df_data_generations))[::n_th], df_data_generations["f_2"][::n_th], c='b', label=labels[3])
    ax_twin.set(ylabel=labels[3])
    ax_twin.legend(loc=0) 
    
    
    # Mutation plots
    if type_mut == 'line':
        df_mut_smooth = ga.exp_smooth_df(df_mut_ratios, alpha=0.1, beta=0.1)
        for mut_nr in range(3,len(df_mut_smooth.columns)): # 3 accomodates for the added columns for the read in df
            axs[1, 0].plot(range(len(df_mut_smooth))[::n_th], df_mut_smooth[df_mut_smooth.columns[mut_nr]][::n_th], label=df_mut_smooth.columns[mut_nr])
    
    elif type_mut == 'stacked':
        mut_names = df_mut_ratios.columns[3:]
        spread_mutation_values = [df_mut_ratios[y][::n_th] for y in mut_names]
        axs[1, 0].stackplot(range(len(df_mut_ratios))[::n_th], spread_mutation_values, labels=mut_names, linewidth=0)
    
    elif type_mut == 'stacked_smooth':
        df_mut_smooth = ga.exp_smooth_df(df_mut_ratios, alpha=0.1, beta=0.1)
        mut_names = df_mut_smooth.columns[3:]
        spread_mutation_values = [df_mut_smooth[y][::n_th] for y in mut_names]
        axs[1, 0].stackplot(range(len(df_mut_ratios))[::n_th], spread_mutation_values, labels=mut_names, linewidth=0)
    else:
        print(f"Error: {type_mut} is not a valid option for argument 'type_mut'")
    
    axs[1, 0].set_title('Averaged mutation ratios over all iterations')
    axs[1, 0].set(xlabel='Total iterations', ylabel='Mutation ratio')
    axs[1, 0].legend(loc='upper left') 
    
    axs[0, 1].plot(range(len(df_data_generations))[::n_th], df_data_generations["HV"][::n_th], c='r', label='HV obtained')
    axs[0, 1].set_title('Average HV over all iterations')
    axs[0, 1].set(xlabel='Total iterations', ylabel='HV [%]')
    axs[0, 1].legend(loc=0)
    if validation_line:
        axs[0, 1].plot(range(len(df_data_generations)), np.ones(len(df_data_generations))*validation_line,\
                        c='black', label='Benchmark')
            
    axs[1, 1].scatter(initial_set[labels[0]], initial_set[labels[1]], s=10, marker="o", label='Last initial set')
    axs[1, 1].scatter(df_non_dominated_set[labels[0]], df_non_dominated_set[labels[1]], s=10, marker="o", label='Non-dom set obtained')
    if isinstance(validation_data, pd.DataFrame):        
        for approach_name in validation_data.Approach.unique():
            df_temp = validation_data[validation_data["Approach"]==approach_name]
            axs[1, 1].scatter(df_temp.iloc[:,0], df_temp.iloc[:,1], s=10, marker="o", label=approach_name)
            
    axs[1, 1].set_title('Overall non-dominated set obtained over all runs')
    axs[1, 1].set(xlabel=labels[2], ylabel=labels[3])
    axs[1, 1].legend(loc=0)
    try:
        plt.ioff()
        plt.savefig(path_results / ("Averaged_summary_"+ type_mut +".pdf"), bbox_inches='tight')
        plt.close()
    except:
        pass    

def save_results_combined_fig(initial_set, df_overall_pareto_set, validation_data, name_input_data, Decisions, path_results, labels):
    '''labels = ["f_1", "f_2", "f1_AETT", "f2_TBR"] # names labels for the visualisations format'''
    fig, axs = plt.subplots(1,1)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    
    
    axs.scatter(initial_set[labels[0]], initial_set[labels[1]], s=10, marker="o", label='Initial set')
    axs.scatter(df_overall_pareto_set[labels[0]], df_overall_pareto_set[labels[1]], s=10, marker="o", label='Pareto front obtained from all runs')
    if Decisions.get('Choice_use_NN_to_predict'):
        axs.scatter(df_overall_pareto_set[labels[0]+"_real"], df_overall_pareto_set[labels[1]+"_real"], s=10, c='orange', marker="o", label='Real Pareto front values')
    if isinstance(validation_data, pd.DataFrame):        
        for approach_name in validation_data.Approach.unique():
            df_temp = validation_data[validation_data["Approach"]==approach_name]
            axs.scatter(df_temp.iloc[:,0], df_temp.iloc[:,1], s=10, marker="o", label=approach_name)
    
    axs.set_title('Pareto front obtained from all runs')
    axs.set(xlabel=labels[2], ylabel=labels[3])
    axs.legend(loc="upper right")
    del axs
    
    if False:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
        plt.savefig(path_results / "Results_combined.pdf", bbox_inches='tight')
        manager.window.close()

    else:
        plt.ioff()
        plt.savefig(path_results / "Results_combined.pdf", bbox_inches='tight')
        plt.close()

def plot_mutation_ratios(df_mut_ratios, alpha=0.1, beta=0.1):
    
    fig = plt.figure()
    #fig.set_figheight(15)
    #fig.set_figwidth(20)
    ax = fig.add_subplot(111)
    
    df_mut_smooth = ga.exp_smooth_df(df_mut_ratios, alpha, beta)
    for mut_nr in range(1,len(df_mut_smooth.columns)):
        ax.plot(df_mut_smooth["Generation"], df_mut_smooth[df_mut_smooth.columns[mut_nr]], label=df_mut_smooth.columns[mut_nr])
    ax.set_title('Mutation ratios over all generations')
    ax.set(xlabel='Generations', ylabel='Mutation ratio')
    ax.legend(loc=0) 
    fig.show()
    
def save_mutation_ratios_plot(df_mut_ratios, path, file_name):
    '''A function to save the mutation summaries'''
    fig = plt.figure()
    #fig.set_figheight(15)
    #fig.set_figwidth(20)
    ax = fig.add_subplot(111)
    
    if "Generation" in df_mut_ratios.columns:
        counter_name = "Generation"
    elif "Total_iterations" in df_mut_ratios.columns:
        counter_name = "Total_iterations"
        if len(df_mut_ratios) < 1000:
            n_th = 1
        elif len(df_mut_ratios) < 10000:
            n_th = 10
        else:
            n_th = 100
    else:
        print("Counter name undefined")
    
    if counter_name == "Generation":
        for mut_nr in range(2,len(df_mut_ratios.columns)): # 2 is to avoid first two columns
            ax.plot(df_mut_ratios[counter_name], df_mut_ratios[df_mut_ratios.columns[mut_nr]], label=df_mut_ratios.columns[mut_nr])
        ax.set_title('Mutation ratios over all generations')
        ax.set(xlabel='Generations', ylabel='Mutation ratio')
        ax.legend(loc=0) 
    else:
        for mut_nr in range(2,len(df_mut_ratios.columns)): # 2 is to avoid first two columns
            ax.plot(df_mut_ratios[counter_name][::n_th], df_mut_ratios[df_mut_ratios.columns[mut_nr]][::n_th], label=df_mut_ratios.columns[mut_nr])
        ax.set_title('Mutation ratios over all generations')
        ax.set(xlabel=counter_name, ylabel='Mutation ratio')
        ax.legend(loc=0) 
    
    try:
        plt.ioff()
        plt.savefig(path / (file_name+".pdf"), bbox_inches='tight')
        plt.close()
    except:
        pass
    
def save_obj_performances_plot(df_data, path, file_name):
    '''A function to save the mutation summaries'''
    for col_nr in range(2,len(df_data.columns)): # 2 is to avoid first two columns 
    
        fig = plt.figure()
        #fig.set_figheight(15)
        #fig.set_figwidth(20)
        ax = fig.add_subplot(111)

        if "Generation" in df_data.columns:
            counter_name = "Generation"
        else:
            counter_name = "Total_iterations"
            if len(df_data) < 1000:
                n_th = 1
            elif len(df_data) < 10000:
                n_th = 10
            else:
                n_th = 100

        
        if counter_name == "Generation":
            ax.plot(df_data[counter_name], df_data[df_data.columns[col_nr]], label=df_data.columns[col_nr])
            ax.set_title(f'{df_data.columns[col_nr]} over all generations')
            ax.set(xlabel=counter_name, ylabel=df_data.columns[col_nr])
            ax.legend(loc=0) 
        else:
            ax.plot(range(len(df_data))[::n_th], df_data[df_data.columns[col_nr]][::n_th], label=df_data.columns[col_nr])
            ax.set_title(f'{df_data.columns[col_nr]} over all iterations')
            ax.set(xlabel=counter_name, ylabel=df_data.columns[col_nr])
            ax.legend(loc=0) 
        
        try:
            plt.ioff()
            plt.savefig(path / (df_data.columns[col_nr]+'_'+file_name+".pdf"), bbox_inches='tight')
            plt.close()
        except:
            pass
    
    
def save_all_mutation_stats_and_plots(path_to_main_folder):
    '''A function that saves all the csv files and plots'''
    df_list, df_names = ga.get_mutation_stats_from_model_runs(path_to_main_folder)
    mut_folder = path_to_main_folder/'Mutations'
    if not os.path.isdir(mut_folder):
        os.mkdir(mut_folder)
    
    for df, name in zip(df_list, df_names):
        df.to_csv(mut_folder / (name+'.csv'))    
        save_mutation_ratios_plot(df, mut_folder, name)

def save_all_obj_stats_and_plots(path_to_main_folder):
    '''A function that saves all the csv files and plots'''
    df_list, df_names = ga.get_obj_stats_from_model_runs(path_to_main_folder)
    save_folder = path_to_main_folder/'Performance'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    
    for df, name in zip(df_list, df_names):
        df.to_csv(save_folder / (name+'.csv'))    
        save_obj_performances_plot(df, save_folder, name)    
    
#plot_mutation_ratios(df_mut_ratios, alpha=0.1, beta=0.1)
