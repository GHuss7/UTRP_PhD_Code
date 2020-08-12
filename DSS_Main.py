# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:02:00 2019

@author: 17832020
"""

# %% Import Libraries
import os
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
import matplotlib.pyplot as plt
import igraph as ig
import networkx as nx





# %% Import personal functions
import DSS_Admin as ga
import DSS_UTNDP_Functions as gf
import DSS_Visualisation as gv
import EvaluateRouteSet as ev

# todo def main_dss(): # create a main function to encapsulate the main body
# def main():
# %% Load the respective files

mx_dist = pd.read_csv("./Input_Data/Mandl_Data/Distance_Matrix.csv") 
mx_dist = gf.format_mx_dist(mx_dist)
mx_dist = mx_dist.values


mx_demand = pd.read_csv("./Input_Data/Mandl_Data/OD_Demand_Matrix.csv") 
mx_demand = gf.format_mx_dist(mx_demand)
mx_demand = mx_demand.values

mx_coords = pd.read_csv("./Input_Data/Mandl_Data/Node_Coords.csv")
mx_coords = gf.format_mx_coords(mx_coords)
# %% Set variables
    
'''Enter the number of allowed routes''' 
parameters_constraints = {
'con_r' : 6,               # (aim for > [numNodes N ]/[maxNodes in route])
'con_minNodes' : 3,                        # minimum nodes in a route
'con_maxNodes' : 10,                       # maximum nodes in a route
'con_N_nodes' : len(mx_dist)              # number of nodes in the network
}

parameters_input = {
'total_demand' : sum(sum(mx_demand))/2, # total demand from demand matrix
'n' : len(mx_dist), # total number of nodes
'wt' : 0, # waiting time [min]
'tp' : 5, # transfer penalty [min]
'Problem_name' : "Mandl", # Specify the name of the problem currently being addresses
'ref_point_max_f1_ATT' : 15.1304, # max f1_ATT for the Hypervolume calculations
'ref_point_min_f1_ATT' : 10.3301, # min f1_ATT for the Hypervolume calculations
'ref_point_max_f2_TRT' : 224, # max f2_TRT for the Hypervolume calculations
'ref_point_min_f2_TRT' : 63 # min f2_TRT for the Hypervolume calculations
}

parameters_SA_routes={
"max_iterations_t" : 10000, # maximum allowable number length of iterations per epoch c
"max_accepts" : 3, # maximum number of accepted moves per epoch
"max_attempts" : 6, # maximum number of attempted moves per epoch
"max_reheating_times" : 4, # the maximum number of times that reheating can take place
"max_poor_epochs" : 3, # maximum number of epochs which may pass without the acceptance of any new solution
"Temp" : 40,  # starting temperature and a geometric cooling schedule is used on it
"Cooling_rate" : 0.998, # the geometric cooling rate
"Reheating_rate" : 1.01, # the geometric reheating rate
"Number_of_initial_solutions" : 1000, # sets the number of initial solutions to generate as starting position
"Feasibility_repair_attempts" : 100 # the max number of edges that will be added and/or removed to try and repair the route feasibility
}

stats_overall = {
'execution_start_time' : datetime.datetime.now()} # enter the begin time

stats = {} # define the stats dictionary

'''Set the reference point for the Hypervolume calculations'''
max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])

# %% Define the adjacent mapping of each node
mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes

# %% Directory functions
result_entries = os.listdir("./Results/Results_"+parameters_input['Problem_name']) # gets the names of all the results entries
results_folder_name = result_entries[len(result_entries)-1] # gets the most recent folder
results_file_path = "./Results/Results_"+parameters_input['Problem_name']+"/"+results_folder_name # sets the path
del result_entries, results_folder_name, results_file_path
  
# %% Generate an initial feasible solution
routes_R = gf.generate_initial_feasible_route_set(mx_dist, parameters_constraints)

if parameters_constraints['con_r'] != len(routes_R): # if the constraint was leveraged, update constraints
    parameters_constraints['con_r'] = len(routes_R)
    print("Number of allowed routes constraint updated to", parameters_constraints['con_r'])

M_iterations = 1000


# %% Simulated Annealing
'''Dominance-based Simulated Annealing'''
# Preliminary inputs
# Let x be the starting solution 
# Let archive be the associated archive
# Let Lc be maximum allowable number length of iterations per epoch c
# Let Amin be minimum number of accepted moves per epoch
# Let Cmax be maximum number of epochs which may pass without the acceptance of any new solution
# Let Temp be the starting temperature and a geometric cooling schedule is used on it
'''Generate initial route sets for input as initial solutions'''
print("Started initial route set generation"+"("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")") 
routes_R_initial_set = list()
for i in range(parameters_SA_routes['Number_of_initial_solutions']):
    routes_R_initial_set.append(gf.generate_initial_feasible_route_set(mx_dist, parameters_constraints))
print("Initial route set generated"+"("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")   

df_routes_R_initial_set =  pd.DataFrame(columns=["f1_ATT","f2_TRT","Routes"])   
for i in range(len(routes_R_initial_set)):
    f_new = ev.evalObjs(routes_R_initial_set[i], mx_dist, mx_demand, parameters_input)
    df_routes_R_initial_set.loc[i] = [f_new[0], f_new[1], gf.convert_routes_list2str(routes_R_initial_set[i])]

print("Started Pareto generation"+"("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")") 
df_routes_R_initial_set = df_routes_R_initial_set[gf.is_pareto_efficient(df_routes_R_initial_set.iloc[:,0:2].values, True)] # reduce the pareto front from the total archive
print("Ended Pareto generation"+"("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")") 

routes_R_initial_set = list()
for i in range(len(df_routes_R_initial_set)):
    routes_R_initial_set.append(gf.convert_routes_str2list(df_routes_R_initial_set.iloc[i,2]))

print("Initial route set generated with size: "+str(len(routes_R_initial_set)))

# %%
'''Simulated Annealing Algorithm for each of the initial route sets'''

for run_nr in range(len(routes_R_initial_set)):
    print("Started run number "+str(run_nr + 1)+" ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
    stats['begin_time'] = datetime.datetime.now() # enter the begin time
    stats['run_number'] = run_nr + 1

    routes_R = routes_R_initial_set[run_nr] # Choose the initial route set to begin with
    '''Initiate algorithm'''
    epoch = 1 # Initialise the epoch counter
    iteration_t = 1 # Initialise the number of iterations 
    poor_epoch = 0 # Initialise the number of epochs without an accepted solution
    attempts = 0 # Initialise the number of attempts made without accepting a solution
    accepts = 0 # Initialise the number of accepts made within an epoch
    reheated = 0 # Initialise the number of times reheated
    SA_Temp = parameters_SA_routes['Temp'] # initialise the starting temperature
    
    df_archive = pd.DataFrame(columns=["f1_ATT","f2_TRT","Routes"]) # create an archive in the correct format
    counter_archive = 1
    df_SA_analysis = pd.DataFrame(columns = ["f1_ATT",\
                                             "f2_TRT",\
                                             "HV",\
                                             "Temperature",\
                                             "C_epoch_number",\
                                             "L_iteration_per_epoch",\
                                             "A_num_accepted_moves_per_epoch",\
                                             "eps_num_epochs_without_accepting_solution",\
                                             "Route"]) # create a df to keep data for SA Analysis
    
    
    f_cur = ev.evalObjs(routes_R, mx_dist, mx_demand, parameters_input)
    df_archive.loc[0] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)]
    df_SA_analysis.loc[0] = [f_cur[0], f_cur[1], gf.norm_and_calc_2d_hv(df_archive.iloc[:,0:2], max_objs, min_objs),\
                       SA_Temp, epoch, 0, 0, 0, gf.convert_routes_list2str(routes_R)]
    
    while poor_epoch <= parameters_SA_routes['max_poor_epochs']:

        while (iteration_t <= parameters_SA_routes['max_iterations_t']) and (accepts < parameters_SA_routes['max_accepts']):
            '''Generate neighbouring solution'''
            routes_R_new = gf.perturb_make_small_change(routes_R, parameters_constraints['con_r'], mapping_adjacent)
            
            while not gf.test_route_feasibility(routes_R_new, parameters_constraints):    # tests whether the new route is feasible
                for i in range(parameters_SA_routes["Feasibility_repair_attempts"]): # this tries to fix the feasibility, but can be time consuming, 
                                        # could also include a "connectivity" characteristic to help repair graph
                    routes_R_new = gf.perturb_make_small_change(routes_R_new, parameters_constraints['con_r'], mapping_adjacent)
                    if gf.test_route_feasibility(routes_R_new, parameters_constraints):
                        break
                routes_R_new = gf.perturb_make_small_change(routes_R, parameters_constraints['con_r'], mapping_adjacent) # if unsuccesful, start over
        
            f_new = ev.evalObjs(routes_R_new, mx_dist, mx_demand, parameters_input)
            df_SA_analysis.loc[len(df_SA_analysis)] = [f_new[0], f_new[1], gf.norm_and_calc_2d_hv(df_archive.iloc[:,0:2], max_objs, min_objs),\
                                                       SA_Temp, epoch, iteration_t, accepts, poor_epoch, gf.convert_routes_list2str(routes_R)]
        
            '''Test solution acceptance and add to archive if accepted and non-dominated'''
            if random.uniform(0,1) < gf.prob_accept_neighbour(df_archive, f_cur, f_new, SA_Temp): # probability to accept neighbour solution as current solution
                routes_R = routes_R_new
                f_cur = f_new
                
                if gf.test_min_min_non_dominated(df_archive, f_cur[0], f_cur[1]): # means solution is undominated
                    df_archive.loc[counter_archive] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)] # adds the new solution
                    counter_archive = counter_archive + 1 # this helps with speed 
                    df_archive = df_archive[gf.is_pareto_efficient(df_archive.iloc[:,0:2].values, True)] # remove dominated solutions from archive
                    accepts = accepts + 1 # updates the number of solutions accepted in the epoch
                    poor_epoch = 0 # resets the number of epochs without acceptance
            
            else:
                if reheated < parameters_SA_routes["max_reheating_times"]: #determines if the max number of reheats have occured
                    attempts = attempts + 1
                    
                    if attempts > parameters_SA_routes['max_attempts']:
                        SA_Temp = parameters_SA_routes['Reheating_rate']*SA_Temp # update temperature based on cooling schedule
                        reheated = reheated + 1
                        break # gets out of the inner while loop
            
            iteration_t = iteration_t + 1
        
        '''Max accepts reached and continue''' # end of inner while loop
        if accepts == 0:
            poor_epoch = poor_epoch + 1 # update number of epochs without an accepted solution
        
        SA_Temp = parameters_SA_routes['Cooling_rate']*SA_Temp # update temperature based on cooling schedule
        epoch = epoch + 1 # Increase Epoch counter
        attempts = 0 # resets the attempts
        accepts = 0 # resets the accepts
        iteration_t = 1 # Initialise the number of iterations 
             
    del f_cur, f_new, accepts, attempts, SA_Temp, epoch, poor_epoch, i, iteration_t, counter_archive

# %%
#'''Simulated Annealing Algorithm for each of the initial route sets'''
#
#for run_nr in range(len(routes_R_initial_set)):
#    print("Started run number "+str(run_nr + 1)+" ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
#    stats['begin_time'] = datetime.datetime.now() # enter the begin time
#    stats['run_number'] = run_nr + 1
#
#    routes_R = routes_R_initial_set[run_nr] # Choose the initial route set to begin with
#    '''Initiate algorithm'''
#    epoch = 1 # Initialise the epoch counter
#    iteration_t = 1 # Initialise the number of iterations 
#    poor_epoch = 0 # Initialise the number of epochs without an accepted solution
#    attempts = 0
#    SA_Temp = parameters_SA_routes['Temp'] # initialise the starting temperature
#    
#    df_archive = pd.DataFrame(columns=["f1_ATT","f2_TRT","Routes"]) # create an archive in the correct format
#    counter_archive = 1
#    df_SA_analysis = pd.DataFrame(columns = ["f1_ATT",\
#                                             "f2_TRT",\
#                                             "HV",\
#                                             "Temperature",\
#                                             "C_epoch_number",\
#                                             "L_iteration_per_epoch",\
#                                             "A_num_accepted_moves_per_epoch",\
#                                             "eps_num_epochs_without_accepting_solution",\
#                                             "Route"]) # create a df to keep data for SA Analysis
#    
#    
#    f_cur = ev.evalObjs(routes_R, mx_dist, mx_demand, parameters_input)
#    df_archive.loc[0] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)]
#    df_SA_analysis.loc[0] = [f_cur[0], f_cur[1], gf.norm_and_calc_2d_hv(df_archive.iloc[:,0:2], max_objs, min_objs),\
#                       SA_Temp, epoch, 0, 0, 0, gf.convert_routes_list2str(routes_R)]
#    
#    while poor_epoch <= parameters_SA_routes['max_poor_epochs']:
#        accepts = 0
#        iteration_t = 1
#        while (iteration_t <= parameters_SA_routes['max_iterations_t']) and (accepts < parameters_SA_routes['max_accepts']):
#            '''Generate neighbouring solution'''
#            routes_R_new = gf.perturb_make_small_change(routes_R, parameters_constraints['con_r'], mapping_adjacent)
#            
#            while not gf.test_route_feasibility(routes_R_new, parameters_constraints):    # tests whether the new route is feasible
#                for i in range(100): # this tries to fix the feasibility, but can be time consuming, 
#                                        # could also include a "connectivity" characteristic to help repair graph
#                    routes_R_new = gf.perturb_make_small_change(routes_R_new, parameters_constraints['con_r'], mapping_adjacent)
#                    if gf.test_route_feasibility(routes_R_new, parameters_constraints):
#                        break
#                routes_R_new = gf.perturb_make_small_change(routes_R, parameters_constraints['con_r'], mapping_adjacent) # if unsuccesful, start over
#        
#            f_new = ev.evalObjs(routes_R_new, mx_dist, mx_demand, parameters_input)
#            df_SA_analysis.loc[len(df_SA_analysis)] = [f_new[0], f_new[1], gf.norm_and_calc_2d_hv(df_archive.iloc[:,0:2], max_objs, min_objs),\
#                                                       SA_Temp, epoch, iteration_t, accepts, poor_epoch, gf.convert_routes_list2str(routes_R)]
#        
#            '''Test solution acceptance and add to archive if accepted and non-dominated'''
#            if random.uniform(0,1) < gf.prob_accept_neighbour(df_archive, f_cur, f_new, SA_Temp): # probability to accept neighbour solution as current solution
#                routes_R = routes_R_new
#                f_cur = f_new
#                
#                if gf.test_min_min_non_dominated(df_archive, f_cur[0], f_cur[1]): # means solution is undominated
#                    df_archive.loc[counter_archive] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)] # adds the new solution
#                    counter_archive = counter_archive + 1 # this helps with speed 
#                    df_archive = df_archive[gf.is_pareto_efficient(df_archive.iloc[:,0:2].values, True)] # remove dominated solutions from archive
#                    accepts = accepts + 1 # updates the number of solutions accepted in the epoch
#                    poor_epoch = 0 # resets the number of epochs without acceptance
#                    
#            iteration_t = iteration_t + 1
#        
#        epoch = epoch + 1 # Increase Epoch counter
#        SA_Temp = parameters_SA_routes['Cooling_rate']*SA_Temp # update temperature based on cooling schedule
#        
#        if accepts == 0:
#            poor_epoch = poor_epoch + 1 # update number of epochs without an accepted solution
#             
#    del f_cur, f_new, accepts, SA_Temp, epoch, poor_epoch, i, iteration_t, counter_archive
#    
 # %% Saving Results per run
    stats['end_time'] = datetime.datetime.now() # save the end time of the run
    
    print("Run number "+str(run_nr+1)+" duration: "+ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']))

    stats['duration'] = stats['end_time'] - stats['begin_time'] # calculate and save the duration of the run
    stats['begin_time'] = stats['begin_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
    stats['end_time'] =  stats['end_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
        
    
    '''Write all results and parameters to files'''
    '''Main folder path'''
    path_main = "./Results/Results_"+parameters_input['Problem_name']+"/"+parameters_input['Problem_name']+"_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(path_main):
        os.mkdir(path_main)
    '''Sub folder path'''
    path = "./Results/Results_"+parameters_input['Problem_name']+"/"+parameters_input['Problem_name']+"_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S")+"/Run_"+str(run_nr+1)
    if not os.path.exists(path):
        os.mkdir(path)
    
    df_SA_analysis.to_csv(path+"/SA_Analysis.csv")
    df_archive.to_csv(path+"/Archive_Routes.csv")
    
    json.dump(parameters_input, open(path+"/parameters_input"+".json", "w")) # saves the parameters in a json file
    json.dump(parameters_constraints, open(path+"/parameters_constraints"+".json", "w"))
    json.dump(parameters_SA_routes, open(path+"/parameters_SA_routes"+".json", "w"))
    pickle.dump(stats, open(path+"/stats"+".pickle", "ab"))
    
    with open(path+"/Run_summary_stats.csv", "w") as archive_file:
        w = csv.writer(archive_file)
        for key, val in {**parameters_input, **parameters_constraints, **parameters_SA_routes, **stats}.items():
            w.writerow([key, val])
        del key, val
    

    #return df_archive, df_SA_analysis, stats
# END MAIN()

# %% Display and save results per run'''
    if True:   
        if False:
            '''Print Archive'''   
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter( df_archive["ATT"], df_archive["TRT"], s=1, c='b', marker="o", label='Archive')
            #ax1.scatter(f_cur[0], f_cur[1], s=1, c='y', marker="o", label='Current')
            #ax1.scatter(f_new[0], f_new[1], s=1, c='r', marker="o", label='New')
            plt.legend(loc='upper left');
            plt.show()
            
        '''Load validation data'''
        Mumford_validation_data = pd.read_csv("./Validation_Data/Mumford_results_on_Mandl_2013/MumfordResultsParetoFront_headers.csv")

        '''Print Objective functions over time, all solutions and pareto set obtained'''
        fig, axs = plt.subplots(2, 2)
        fig.set_figheight(15)
        fig.set_figwidth(20)
        axs[0, 0].scatter(range(len(df_SA_analysis)), df_SA_analysis["f1_ATT"], s=1, c='r', marker="o", label='f1_ATT')
        axs[0, 0].set_title('ATT over all iterations')
        axs[0, 0].set(xlabel='Iterations', ylabel='f1_ATT')
        axs[0, 0].legend(loc="upper right")
        
        axs[1, 0].scatter(range(len(df_SA_analysis)), df_SA_analysis["f2_TRT"], s=1, c='b', marker="o", label='f2_TRT')
        axs[1, 0].set_title('TRT over all iterations')
        axs[1, 0].set(xlabel='Iterations', ylabel='f2_TRT')
        axs[1, 0].legend(loc="upper right") 
        
        axs[0, 1].scatter(range(len(df_SA_analysis)), df_SA_analysis["HV"], s=1, c='r', marker="o", label='HV obtained')
        axs[0, 1].scatter(range(len(df_SA_analysis)), df_SA_analysis["Temperature"]/parameters_SA_routes['Temp'], s=1, c='b', marker="o", label='SA Temperature')
        axs[0, 1].scatter(range(len(df_SA_analysis)), np.ones(len(df_SA_analysis))*gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], max_objs, min_objs),\
           s=1, c='g', marker="o", label='HV Mumford (2013)')
        axs[0, 1].set_title('HV and Temperature over all iterations')
        axs[0, 1].set(xlabel='Iterations', ylabel='%')
        axs[0, 1].legend(loc="upper right")
        
        axs[1, 1].scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=10, c='b', marker="o", label='Initial route sets')
        axs[1, 1].scatter(df_archive["f2_TRT"], df_archive["f1_ATT"], s=10, c='r', marker="o", label='Pareto front obtained')
        axs[1, 1].scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="o", label='Mumford results (2013)')
        axs[1, 1].set_title('Pareto front obtained vs Mumford Results')
        axs[1, 1].set(xlabel='f2_TRT', ylabel='f1_ATT')
        axs[1, 1].legend(loc="upper right")
        
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
        plt.savefig(path+"/Results.png", bbox_inches='tight')
        manager.window.close()
        

# %% Save results after all runs
'''Save the summarised results'''
df_overall_pareto_set = ga.group_pareto_fronts_from_model_runs(path_main, parameters_input).iloc[:,1:]
df_overall_pareto_set = df_overall_pareto_set[gf.is_pareto_efficient(df_overall_pareto_set.iloc[:,0:2].values, True)] # reduce the pareto front from the total archive
df_overall_pareto_set.to_csv(path_main+"/Overall_Pareto_set.csv")   # save the csv file

'''Save the stats for all the runs'''
df_routes_R_initial_set.to_csv(path_main+"/Routes_initial_set.csv")
df_durations = ga.get_stats_from_model_runs(path_main)

stats_overall['execution_end_time'] =  datetime.datetime.now()

stats_overall['total_model_runs'] = run_nr + 1
stats_overall['average_run_time'] = str(df_durations["Duration"].mean())
stats_overall['total_duration'] = stats_overall['execution_end_time']-stats_overall['execution_start_time']
stats_overall['execution_start_time'] = stats_overall['execution_start_time'].strftime("%m/%d/%Y, %H:%M:%S")
stats_overall['execution_end_time'] = stats_overall['execution_end_time'].strftime("%m/%d/%Y, %H:%M:%S")
stats_overall['HV initial set'] = gf.norm_and_calc_2d_hv(df_routes_R_initial_set.iloc[:,0:2], max_objs, min_objs)
stats_overall['HV obtained'] = gf.norm_and_calc_2d_hv(df_overall_pareto_set.iloc[:,0:2], max_objs, min_objs)
stats_overall['HV Benchmark'] = gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], max_objs, min_objs)

df_durations.loc[len(df_durations)] = ["Average", df_durations["Duration"].mean()]
df_durations.to_csv(path_main+"\Run_durations.csv")
del df_durations

with open(path_main+"/Stats_overall.csv", "w") as archive_file:
    w = csv.writer(archive_file)
    for key, val in {**stats_overall, **parameters_input, **parameters_constraints, **parameters_SA_routes}.items():
        w.writerow([key, val])
    del key, val
    
# %%
'''Plot the summarised graph'''
fig, axs = plt.subplots(1,1)
fig.set_figheight(15)
fig.set_figwidth(20)

axs.scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=20, c='b', marker="o", label='Initial route sets')
axs.scatter(df_overall_pareto_set["f2_TRT"], df_overall_pareto_set["f1_ATT"], s=10, c='r', marker="o", label='Pareto front obtained from all runs')
axs.scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="x", label='Mumford results (2013)')
axs.set_title('Pareto front obtained vs Mumford Results')
axs.set(xlabel='f2_TRT', ylabel='f1_ATT')
axs.legend(loc="upper right")
del axs

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
plt.savefig(path_main+"/Results_combined.png", bbox_inches='tight')
manager.window.close()

# %%
'''Plot the analysis graph'''
fig, axs = plt.subplots(1,1)
fig.set_figheight(15)
fig.set_figwidth(20)

axs.scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=20, c='b', marker="o", label='Initial route sets')
axs.scatter(df_overall_pareto_set["f2_TRT"], df_overall_pareto_set["f1_ATT"], s=10, c='r', marker="o", label='Pareto front obtained from all runs')
axs.scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="x", label='Mumford results (2013)')
axs.set_title('Pareto front obtained vs Mumford Results')
axs.set(xlabel='f2_TRT', ylabel='f1_ATT')
axs.legend(loc="upper right")
del axs

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
plt.savefig(path_main+"/Results_combined.png", bbox_inches='tight')
manager.window.close()

# %% Route visualisations
'''Graph visualisation'''
# gv.plotRouteSet2(mx_dist, routes_R, mx_coords)
#gv.plotRouteSet2(mx_dist, routes_R_new, mx_coords)
#gv.plotRouteSet2(mx_dist, gf.convert_routes_str2list(df_routes_R_initial_set.loc[6739,'Routes']), mx_coords)  # gets a route set from initial
#points = list()
#for i in range(len(df_overall_pareto_set)):
    #points = points.append([df_overall_pareto_set.iloc[:,1], df_overall_pareto_set.iloc[:,2]])
