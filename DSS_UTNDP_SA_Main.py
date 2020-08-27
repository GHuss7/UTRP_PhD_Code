# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:02:00 2019

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

# todo def main_dss(): # create a main function to encapsulate the main body
# def main():
# %% Load the respective files
name_input_data = "Mandl_Data"      # set the name of the input data
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)
del name_input_data

# %% Set variables
Decisions = {
"Choice_generate_initial_set" : True, # the alternative loads a set that is prespecified
"Choice_print_results" : True, 
"Choice_conduct_sensitivity_analysis" : False,
"Choice_init_temp_with_trial_runs" : False, # runs M trial runs for the initial temperature
"Choice_normal_run" : True, # choose this for a normal run without Sensitivity Analysis
"Choice_import_saved_set" : False, # import the prespecified set
"Set_name" : "Overall_Pareto_test_set_for_GA.csv" # the name of the set in the main working folder
}

'''Enter the number of allowed routes''' 
parameters_constraints = {
'con_r' : 6,               # (aim for > [numNodes N ]/[maxNodes in route])
'con_minNodes' : 2,                        # minimum nodes in a route
'con_maxNodes' : 10,                       # maximum nodes in a route
'con_N_nodes' : len(mx_dist)              # number of nodes in the network
}

parameters_input = {
'total_demand' : sum(sum(mx_demand))/2, # total demand from demand matrix
'n' : len(mx_dist), # total number of nodes
'wt' : 0, # waiting time [min]
'tp' : 5, # transfer penalty [min]
'Problem_name' : "Mandl_UTRP_DBMOSA", # Specify the name of the problem currently being addresses
'ref_point_max_f1_ATT' : 15, # max f1_ATT for the Hypervolume calculations
'ref_point_min_f1_ATT' : 10, # min f1_ATT for the Hypervolume calculations
'ref_point_max_f2_TRT' : 224, # max f2_TRT for the Hypervolume calculations
'ref_point_min_f2_TRT' : 63 # min f2_TRT for the Hypervolume calculations
}

parameters_SA_routes={
"method" : "SA",
# ALSO: t_max > A_min (max_iterations_t > min_accepts)
"max_iterations_t" : 250, # maximum allowable number length of iterations per epoch; Danie PhD (pg. 98): Dreo et al. chose 100
"max_total_iterations" : 30000, # the total number of accepts that are allowed
"max_epochs" : 1500, # the maximum number of epochs that are allowed
"min_accepts" : 60, # minimum number of accepted moves per epoch; Danie PhD (pg. 98): Dreo et al. chose 12N (N being some d.o.f.)
"max_attempts" : 3, # maximum number of attempted moves per epoch
"max_reheating_times" : 50, # the maximum number of times that reheating can take place
"max_poor_epochs" : 5, # maximum number of epochs which may pass without the acceptance of any new solution
"Temp" : 1,  # starting temperature and a geometric cooling schedule is used on it # M = 1000 gives 93.249866 from 20 runs
"M_iterations_for_temp" : 1000, # the number of initial iterations to establish initial starting temperature
"Cooling_rate" : 0.97, # the geometric cooling rate 0.97 has been doing good, but M =1000 gives 0.996168
"Reheating_rate" : 1.05, # the geometric reheating rate
"number_of_initial_solutions" : 1000, # sets the number of initial solutions to generate as starting position
"Feasibility_repair_attempts" : 1, # the max number of edges that will be added and/or removed to try and repair the route feasibility
"number_of_runs" : 1 # number of runs to complete John 2016 set 20
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

if Decisions["Choice_init_temp_with_trial_runs"]:
    UTNDP_problem_1.problem_SA_parameters.Temp, UTNDP_problem_1.problem_SA_parameters.Cooling_rate =  gf.init_temp_trial_searches(UTNDP_problem_1, number_of_runs=1)
    parameters_SA_routes["Temp"], parameters_SA_routes["Cooling_rate"] = UTNDP_problem_1.problem_SA_parameters.Temp, UTNDP_problem_1.problem_SA_parameters.Cooling_rate

def main(UTNDP_problem_1):
    
    """ Keep track of the stats """
    stats_overall = {
        'execution_start_time' : datetime.datetime.now() # enter the begin time
        } 
    
    stats = {} # define the stats dictionary
    
    #%% Define the Objective UTNDP functions
    def fn_obj(routes, UTNDP_problem_input):
        return (ev.evalObjs(routes, 
                UTNDP_problem_input.problem_data.mx_dist, 
                UTNDP_problem_input.problem_data.mx_demand, 
                UTNDP_problem_input.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)
    
    def fn_obj_row(routes):
        return (ev.evalObjs(routes, 
                UTNDP_problem_1.problem_data.mx_dist, 
                UTNDP_problem_1.problem_data.mx_demand, 
                UTNDP_problem_1.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)
    
    # %% Generate an initial feasible solution
    routes_R = gf.generate_initial_feasible_route_set(mx_dist, UTNDP_problem_1.problem_constraints.__dict__)
    
    if UTNDP_problem_1.problem_constraints.con_r != len(routes_R): # if the constraint was leveraged, update constraints
        UTNDP_problem_1.problem_constraints.con_r = len(routes_R)
        print("Number of allowed routes constraint updated to", UTNDP_problem_1.problem_constraints.con_r)
    
    # %% Simulated Annealing: Initial solutions
    '''Initial solutions'''
    
    if Decisions["Choice_generate_initial_set"]:
        '''Generate initial route sets for input as initial solutions'''
        routes_R_initial_set, df_routes_R_initial_set = gf.generate_initial_route_sets(UTNDP_problem_1)
    
    else: # use this alternative if you want to use another set as input
        """Standard route to begin with"""
        routes_R_initial_set = list()
        routes_R_initial_set.append(gf.convert_routes_str2list("5-7-9-12*9-7-5-3-4*0-1-2-5-14-6*13-9-6-14-8*1-2-5-14*9-10-11-3*"))
      
        
        df_routes_R_initial_set =  pd.DataFrame(columns=["f1_ATT","f2_TRT","Routes"])   
        for i in range(len(routes_R_initial_set)):
            f_new = ev.evalObjs(routes_R_initial_set[i], UTNDP_problem_1.problem_data.mx_dist, UTNDP_problem_1.problem_data.mx_demand, UTNDP_problem_1.problem_inputs.__dict__)
            df_routes_R_initial_set.loc[i] = [f_new[0], f_new[1], gf.convert_routes_list2str(routes_R_initial_set[i])]
        
    if Decisions["Choice_import_saved_set"]: # Make true to import a set that is saved
        df_routes_R_initial_set = pd.read_csv(Decisions["Set_name"]) 
        df_routes_R_initial_set = df_routes_R_initial_set.drop(df_routes_R_initial_set.columns[0], axis=1)
    
        routes_R_initial_set = list()
        for i in range(len(df_routes_R_initial_set)):
            routes_R_initial_set.append(gf.convert_routes_str2list(df_routes_R_initial_set.iloc[i,2]))
        
        print("Initial route set imported with size: "+str(len(routes_R_initial_set)))
        
    # %% Simulated Annealing Algorithm for each of the initial route sets
    '''Simulated Annealing Algorithm for each of the initial route sets'''
    run_nr_counter = range(UTNDP_problem_1.problem_SA_parameters.number_of_runs) # default values
    
    if Decisions["Choice_normal_run"]:
        run_nr_counter = range(len(routes_R_initial_set)) # sets the tests ito for loops to run
        UTNDP_problem_1.add_text = f"Normal_run_{UTNDP_problem_1.problem_SA_parameters.number_of_runs}_routes_{len(df_routes_R_initial_set)}"
    if Decisions["Choice_conduct_sensitivity_analysis"]:
        run_nr_counter = range(UTNDP_problem_1.problem_SA_parameters.number_of_runs) # sets the tests ito for loops to run
        
    for run_nr in run_nr_counter:
        route_set_nr_counter = [run_nr]
        if Decisions["Choice_conduct_sensitivity_analysis"]: route_set_nr_counter = [0] # standardizes the sensitivity analysis to only the first route set
        
        for route_set_nr in route_set_nr_counter:
            print("Started route set number "+str(route_set_nr + 1)+" ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
            stats['begin_time'] = datetime.datetime.now() # enter the begin time
            stats['run_number'] = f"{run_nr + 1}.{route_set_nr}"
        
            routes_R = routes_R_initial_set[route_set_nr] # Choose the initial route set to begin with
            '''Initiate algorithm'''
            epoch = 1 # Initialise the epoch counter
            total_iterations = 0 
            poor_epoch = 0 # Initialise the number of epochs without an accepted solution
            attempts = 0 # Initialise the number of attempts made without accepting a solution
            accepts = 0 # Initialise the number of accepts made within an epoch
            reheated = 0 # Initialise the number of times reheated
            SA_Temp = UTNDP_problem_1.problem_SA_parameters.Temp # initialise the starting temperature
            
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
                                                     "Route",\
                                                     "Attempts"]) # create a df to keep data for SA Analysis
            
            
            f_cur = fn_obj(routes_R, UTNDP_problem_1)
            df_archive.loc[0] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)]
            HV = gf.norm_and_calc_2d_hv(df_archive.iloc[:,0:2], max_objs, min_objs)
            df_SA_analysis.loc[0] = [f_cur[0], f_cur[1], HV,\
                               SA_Temp, epoch, 0, 0, 0, gf.convert_routes_list2str(routes_R), 0]
            
            print(f'Epoch:{epoch-1} \tHV:{round(HV, 4)}')
             
            
            
            while poor_epoch <= UTNDP_problem_1.problem_SA_parameters.max_poor_epochs and total_iterations <= UTNDP_problem_1.problem_SA_parameters.max_total_iterations and epoch <= UTNDP_problem_1.problem_SA_parameters.max_epochs:
                iteration_t = 1 # Initialise the number of iterations 
                accepts = 0 # Initialise the accepts
                while (iteration_t <= UTNDP_problem_1.problem_SA_parameters.max_iterations_t) and (accepts < UTNDP_problem_1.problem_SA_parameters.min_accepts):
                    '''Generate neighbouring solution'''
                    routes_R_new = gf.perturb_make_small_change(routes_R, UTNDP_problem_1.problem_constraints.con_r, UTNDP_problem_1.mapping_adjacent)
                    
                    while not gf.test_route_feasibility(routes_R_new, UTNDP_problem_1.problem_constraints.__dict__):    # tests whether the new route is feasible
                        for i in range(UTNDP_problem_1.problem_SA_parameters.Feasibility_repair_attempts): # this tries to fix the feasibility, but can be time consuming, 
                                                # could also include a "connectivity" characteristic to help repair graph
                            routes_R_new = gf.perturb_make_small_change(routes_R_new, UTNDP_problem_1.problem_constraints.con_r, UTNDP_problem_1.mapping_adjacent)
                            if gf.test_route_feasibility(routes_R_new, UTNDP_problem_1.problem_constraints.__dict__):
                                break
                        routes_R_new = gf.perturb_make_small_change(routes_R, UTNDP_problem_1.problem_constraints.con_r, UTNDP_problem_1.mapping_adjacent) # if unsuccesful, start over
                
                    f_new = fn_obj(routes_R_new, UTNDP_problem_1)
                    HV = gf.norm_and_calc_2d_hv(df_archive.iloc[:,0:2], max_objs, min_objs)
                    df_SA_analysis.loc[len(df_SA_analysis)] = [f_new[0], f_new[1], HV,\
                                                               SA_Temp, epoch, iteration_t, accepts, poor_epoch, gf.convert_routes_list2str(routes_R), attempts]
                    
                    total_iterations = total_iterations + 1 # increments the total iterations for stopping criteria
                
                    '''Test solution acceptance and add to archive if accepted and non-dominated'''
                    if random.uniform(0,1) < gf.prob_accept_neighbour(df_archive, f_cur, f_new, SA_Temp): # probability to accept neighbour solution as current solution
                        routes_R = routes_R_new
                        f_cur = f_new
                        accepts = accepts + 1 
                        
                        if gf.test_min_min_non_dominated(df_archive, f_cur[0], f_cur[1]): # means solution is undominated
                            df_archive.loc[counter_archive] = [f_cur[0], f_cur[1], gf.convert_routes_list2str(routes_R)] # adds the new solution
                            counter_archive = counter_archive + 1 # this helps with speed 
                            df_archive = df_archive[gf.is_pareto_efficient(df_archive.iloc[:,0:2].values, True)] # remove dominated solutions from archive
                            accepts = accepts - 1 # updates the number of solutions accepted in the epoch
                            poor_epoch_flag = False # lowers flag when a solution is added to the archive
                            poor_epoch = 0 # resets the number of epochs without acceptance
                    
                    else:
                        if reheated < UTNDP_problem_1.problem_SA_parameters.max_reheating_times: #determines if the max number of reheats have occured
                            attempts = attempts + 1
                            
                            if attempts > UTNDP_problem_1.problem_SA_parameters.max_attempts:
                                SA_Temp = UTNDP_problem_1.problem_SA_parameters.Reheating_rate*SA_Temp # update temperature based on cooling schedule
                                reheated = reheated + 1
                                print(f"Reheated:{reheated}/{UTNDP_problem_1.problem_SA_parameters.max_reheating_times}")
                                break # gets out of the inner while loop
                    
                    iteration_t = iteration_t + 1
                
                '''Max accepts reached and continue''' # end of inner while loop
                if poor_epoch_flag:
                    poor_epoch = poor_epoch + 1 # update number of epochs without an accepted solution
                
                print(f'Epoch:{epoch} \tTemp:{round(SA_Temp,4)} \tHV:{round(HV, 4)} \tAccepts:{accepts} \tAttempts:{attempts} \tPoor_epoch:{poor_epoch}/{UTNDP_problem_1.problem_SA_parameters.max_poor_epochs} \tTotal_i:{total_iterations}[{iteration_t}] ')
    
                '''Úpdate parameters'''
                SA_Temp = UTNDP_problem_1.problem_SA_parameters.Cooling_rate*SA_Temp # update temperature based on cooling schedule
                epoch = epoch + 1 # Increase Epoch counter
                attempts = 0 # resets the attempts
                poor_epoch_flag = True # sets the poor epoch flag, and lowered when solution added to the archive     
                
            del f_cur, f_new, accepts, attempts, SA_Temp, epoch, poor_epoch, i, iteration_t, counter_archive
        
            
         # %% Saving Results per run
            if Decisions["Choice_print_results"]:
                stats['end_time'] = datetime.datetime.now() # save the end time of the run
                
                print("Run number "+str(run_nr+1)+" duration: "+ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']))
            
                stats['duration'] = stats['end_time'] - stats['begin_time'] # calculate and save the duration of the run
                stats['begin_time'] = stats['begin_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
                stats['end_time'] =  stats['end_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
                stats['HV obtained'] = HV
                
                
                '''Write all results and parameters to files'''
                '''Main folder path'''
                path_parent_folder = Path(os.path.dirname(os.getcwd()))
                path_results = path_parent_folder / ("Results/Results_"+UTNDP_problem_1.problem_inputs.Problem_name+"/"+UTNDP_problem_1.problem_inputs.Problem_name+"_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S")+" "+UTNDP_problem_1.problem_SA_parameters.method)
                
                path_results = path_parent_folder / ("Results/Results_"+
                                                     parameters_input['Problem_name']+
                                                     "/"+parameters_input['Problem_name']+
                                                     "_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S")+
                                                     " "+parameters_SA_routes['method']+
                                                     f"_{UTNDP_problem_1.add_text}")
                
                '''Sub folder path'''
                path_results_per_run = path_results / (f"Run_{run_nr + 1}")
                if not path_results_per_run.exists():
                    os.makedirs(path_results_per_run)
                
                df_SA_analysis.to_csv(path_results_per_run / "SA_Analysis.csv")
                df_archive.to_csv(path_results_per_run / "Archive_Routes.csv")
                
                json.dump(UTNDP_problem_1.problem_inputs.__dict__, open(path_results_per_run / "parameters_input.json", "w")) # saves the parameters in a json file
                json.dump(UTNDP_problem_1.problem_constraints.__dict__, open(path_results_per_run / "parameters_constraints.json", "w"))
                json.dump(UTNDP_problem_1.problem_SA_parameters.__dict__, open(path_results_per_run / "parameters_SA_routes.json", "w"))
                pickle.dump(stats, open(path_results_per_run / "stats.pickle", "ab"))
                
                with open(path_results_per_run / "Run_summary_stats.csv", "w") as archive_file:
                    w = csv.writer(archive_file)
                    for key, val in {**UTNDP_problem_1.problem_inputs.__dict__, **UTNDP_problem_1.problem_constraints.__dict__, **UTNDP_problem_1.problem_SA_parameters.__dict__, **stats}.items():
                        w.writerow([key, val])
                    del key, val
            
            # %% Display and save results per run'''
                #plt.rcParams['font.family'] = 'serif'
                #plt.rcParams['font.serif'] = 'CMU Serif, Times New Roman'
                #plt.rcParams['font.size'] = 15 # Makes the text Sans Serif CMU
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
                    John_validation_data = pd.read_csv("./Validation_Data/John_results_on_Mandl_2016/Results_data_headers.csv")
            
            
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
                    axs[0, 1].scatter(range(len(df_SA_analysis)), df_SA_analysis["Temperature"]/UTNDP_problem_1.problem_SA_parameters.Temp, s=1, c='b', marker="o", label='SA Temperature')
                    axs[0, 1].scatter(range(len(df_SA_analysis)), np.ones(len(df_SA_analysis))*gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], max_objs, min_objs),\
                       s=1, c='g', marker="o", label='HV Mumford (2013)')
                    axs[0, 1].scatter(range(len(df_SA_analysis)), np.ones(len(df_SA_analysis))*gf.norm_and_calc_2d_hv(John_validation_data.iloc[:,0:2], max_objs, min_objs),\
                       s=1, c='black', marker="o", label='HV John (2016)')
                    axs[0, 1].set_title('HV and Temperature over all iterations')
                    axs[0, 1].set(xlabel='Iterations', ylabel='%')
                    axs[0, 1].legend(loc="upper right")
                    
                    axs[1, 1].scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=10, c='orange', marker="o", label='Initial route sets')
                    axs[1, 1].scatter(df_archive["f2_TRT"], df_archive["f1_ATT"], s=10, c='r', marker="o", label='Pareto front obtained')
                    axs[1, 1].scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="o", label='Mumford results (2013)')
                    axs[1, 1].scatter(John_validation_data.iloc[:,1], John_validation_data.iloc[:,0], s=10, c='b', marker="o", label='John results (2016)')
                    axs[1, 1].set_title('Pareto front obtained vs Mumford Results')
                    axs[1, 1].set(xlabel='f2_TRT', ylabel='f1_ATT')
                    axs[1, 1].legend(loc="upper right")
                    
                    manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()
                    plt.show()
                    plt.savefig(path_results_per_run / "Results_objectives.pdf", bbox_inches='tight')
            
                    manager.window.close()
                    
                    
                    '''Print parameters over time, all solutions and pareto set obtained'''
                    fig, axs = plt.subplots(2, 2)
                    fig.set_figheight(15)
                    fig.set_figwidth(20)
                    axs[0, 0].scatter(range(len(df_SA_analysis)), df_SA_analysis["L_iteration_per_epoch"], s=1, c='r', marker="o", label='L_iteration_per_epoch')
                    axs[0, 0].set_title('Iteration per epoch over all iterations')
                    axs[0, 0].set(xlabel='Iterations', ylabel='L_iteration_per_epoch')
                    axs[0, 0].legend(loc="upper right")
                    
                    axs[1, 0].scatter(range(len(df_SA_analysis)), df_SA_analysis["A_num_accepted_moves_per_epoch"], s=1, c='b', marker="o", label='A_num_accepted_moves_per_epoch')
                    axs[1, 0].set_title('Number of accepted moves per epoch over all iterations')
                    axs[1, 0].set(xlabel='Iterations', ylabel='A_num_accepted_moves_per_epoch')
                    axs[1, 0].legend(loc="upper right") 
                    
                    axs[0, 1].scatter(range(len(df_SA_analysis)), df_SA_analysis["HV"], s=1, c='r', marker="o", label='HV obtained')
                    axs[0, 1].scatter(range(len(df_SA_analysis)), df_SA_analysis["Temperature"]/UTNDP_problem_1.problem_SA_parameters.Temp, s=1, c='b', marker="o", label='SA Temperature')
                    axs[0, 1].scatter(range(len(df_SA_analysis)), np.ones(len(df_SA_analysis))*gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], max_objs, min_objs),\
                       s=1, c='g', marker="o", label='HV Mumford (2013)')
                    axs[0, 1].scatter(range(len(df_SA_analysis)), np.ones(len(df_SA_analysis))*gf.norm_and_calc_2d_hv(John_validation_data.iloc[:,0:2], max_objs, min_objs),\
                       s=1, c='black', marker="o", label='HV John (2016)')
                    axs[0, 1].set_title('HV and Temperature over all iterations')
                    axs[0, 1].set(xlabel='Iterations', ylabel='%')
                    axs[0, 1].legend(loc="upper right")
                    
                    axs[1, 1].scatter(range(len(df_SA_analysis)), df_SA_analysis["eps_num_epochs_without_accepting_solution"], s=1, c='b', marker="o", label='Num_epochs_without_accepting_solution')
                    axs[1, 1].set_title('Number of epochs without accepting moves over all iterations')
                    axs[1, 1].set(xlabel='Iterations', ylabel='Num_epochs_without_accepting_solution')
                    axs[1, 1].legend(loc="upper left") 
                    
                    manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()
                    plt.show()
                    plt.savefig(path_results_per_run / "Results_parameters.pdf", bbox_inches='tight')
            
                    manager.window.close()
        

    # %% Save results after all runs
    if Decisions["Choice_print_results"]:
        '''Save the summarised results'''
        df_overall_pareto_set = ga.group_pareto_fronts_from_model_runs(path_results, UTNDP_problem_1.problem_inputs.__dict__).iloc[:,1:]
        df_overall_pareto_set = df_overall_pareto_set[gf.is_pareto_efficient(df_overall_pareto_set.iloc[:,0:2].values, True)] # reduce the pareto front from the total archive
        df_overall_pareto_set = df_overall_pareto_set.sort_values(by='f1_ATT', ascending=True) # sort
        df_overall_pareto_set.to_csv(path_results / "Overall_Pareto_set.csv")   # save the csv file
        
        '''Save the stats for all the runs'''
        df_routes_R_initial_set.to_csv(path_results / "Routes_initial_set.csv")
        df_durations = ga.get_stats_from_model_runs(path_results)
        
        stats_overall['execution_end_time'] =  datetime.datetime.now()
        
        stats_overall['total_model_runs'] = run_nr + 1
        stats_overall['average_run_time'] = str(df_durations["Duration"].mean())
        stats_overall['total_duration'] = stats_overall['execution_end_time']-stats_overall['execution_start_time']
        stats_overall['execution_start_time'] = stats_overall['execution_start_time'].strftime("%m/%d/%Y, %H:%M:%S")
        stats_overall['execution_end_time'] = stats_overall['execution_end_time'].strftime("%m/%d/%Y, %H:%M:%S")
        stats_overall['HV initial set'] = gf.norm_and_calc_2d_hv(df_routes_R_initial_set.iloc[:,0:2], max_objs, min_objs)
        stats_overall['HV obtained'] = gf.norm_and_calc_2d_hv(df_overall_pareto_set.iloc[:,0:2], max_objs, min_objs)
        stats_overall['HV Benchmark Mumford 2013'] = gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        stats_overall['HV Benchmark John 2016'] = gf.norm_and_calc_2d_hv(John_validation_data.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
            
        df_durations.loc[len(df_durations)] = ["Average", df_durations["Duration"].mean()]
        df_durations.to_csv(path_results / "Run_durations.csv")
        del df_durations
        
        with open(path_results / "Stats_overall.csv", "w") as archive_file:
            w = csv.writer(archive_file)
            for key, val in {**stats_overall, **UTNDP_problem_1.problem_inputs.__dict__, **UTNDP_problem_1.problem_constraints.__dict__, **UTNDP_problem_1.problem_SA_parameters.__dict__}.items():
                w.writerow([key, val])
            del key, val
            
        ga.get_sens_tests_stats_from_UTRP_SA_runs(path_results)
        ga.capture_all_runs_HV_over_iterations_from_UTRP_SA(path_results)
        
        # %% Plot summary graph
        '''Plot the summarised graph'''
        fig, axs = plt.subplots(1,1)
        fig.set_figheight(15)
        fig.set_figwidth(20)
        
        axs.scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=20, c='orange', marker="o", label='Initial route sets')
        axs.scatter(df_overall_pareto_set["f2_TRT"], df_overall_pareto_set["f1_ATT"], s=10, c='r', marker="o", label='Pareto front obtained from all runs')
        axs.scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="x", label='Mumford results (2013)')
        axs.scatter(John_validation_data.iloc[:,1], John_validation_data.iloc[:,0], s=10, c='b', marker="o", label='John results (2016)')
        axs.set_title('Pareto front obtained vs Mumford Results')
        axs.set(xlabel='f2_TRT', ylabel='f1_ATT')
        axs.legend(loc="upper right")
        del axs
        
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
        plt.savefig(path_results / "Results_combined.pdf", bbox_inches='tight')
        manager.window.close()

    return df_archive

# %% Sensitivity analysis
''' Sensitivity analysis tests'''

# Single Thread
if __name__ == "__main__":
    
    if Decisions["Choice_conduct_sensitivity_analysis"]:
        start = time.perf_counter()
        
        # Set up the list of parameters to test
        sensitivity_list = [[parameters_SA_routes, "max_iterations_t", 10, 50, 100, 250, 500, 1000, 1500], 
                            [parameters_SA_routes, "min_accepts",  1, 3, 5, 10, 25, 50, 100, 200, 400], # takes longer at first... bottleneck
                            [parameters_SA_routes, "max_attempts", 1, 3, 5, 10, 25, 50, 100, 200, 400],
                            [parameters_SA_routes, "max_reheating_times", 1, 3, 5, 10, 25],
                            [parameters_SA_routes, "max_poor_epochs", 1, 3, 5, 10, 25, 50, 100, 200, 400],
                            [parameters_SA_routes, "Temp", 1, 5, 10, 25, 50, 100, 150, 200],
                            [parameters_SA_routes, "Cooling_rate", 0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 0.9961682402927605],
                            [parameters_SA_routes, "Reheating_rate", 1.5, 1.3, 1.1, 1.05, 1.02],
                            [parameters_SA_routes, "Feasibility_repair_attempts", 1, 2, 3, 4, 5, 6],
                            ]
        
        sensitivity_list = [#[parameters_SA_routes, "max_iterations_t", 10, 50, 100, 250, 500, 1000, 1500], 
                            #[parameters_SA_routes, "min_accepts",  1, 3, 5, 10, 25, 50, 100, 200, 400], # takes longer at first... bottleneck
                            #[parameters_SA_routes, "max_attempts", 1, 3, 5, 10, 25, 50, 100, 200, 400],
                            #[parameters_SA_routes, "max_reheating_times", 1, 3, 5, 10, 25],
                            #[parameters_SA_routes, "max_poor_epochs", 1, 3, 5, 10, 25, 50, 100, 200, 400],
                            [parameters_SA_routes, "Temp", 1, 5, 10, 25, 50, 100, 150, 200],
                            #[parameters_SA_routes, "Cooling_rate", 0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 0.9961682402927605],
                            #[parameters_SA_routes, "Reheating_rate", 1.5, 1.3, 1.1, 1.05, 1.02],
                            #[parameters_SA_routes, "Feasibility_repair_attempts", 1, 2, 3, 4, 5, 6],
                            ]

        
        
        for sensitivity_test in sensitivity_list:
            parameter_dict = sensitivity_test[0]
            dict_entry = sensitivity_test[1]
            for test_counter in range(2,len(sensitivity_test)):
                
                print("Test: {0} = {1}".format(sensitivity_test[1], sensitivity_test[test_counter]))
                
                UTNDP_problem_1.add_text = f"{sensitivity_test[1]}_{round(sensitivity_test[test_counter],2)}"
                
                temp_storage = parameter_dict[dict_entry]
                
                # Set new parameters
                parameter_dict[dict_entry] = sensitivity_test[test_counter]
    
                # Update problem instance
                UTNDP_problem_1.problem_constraints = gc.Problem_inputs(parameters_constraints)
                UTNDP_problem_1.problem_SA_parameters = gc.Problem_metaheuristic_inputs(parameters_SA_routes)
                
                # Run model
                df_archive = main(UTNDP_problem_1)
                
                # Reset the original parameters
                parameter_dict[dict_entry] = temp_storage
        
        finish = time.perf_counter()
        
        print(f'Finished in {round(finish-start, 6)} second(s)')
    
    
    if Decisions["Choice_normal_run"]:
        df_archive = main(UTNDP_problem_1)