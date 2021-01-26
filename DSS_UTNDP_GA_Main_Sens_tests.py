# -*- coding: utf-8 -*-
"""
Created on Tue Dec 03 10:54:00 2019

@author: 17832020
"""

# %% Import Libraries
import os
import re
import string
from pathlib import Path
import csv
import json
import pickle
import pandas as pd 
import numpy as np
import math
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
import concurrent.futures

# %% Import personal functions
import DSS_Admin as ga
import DSS_UTNDP_Functions as gf
import DSS_UTNDP_Classes as gc
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import EvaluateRouteSet as ev

    
#%% Pymoo functions
from pymoo.util.function_loader import load_function
from pymoo.util.dominator import Dominator
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.population import Population

from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import Dominator
from pymoo.util.misc import find_duplicates, has_feasible
#from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


    
# %% Load the respective files
name_input_data = "Mumford0"      # set the name of the input data
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)

# %% Set input parameters
Choice_generate_initial_set = True 
Choice_print_results = True 
Choice_conduct_sensitivity_analysis = True 
  
'''State the various parameter constraints''' 
parameters_constraints = {
'con_r' : 12,               # number of allowed routes (aim for > [numNodes N ]/[maxNodes in route])
'con_minNodes' : 2,                        # minimum nodes in a route
'con_maxNodes' : 15,                       # maximum nodes in a route
'con_N_nodes' : len(mx_dist),              # number of nodes in the network
'con_fleet_size' : 40,                     # number of vehicles that are allowed
'con_vehicle_capacity' : 20,               # the max carrying capacity of a vehicle
'con_lower_bound' : 0,                 # the lower bound for the problem
'con_upper_bound' : 1                 # the upper bound for the problem
}

'''State the various input parameter''' 
parameters_input = {
'total_demand' : sum(sum(mx_demand))/2, # total demand from demand matrix
'n' : len(mx_dist), # total number of nodes
'wt' : 0, # waiting time [min]
'tp' : 5, # transfer penalty [min]
'Problem_name' : f"{name_input_data}_UTRP_NSGAII", # Specify the name of the problem currently being addresses
'ref_point_max_f1_ATT' : 32, # max f1_ATT for the Hypervolume calculations
'ref_point_min_f1_ATT' : 13, # min f1_ATT for the Hypervolume calculations
'ref_point_max_f2_TRT' : 700, # max f2_TRT for the Hypervolume calculations
'ref_point_min_f2_TRT' : 94, # min f2_TRT for the Hypervolume calculations
'walkFactor' : 3, # factor it takes longer to walk than to drive
'boardingTime' : 0.1, # assume boarding and alighting time = 6 seconds
'alightingTime' : 0.1, # problem when alighting time = 0 (good test 0.5)(0.1 also works)
'large_dist' : int(mx_dist.max()), # the large number from the distance matrix
'alpha_const_inter' : 0.5 # constant for interarrival times relationship 0.5 (Spiess 1989)
}

'''State the various GA input parameters for frequency setting''' 
parameters_GA_route_design={
"method" : "GA",
"population_size" : 200, #should be an even number STANDARD: 200 (John 2016)
"generations" : 200, # STANDARD: 200 (John 2016)
"number_of_runs" : 20, # STANDARD: 20 (John 2016)
"crossover_probability" : 0.6, 
"crossover_distribution_index" : 5,
"mutation_probability" : 1, # John: 1/|Route set| -> set later
"mutation_distribution_index" : 10,
"tournament_size" : 2,
"termination_criterion" : "StoppingByEvaluations",
"max_evaluations" : 25000,
"number_of_variables" : parameters_constraints["con_r"],
"number_of_objectives" : 2, # this could still be automated in the future
"Number_of_initial_solutions" : 10000 # number of initial solutions to be generated and chosen from
}


'''Set the reference point for the Hypervolume calculations'''
max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])

            
#%% Input parameter tests

'''Test the inputs for feasibility'''
# Test feasibility if there are enough buses to cover each route once
if parameters_constraints["con_r"] > parameters_constraints["con_fleet_size"]:
    print("Warning: Number of available vehicles are less than the number of routes.\n"\
          "Number of routes allowed set to "+ str(parameters_constraints["con_r"]))
    parameters_constraints["con_r"] = parameters_constraints["con_fleet_size"]

# %% Import the route set to be evaluated
''' Import the route set '''
if False:
    R_x = gf.convert_routes_str2list("13-14-10-8-15*4-2-3-6*13-11-12*7-15-9*5-4-6-15*3-2-1*")
    for i in range(len(R_x)): # get routes in the correct format
        R_x[i] = [x - 1 for x in R_x[i]] # subtract 1 from each element in the list
    del i
    R_routes = gf2.Routes(R_x)


#%% Define the UTNDP Problem      
UTNDP_problem_1 = gc.UTNDP_problem()
UTNDP_problem_1.problem_data = gc.Problem_data(mx_dist, mx_demand, mx_coords)
UTNDP_problem_1.problem_constraints = gc.Problem_constraints(parameters_constraints)
UTNDP_problem_1.problem_inputs = gc.Problem_inputs(parameters_input)
UTNDP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA_route_design)
UTNDP_problem_1.mapping_adjacent = gf.get_mapping_of_adj_edges(mx_dist) # creates the mapping of all adjacent nodes
UTNDP_problem_1.max_objs = max_objs
UTNDP_problem_1.min_objs = min_objs
UTNDP_problem_1.add_text = "" # define the additional text for the file name
# UTNDP_problem_1.R_routes = R_routes


def main(UTNDP_problem_1):
    
    """ Keep track of the stats """
    stats_overall = {
        'execution_start_time' : datetime.datetime.now() # enter the begin time
        } 

    stats = {} # define the stats dictionary
    
    #%% Define the Objective UTNDP functions
    def fn_obj_2(routes, UTNDP_problem_input):
        return (ev.evalObjs(routes, 
                UTNDP_problem_input.problem_data.mx_dist, 
                UTNDP_problem_input.problem_data.mx_demand, 
                UTNDP_problem_input.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)
    
    def fn_obj_2_row(routes):
        return (ev.evalObjs(routes, 
                UTNDP_problem_1.problem_data.mx_dist, 
                UTNDP_problem_1.problem_data.mx_demand, 
                UTNDP_problem_1.problem_inputs.__dict__)) # returns (f1_ATT, f2_TRT)
    
    # Add/Delete individuals to/from population
    def combine_offspring_with_pop_3(pop, offspring_variables):
        """Function to combine the offspring with the population for the UTNDP routes
        NB: avoid casting lists to numpy arrays, keep it lists"""
        
        len_pop = len(pop.objectives)
        pop.variables = pop.variables + offspring_variables # adds two lists to each other
        
        # TODO: Filter out duplicates
        #is_unique = np.where(np.logical_not(find_duplicates(pop.variables, epsilon=1e-24)))[0]
        #pop.variables = pop.variables[is_unique]
        
        # Only evaluate the offspring
        offspring_variables = pop.variables[len_pop:]
        # offspring_variables_str = list(np.apply_along_axis(gf.convert_routes_list2str, 1, offspring_variables)) # potentially gave errors
        
        offspring_variables_str = [None] * len(offspring_variables)
        for index_i in range(len(offspring_variables)):
            offspring_variables_str[index_i] = gf.convert_routes_list2str(offspring_variables[index_i])
        
        offspring_objectives = np.apply_along_axis(fn_obj_2_row, 1, offspring_variables)   
    
        # Add evaluated offspring to population
        # pop.variables = np.vstack([pop.variables, offspring_variables])
        pop.variables_str = pop.variables_str + offspring_variables_str # adds two lists to each other
        pop.objectives = np.vstack([pop.objectives, offspring_objectives])  
        
        #pop_1.variables_str = np.vstack([pop_1.variables_str, offspring_variables_str])
        # This continues as normal
        pop.rank = np.empty([len(pop.variables), 1])
        pop.crowding_dist = np.empty([len(pop.variables), 1])
        
        # get the objective space values and objects
        F = pop.objectives
    
        # do the non-dominated sorting until splitting front
        fronts = gc.NonDominated_Sorting().do(F)
    
        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = gf.calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop.rank[i] = k
                pop.crowding_dist[i] = crowding_of_front[j]
    
    
    #%% GA Implementation UTNDP 
    
    for run_nr in range(0, parameters_GA_route_design["number_of_runs"]):
        
        # Create the initial population
        # TODO: Insert initial 10000 solutions generatations and NonDom Sort your initial population, ensuring diversity
        # TODO: Remove duplicate functions! (compare set similarity and obj function values)
        
        stats['begin_time'] = datetime.datetime.now() # enter the begin time
        print("######################### RUN {0} #########################".format(run_nr+1))
        print("Generation 0 initiated" + " ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
        pop_1 = gc.PopulationRoutes(UTNDP_problem_1)   
        pop_1.generate_initial_population_robust(UTNDP_problem_1, fn_obj_2) 
        pop_generations = np.hstack([pop_1.objectives, ga.extractDigits(pop_1.variables_str), np.full((len(pop_1.objectives),1),0)])
        
        data_for_analysis = np.hstack([pop_1.objectives, ga.extractDigits(pop_1.variables_str)]) # create an object to contain all the data for analysis
        df_data_generations = pd.DataFrame(columns = ["Generation","HV"]) # create a df to keep data for SA Analysis
        df_data_generations.loc[0] = [0, gf.norm_and_calc_2d_hv_np(pop_1.objectives, UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)]
        
        stats['end_time'] = datetime.datetime.now() # enter the begin time
        print("Generation 0 duration: "+ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']))
        
        
        """Run each generation"""
        for i_generation in range(UTNDP_problem_1.problem_GA_parameters.generations):    
            # Some stats
            stats['begin_time_run'] = datetime.datetime.now() # enter the begin time
            stats['generation'] = i_generation + 1
            print("Generation " + str(int(i_generation+1)) + " initiated ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
            
            # Crossover amd Mutation
            offspring_variables = gf.crossover_pop_routes(pop_1, UTNDP_problem_1)
            
            mutated_variables = gf.mutate_route_population(offspring_variables, UTNDP_problem_1)
            
            
            # Combine offspring with population
            combine_offspring_with_pop_3(pop_1, mutated_variables)
            # TODO: Adding the mutated variables twice and causing weird problems with the other attributes
            
            pop_size = UTNDP_problem_1.problem_GA_parameters.population_size
            data_for_analysis = np.vstack([data_for_analysis, np.hstack([pop_1.objectives[pop_size:,], ga.extractDigits(pop_1.variables_str)[pop_size:]])])
              
            # Get new generation
            survivor_indices = gf.get_survivors(pop_1, pop_size)
            gf.keep_individuals(pop_1, survivor_indices)
        
            # Calculate the HV Quality Measure
            HV = gf.norm_and_calc_2d_hv_np(pop_1.objectives, UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
            df_data_generations.loc[i_generation+1] = [i_generation+1, HV]
        
            pop_generations = np.vstack([pop_generations, np.hstack([pop_1.objectives, ga.extractDigits(pop_1.variables_str), np.full((len(pop_1.objectives),1),i_generation+1)])]) # add the population to the generations
            
            stats['end_time_run'] = datetime.datetime.now() # save the end time of the run
            print("Generation {0} duration: {1} [HV:{2}]".format(str(int(i_generation+1)),
                                                                ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']),
                                                                round(HV, 4)))
                
        #%% Stats updates
        stats['end_time'] = datetime.datetime.now() # save the end time of the run
        stats['duration'] = stats['end_time'] - stats['begin_time'] # calculate and save the duration of the run
        stats['begin_time'] = stats['begin_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
        stats['end_time'] =  stats['end_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
        stats['HV obtained'] = HV
        
            
        #%% Save the results
        if Choice_print_results:
            
            '''Write all results and parameters to files'''
            '''Main folder path'''
            path_parent_folder = Path(os.path.dirname(os.getcwd()))
            path_results = path_parent_folder / ("Results/Results_"+
                                                 parameters_input['Problem_name']+
                                                 "/"+parameters_input['Problem_name']+
                                                 "_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S")+
                                                 " "+parameters_GA_route_design['method']+
                                                 f"_{UTNDP_problem_1.add_text}")
            
            '''Sub folder path'''
            path_results_per_run = path_results / ("Run_"+str(run_nr+1))
            if not path_results_per_run.exists():
                os.makedirs(path_results_per_run)
            
            # Create and save the dataframe 
            df_pop_generations = ga.create_df_from_pop(pop_generations)
            df_non_dominated_set = df_pop_generations.iloc[-UTNDP_problem_1.problem_GA_parameters.population_size:,0:3]
            df_non_dominated_set = df_non_dominated_set[gf.is_pareto_efficient(df_non_dominated_set.values, True)]
            df_non_dominated_set = df_non_dominated_set.sort_values(by='f_1', ascending=True) # sort
            
            df_data_for_analysis = pd.DataFrame(data=data_for_analysis,columns=ga.generate_data_analysis_labels_routes(2,1)) #TODO: automate the 2,6
            
            df_pop_generations.to_csv(path_results_per_run / "Pop_generations.csv")
            df_non_dominated_set.to_csv(path_results_per_run / "Non_dominated_set.csv")
            df_data_for_analysis.to_csv(path_results_per_run / "Data_for_analysis.csv")
            
            df_data_generations = df_data_generations.assign(mean_f_1=df_pop_generations.groupby('generation', as_index=False)['f_1'].mean().iloc[:,1],
                                       mean_f_2=df_pop_generations.groupby('generation', as_index=False)['f_2'].mean().iloc[:,1])
            df_data_generations.to_csv(path_results_per_run / "Data_generations.csv")
            
            json.dump(parameters_input, open(path_results_per_run / "parameters_input.json", "w")) # saves the parameters in a json file
            json.dump(parameters_constraints, open(path_results_per_run / "parameters_constraints.json", "w"))
            json.dump(parameters_GA_route_design, open(path_results_per_run / "parameters_GA_route_design.json", "w"))
            pickle.dump(stats, open(path_results_per_run / "stats.pickle", "ab"))
            
            with open(path_results_per_run / "Run_summary_stats.csv", "w") as archive_file:
                w = csv.writer(archive_file)
                for key, val in {**parameters_input, **parameters_constraints, **parameters_GA_route_design, **stats}.items():
                    w.writerow([key, val])
                del key, val
            
            print("End of generations: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            
            # Visualise the generations
            if False: # becomes useless when more than 10 generations
                gv.plot_generations_objectives(pop_generations)
                
                # Opens and saves the plot
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()
                plt.show()
                plt.savefig(path_results_per_run / "Results_combined.pdf", bbox_inches='tight')
                manager.window.close()
            
            #%% Print and save result summary figures:
                
            if True:   
                if False:
                    '''Print Archive'''   
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    ax1.scatter( df_non_dominated_set["f_1"], df_non_dominated_set["f_2"], s=1, c='b', marker="o", label='Non-dominated set')
                    #ax1.scatter(f_cur[0], f_cur[1], s=1, c='y', marker="o", label='Current')
                    #ax1.scatter(f_new[0], f_new[1], s=1, c='r', marker="o", label='New')
                    plt.legend(loc='upper left');
                    plt.show()
                    
                '''Load validation data'''
                #Mumford_validation_data = pd.read_csv("./Validation_Data/Mumford_results_on_Mandl_2013/MumfordResultsParetoFront_headers.csv")
                John_validation_data = pd.read_csv("./Input_Data/"+name_input_data+"/Validation_data/Results_data_headers.csv")
                #hv_test_data = pd.read_csv("./Validation_Data/FULL_PARETO_SET_BOTH_SA_AND_GA.csv")
                #gf.norm_and_calc_2d_hv(hv_test_data.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
            
                if True:
                    '''Print Objective functions over time, all solutions and pareto set obtained'''
                    fig, axs = plt.subplots(2, 2)
                    fig.set_figheight(15)
                    fig.set_figwidth(20)
                    axs[0, 0].scatter(df_data_generations["Generation"], df_data_generations["mean_f_1"], s=1, c='r', marker="o", label='f1_ATT')
                    axs[0, 0].set_title('Mean ATT over all generations')
                    axs[0, 0].set(xlabel='Generations', ylabel='f1_ATT')
                    axs[0, 0].legend(loc="upper right")
                    
                    axs[1, 0].scatter(df_data_generations["Generation"], df_data_generations["mean_f_2"], s=1, c='b', marker="o", label='f2_TRT')
                    axs[1, 0].set_title('Mean TRT over all generations')
                    axs[1, 0].set(xlabel='Generations', ylabel='f2_TRT')
                    axs[1, 0].legend(loc="upper right") 
                    
                    axs[0, 1].scatter(df_data_generations["Generation"], df_data_generations["HV"], s=1, c='r', marker="o", label='HV obtained')
                    #axs[0, 1].scatter(range(len(df_SA_analysis)), np.ones(len(df_SA_analysis))*gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs),\
                    #   s=1, c='g', marker="o", label='HV Mumford (2013)')
                    axs[0, 1].set_title('HV over all generations')
                    axs[0, 1].set(xlabel='Generations', ylabel='%')
                    axs[0, 1].legend(loc="upper right")
                    
                    #axs[1, 1].scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=10, c='b', marker="o", label='Initial route sets')
                    axs[1, 1].scatter(df_non_dominated_set["f_2"], df_non_dominated_set["f_1"], s=10, c='r', marker="o", label='Non-dom set obtained')
                    #axs[1, 1].scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="o", label='Mumford results (2013)')
                    axs[1, 1].scatter(John_validation_data.iloc[:,1], John_validation_data.iloc[:,0], s=10, c='b', marker="o", label='John results (2016)')
                    axs[1, 1].set_title('Non-dominated set obtained vs benchmark results')
                    axs[1, 1].set(xlabel='f2_TRT', ylabel='f1_ATT')
                    axs[1, 1].legend(loc="upper right")
                    
                    manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()
                    plt.show()
                    plt.savefig(path_results_per_run / "Results_summary.pdf", bbox_inches='tight')
                
                    manager.window.close()
            
    del HV, i_generation, mutated_variables, offspring_variables, pop_size, survivor_indices
    
    # %% Save results after all runs
    if Choice_print_results:
        '''Save the summarised results'''
        df_overall_pareto_set = ga.group_pareto_fronts_from_model_runs_2(path_results, parameters_input, "Non_dominated_set.csv").iloc[:,1:]
        df_overall_pareto_set = df_overall_pareto_set[gf.is_pareto_efficient(df_overall_pareto_set.iloc[:,0:2].values, True)] # reduce the pareto front from the total archive
        df_overall_pareto_set = df_overall_pareto_set.sort_values(by='f_1', ascending=True) # sort
        df_overall_pareto_set.to_csv(path_results / "Overall_Pareto_set.csv")   # save the csv file
        
        
        '''Save the stats for all the runs'''
        # df_routes_R_initial_set.to_csv(path_results / "Routes_initial_set.csv")
        df_durations = ga.get_stats_from_model_runs(path_results)
        
        stats_overall['execution_end_time'] =  datetime.datetime.now()
        
        stats_overall['total_model_runs'] = run_nr + 1
        stats_overall['average_run_time'] = str(df_durations["Duration"].mean())
        stats_overall['total_duration'] = stats_overall['execution_end_time']-stats_overall['execution_start_time']
        stats_overall['execution_start_time'] = stats_overall['execution_start_time'].strftime("%m/%d/%Y, %H:%M:%S")
        stats_overall['execution_end_time'] = stats_overall['execution_end_time'].strftime("%m/%d/%Y, %H:%M:%S")
        #stats_overall['HV initial set'] = gf.norm_and_calc_2d_hv(df_routes_R_initial_set.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        stats_overall['HV obtained'] = gf.norm_and_calc_2d_hv(df_overall_pareto_set.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        #stats_overall['HV Benchmark Mumford 2013'] = gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        stats_overall['HV Benchmark John 2016'] = gf.norm_and_calc_2d_hv(John_validation_data.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        
        df_durations.loc[len(df_durations)] = ["Average", df_durations["Duration"].mean()]
        df_durations.to_csv(path_results / "Run_durations.csv")
        del df_durations
        
        with open(path_results / "Stats_overall.csv", "w") as archive_file:
            w = csv.writer(archive_file)
            for key, val in {**stats_overall,
                             **UTNDP_problem_1.problem_inputs.__dict__, 
                             **UTNDP_problem_1.problem_constraints.__dict__, 
                             **UTNDP_problem_1.problem_GA_parameters.__dict__}.items():
                w.writerow([key, val])
            del key, val
      
        ga.get_sens_tests_stats_from_model_runs(path_results, parameters_GA_route_design["number_of_runs"]) # prints the runs summary
        ga.get_sens_tests_stats_from_UTRP_GA_runs(path_results)            
        
        # %% Plot analysis graph
        '''Plot the analysis graph'''
        if True:
            fig, axs = plt.subplots(1,1)
            fig.set_figheight(15)
            fig.set_figwidth(20)
            
            # axs.scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=20, c='b', marker="o", label='Initial route sets')
            axs.scatter(df_overall_pareto_set["f_2"], df_overall_pareto_set["f_1"], s=10, c='r', marker="o", label='Pareto front obtained from all runs')
            #axs.scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="x", label='Mumford results (2013)')
            axs.scatter(John_validation_data.iloc[:,1], John_validation_data.iloc[:,0], s=10, c='b', marker="o", label='John results (2016)')
            axs.set_title('Pareto front obtained vs validation Results')
            axs.set(xlabel='f2_TRT', ylabel='f1_ATT')
            axs.legend(loc="upper right")
            del axs
            
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()
            plt.savefig(path_results / "Results_combined.pdf", bbox_inches='tight')
            manager.window.close()
            del fig, manager
        del archive_file, data_for_analysis, df_non_dominated_set, path_parent_folder, path_results, path_results_per_run, pop_generations, w
        
        
    del run_nr

# %% Sensitivity analysis
''' Sensitivity analysis tests'''

# Single Thread
if __name__ == "__main__":
    
    if Choice_conduct_sensitivity_analysis:
        start = time.perf_counter()
        ''' Create copies of the original input data '''
        #original_UTNDP_problem_1 = copy.deepcopy(UTNDP_problem_1)    
        #original_parameters_constraints = copy.deepcopy(parameters_constraints)  
        #original_parameters_GA_route_design = copy.deepcopy(parameters_GA_route_design)  
        
        # Set up the list of parameters to test
        sensitivity_list = [#[parameters_constraints, "con_r", 6, 7, 8], # TODO: add 4 , find out infeasibility
                            #[parameters_constraints, "con_minNodes", 2, 3, 4, 5],
                            [parameters_GA_route_design, "population_size", 10, 20, 50, 100, 150, 200, 300],
                            [parameters_GA_route_design, "generations", 10, 20, 50, 100, 150, 200, 300],
                            [parameters_GA_route_design, "crossover_probability", 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                            [parameters_GA_route_design, "mutation_probability", 0.05, 0.1, 1/parameters_constraints["con_r"], 0.2, 0.3, 0.5, 0.7, 0.9, 1]
                            ]
        
        sensitivity_list = [
                            #[parameters_GA_route_design, "population_size", 10, 20, 50, 100, 150, 200, 300], 
                            #[parameters_GA_route_design, "generations", 10, 20, 50, 100, 150, 200, 300],
                            #[parameters_GA_route_design, "crossover_probability", 0.5, 0.6, 0.7, 0.8],
                            #[parameters_GA_route_design, "crossover_probability", 0.9, 0.95, 1],
                            #[parameters_GA_route_design, "mutation_probability", 0.05, 0.1, 1/parameters_constraints["con_r"], 0.2, 0.3],                   
                            [parameters_GA_route_design, "mutation_probability", 0.5, 0.7, 0.9, 1]
                            ]
        
        #sensitivity_list = [#[parameters_GA_route_design, "population_size", 20, 400], 
                            #[parameters_GA_route_design, "generations", 20, 400],
                            #[parameters_GA_route_design, "crossover_probability", 0.1],
                            #[parameters_GA_route_design, "crossover_probability", 1],
                            #[parameters_GA_route_design, "mutation_probability", 0.05], 
                            #[parameters_GA_route_design, "mutation_probability", 0.9],
                            #[parameters_GA_route_design, "mutation_probability", 1],  
                            #]
        
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
                UTNDP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA_route_design)
                
                # Run model
                main(UTNDP_problem_1)
                
                # Reset the original parameters
                parameter_dict[dict_entry] = temp_storage
        
        finish = time.perf_counter()
        
        print(f'Finished in {round(finish-start, 6)} second(s)')
        
