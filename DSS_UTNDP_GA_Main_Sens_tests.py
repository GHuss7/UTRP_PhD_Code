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
#import pygmo as pg
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

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning) # find VisibleDeprecationWarning
    
# %% Load the respective files
name_input_data = ["Mandl_UTRP", #0
                   "Mumford0_UTRP", #1
                   "Mumford1_UTRP", #2
                   "Mumford2_UTRP", #3
                   "Mumford3_UTRP",][2]   # set the name of the input data

# %% Set input parameters
sens_from = 1
sens_to = (sens_from + 1) if True else -1
if True:
    Decisions = json.load(open("./Input_Data/"+name_input_data+"/Decisions.json"))

else:
    Decisions = {
    "Choice_print_results" : True, 
    "Choice_conduct_sensitivity_analysis" : True,
    "Choice_import_dictionaries" : True,
    "Choice_print_full_data_for_analysis" : True,
    "Choice_relative_results_referencing" : False,
    "Additional_text" : "Tests"
    }

# %% Load the respective files
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)

#%% Load and set problem parameters #######################################
if Decisions["Choice_import_dictionaries"]:
    parameters_constraints = json.load(open("./Input_Data/"+name_input_data+"/parameters_constraints.json"))
    parameters_input = json.load(open("./Input_Data/"+name_input_data+"/parameters_input.json"))
    #parameters_GA = json.load(open("./Input_Data/"+name_input_data+"/parameters_GA.json"))
   
    '''State the various GA input parameters for frequency setting''' 
    parameters_GA={
    "method" : "GA",
    "population_size" : 200, #should be an even number STANDARD: 200 (John 2016)
    "generations" : 200, # STANDARD: 200 (John 2016)
    "number_of_runs" : 10, # STANDARD: 20 (John 2016)
    "crossover_probability" : 0.6, 
    "crossover_distribution_index" : 5,
    "mutation_probability" : 1, # John: 1/|Route set| -> set later
    "mutation_distribution_index" : 10,
    "mutation_ratio" : 0.2, # Ratio used for the probabilites of mutations applied
    "tournament_size" : 2,
    "termination_criterion" : "StoppingByEvaluations",
    "max_evaluations" : 25000,
    "number_of_variables" : parameters_constraints["con_r"],
    "number_of_objectives" : 2, # this could still be automated in the future
    "Number_of_initial_solutions" : 10000 # number of initial solutions to be generated and chosen from
    }
 
else:    
    '''State the various parameter constraints''' 
    parameters_constraints = {
    'con_r' : 6,               # number of allowed routes (aim for > [numNodes N ]/[maxNodes in route])
    'con_minNodes' : 2,                        # minimum nodes in a route
    'con_maxNodes' : 10,                       # maximum nodes in a route
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
    parameters_GA={
    "method" : "GA",
    "population_size" : 10, #should be an even number STANDARD: 200 (John 2016)
    "generations" : 10, # STANDARD: 200 (John 2016)
    "number_of_runs" : 1, # STANDARD: 20 (John 2016)
    "crossover_probability" : 0.6, 
    "crossover_distribution_index" : 5,
    "mutation_probability" : 1, # John: 1/|Route set| -> set later
    "mutation_distribution_index" : 10,
    "mutation_ratio" : 0.1, # Ratio used for the probabilites of mutations applied
    "tournament_size" : 2,
    "termination_criterion" : "StoppingByEvaluations",
    "max_evaluations" : 25000,
    "number_of_variables" : parameters_constraints["con_r"],
    "number_of_objectives" : 2, # this could still be automated in the future
    "Number_of_initial_solutions" : 10000 # number of initial solutions to be generated and chosen from
    }
    
# Sensitivity analysis lists    
    sensitivity_list = [#[parameters_constraints, "con_r", 6, 7, 8], # TODO: add 4 , find out infeasibility
                        #[parameters_constraints, "con_minNodes", 2, 3, 4, 5],
                        ["population_size", 10, 20, 50, 100, 150, 200, 300, 400],
                        ["generations", 10, 20, 50, 100, 150, 200, 300, 400],
                        ["crossover_probability", 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                        ["mutation_probability", 0.05, 0.1, 1/parameters_constraints["con_r"], 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                        ["mutation_ratio", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        ["population_size", 10]
                        ]
    
    #sensitivity_list = [
                        #["population_size", 10, 20, 50, 100, 150, 200, 300], 
                        #["generations", 10, 20, 50, 100, 150, 200, 300],
                        #["crossover_probability", 0.5, 0.6, 0.7, 0.8],
                        #["crossover_probability", 0.9, 0.95, 1],
                        #["mutation_probability", 0.05, 0.1, 1/parameters_constraints["con_r"], 0.2, 0.3],                   
                        #["mutation_probability", 0.5, 0.7, 0.9, 1],
                        #["mutation_ratio", 0.1, 0.2]
                        #]
    
    #sensitivity_list = [
                        #["population_size", 20, 400], 
                        #["generations", 20, 400],
                        #["crossover_probability", 0.1],
                        #["crossover_probability", 1],
                        #["mutation_probability", 0.05], 
                        #["mutation_probability", 0.9],
                        #["mutation_probability", 1],  
                        
                        # ["mutation_ratio", 0.01],
                        # ["mutation_ratio", 0.05],
                        # ["mutation_ratio", 0.1],
                        # ["mutation_ratio", 0.2],
                        # ["mutation_ratio", 0.3],
                        # ["mutation_ratio", 0.4],
                        # ["mutation_ratio", 0.5],
                        # ["mutation_ratio", 0.6],
                        # ["mutation_ratio", 0.7],
                        # ["mutation_ratio", 0.8],
                        # ["mutation_ratio", 0.9],
                        # ["mutation_ratio", 0.95],
                        # ]
    
    sensitivity_list = sensitivity_list[sens_from:sens_to] # truncates the sensitivity list


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


#%% Define the UTNDP Problem      
UTNDP_problem_1 = gc.UTNDP_problem()
UTNDP_problem_1.problem_data = gc.Problem_data(mx_dist, mx_demand, mx_coords)
UTNDP_problem_1.problem_constraints = gc.Problem_constraints(parameters_constraints)
UTNDP_problem_1.problem_inputs = gc.Problem_inputs(parameters_input)
UTNDP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA)
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
        offspring_variables = pop.variables[len_pop:] # this is done so that if duplicates are removed, no redundant calculations are done
        # offspring_variables_str = list(np.apply_along_axis(gf.convert_routes_list2str, 1, offspring_variables)) # potentially gave errors
        
        offspring_variables_str = [None] * len(offspring_variables)
        offspring_objectives = np.empty([len_pop, pop.objectives.shape[1]])
        
        for index_i in range(len(offspring_variables)):
            offspring_variables_str[index_i] = gf.convert_routes_list2str(offspring_variables[index_i])
            
            offspring_objectives[index_i,] = fn_obj_2_row(offspring_variables[index_i])
        # offspring_objectives = np.apply_along_axis(fn_obj_2_row, 1, offspring_variables)   #gave VisibleDeprecationWarning error, rather loop
    
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
    
    
    #%% GA Implementation UTNDP ############################################
    '''Load validation data'''
    validation_data = pd.read_csv("./Input_Data/"+name_input_data+"/Validation_Data/Results_data_headers.csv")
    
    '''Main folder path'''
    if Decisions["Choice_relative_results_referencing"]:
        path_parent_folder = Path(os.path.dirname(os.getcwd()))
    else:
        path_parent_folder = Path("C:/Users/17832020/OneDrive - Stellenbosch University/Academics 2019 MEng/DSS")
    
    path_results = path_parent_folder / ("Results/Results_"+
                                         name_input_data+
                                         "/"+name_input_data+
                                         "_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S")+
                                         " "+parameters_GA['method']+
                                         f"_{UTNDP_problem_1.add_text}")
    
    '''Save the parameters used in the runs'''
    if not (path_results / "Parameters").exists():
        os.makedirs(path_results / "Parameters")
    
    json.dump(parameters_input, open(path_results / "Parameters" / "parameters_input.json", "w")) # saves the parameters in a json file
    json.dump(parameters_constraints, open(path_results / "Parameters" / "parameters_constraints.json", "w"))
    json.dump(parameters_GA, open(path_results / "Parameters" / "parameters_GA.json", "w"))
    json.dump(Decisions, open(path_results / "Parameters" / "Decisions.json", "w"))
    if Decisions["Choice_conduct_sensitivity_analysis"]:
        json.dump(sensitivity_list, open(path_results / "Parameters" / 'Sensitivity_list.txt', 'w'))
    
    
    '''################## Initiate the runs ##################'''
    for run_nr in range(1, parameters_GA["number_of_runs"]+1):

        if Decisions["Choice_print_results"]:           
            '''Sub folder path'''
            path_results_per_run = path_results / ("Run_"+str(run_nr))
            if not path_results_per_run.exists():
                os.makedirs(path_results_per_run)  
                
        # Create the initial population
        # TODO: Insert initial 10000 solutions generatations and NonDom Sort your initial population, ensuring diversity
        # TODO: Remove duplicate functions! (compare set similarity and obj function values)
        
        stats['begin_time'] = datetime.datetime.now() # enter the begin time
        print("######################### RUN {0} #########################".format(run_nr))
        print("Generation 0 initiated" + " ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
        pop_1 = gc.PopulationRoutes(UTNDP_problem_1)   
        pop_1.generate_initial_population_robust(UTNDP_problem_1, fn_obj_2) 
        
        # Create generational dataframe
        pop_generations = np.hstack([pop_1.objectives, ga.extractDigits(pop_1.variables_str), np.full((len(pop_1.objectives),1),0)])
        df_pop_generations = ga.add_UTRP_pop_generations_data(pop_1, UTNDP_problem_1, generation_num=0)
        
        # Create data for analysis dataframe
        df_data_for_analysis = ga.add_UTRP_analysis_data_with_generation_nr(pop_1, UTNDP_problem_1, generation_num=0) 
        
        # Determine non-dominated set
        df_non_dominated_set = gf.create_non_dom_set_from_dataframe(df_data_for_analysis, obj_1_name='f_1', obj_2_name='f_2')
        HV = gf.norm_and_calc_2d_hv_np(df_non_dominated_set[["f_1","f_2"]].values, UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs) # Calculate HV

        
        df_data_generations = pd.DataFrame(columns = ["Generation","HV"]) # create a df to keep data for SA Analysis
        df_data_generations.loc[0] = [0, HV]
        
        stats['end_time'] = datetime.datetime.now() # enter the begin time
        
        initial_set = df_pop_generations.iloc[0:UTNDP_problem_1.problem_GA_parameters.population_size,:] # load initial set

        print("Generation {0} duration: {1} [HV:{2}]".format(str(0),
                                                        ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']),
                                                        round(HV, 4)))
        
        """ ######## Run each generation ################################################################ """
        for i_generation in range(1, UTNDP_problem_1.problem_GA_parameters.generations + 1):    
            # Some stats
            stats['begin_time_gen'] = datetime.datetime.now() # enter the begin time
            stats['generation'] = i_generation
            
            if i_generation % 20 == 0 or i_generation == UTNDP_problem_1.problem_GA_parameters.generations:
                print("Generation " + str(int(i_generation)) + " initiated ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
            
            # Crossover amd Mutation
            offspring_variables = gf.crossover_pop_routes_individuals(pop_1, UTNDP_problem_1)
            
            mutated_variables = gf.mutate_route_population(offspring_variables, UTNDP_problem_1)
            
            
            # Combine offspring with population
            combine_offspring_with_pop_3(pop_1, mutated_variables)
            # TODO: Adding the mutated variables twice and causing weird problems with the other attributes
            
            # Append data for analysis
            pop_size = UTNDP_problem_1.problem_GA_parameters.population_size
            df_data_for_analysis = ga.add_UTRP_analysis_data_with_generation_nr(pop_1, UTNDP_problem_1, i_generation, df_data_for_analysis) 

            # Determine non-dominated set
            df_non_dominated_set = gf.create_non_dom_set_from_dataframe(df_data_for_analysis, obj_1_name='f_1', obj_2_name='f_2')
            HV = gf.norm_and_calc_2d_hv_np(df_non_dominated_set[["f_1","f_2"]].values, UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs) # Calculate HV

            # Calculate the HV Quality Measure
            HV = gf.norm_and_calc_2d_hv_np(pop_1.objectives, UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
            df_data_generations.loc[i_generation] = [i_generation, HV]
            
            # Intermediate print-outs for observance 
            df_data_for_analysis.to_csv(path_results_per_run / "Data_for_analysis.csv")
            df_pop_generations.to_csv(path_results_per_run / "Pop_generations.csv")
            df_non_dominated_set.to_csv(path_results_per_run / "Non_dominated_set.csv")
            df_data_generations.to_csv(path_results_per_run / "Data_generations.csv")
            
            if i_generation % 20 == 0 or i_generation == UTNDP_problem_1.problem_GA_parameters.generations:
                gv.save_results_analysis_fig_interim_UTRP(initial_set, df_non_dominated_set, validation_data, df_data_generations, name_input_data, path_results_per_run) 
            #gv.save_results_analysis_fig_interim_save_all(initial_set, df_non_dominated_set, validation_data, df_data_generations, name_input_data, path_results_per_run, add_text=i_generation)
                
            # Get new generation
            pop_size = UTNDP_problem_1.problem_GA_parameters.population_size
            survivor_indices = gf.get_survivors(pop_1, pop_size)
            gf.keep_individuals(pop_1, survivor_indices)
        
            # Adds the population to the dataframe
            df_pop_generations = ga.add_UTRP_pop_generations_data(pop_1, UTNDP_problem_1, i_generation, df_pop_generations)
            pop_generations = np.vstack([pop_generations, np.hstack([pop_1.objectives, ga.extractDigits(pop_1.variables_str), np.full((len(pop_1.objectives),1),i_generation)])]) # add the population to the generations
            
            stats['end_time_gen'] = datetime.datetime.now() # save the end time of the run
            
            if i_generation % 20 == 0 or i_generation == UTNDP_problem_1.problem_GA_parameters.generations:
                print("Generation {0} duration: {1} [HV:{2}]".format(str(int(i_generation)),
                                                                ga.print_timedelta_duration(stats['end_time_gen'] - stats['begin_time_gen']),
                                                                round(HV, 4)))
                
        #%% Stats updates
        stats['end_time'] = datetime.datetime.now() # save the end time of the run
        stats['duration'] = stats['end_time'] - stats['begin_time'] # calculate and save the duration of the run
        stats['begin_time'] = stats['begin_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
        stats['end_time'] =  stats['end_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
        stats['HV obtained'] = HV
        
            
        #%% Save the results #####################################################
        if Decisions["Choice_print_results"]:
            '''Write all results to files'''
            
            # Create and save the dataframe 
            df_non_dominated_set = gf.create_non_dom_set_from_dataframe(df_data_for_analysis, obj_1_name='f_1', obj_2_name='f_2')
            
            # Compute means for generations
            df_data_generations = df_data_generations.assign(mean_f_1=df_pop_generations.groupby('Generation', as_index=False)['f_1'].mean().iloc[:,1],
                                       mean_f_2=df_pop_generations.groupby('Generation', as_index=False)['f_2'].mean().iloc[:,1])
            
            """Print-outs for observations"""
            df_pop_generations.to_csv(path_results_per_run / "Pop_generations.csv")
            df_non_dominated_set.to_csv(path_results_per_run / "Non_dominated_set.csv")
            df_data_for_analysis.to_csv(path_results_per_run / "Data_for_analysis.csv")
            df_data_generations.to_csv(path_results_per_run / "Data_generations.csv")
            
            # Print and save result summary figures:
            labels = ["f_1", "f_2", "f1_AETT", "f2_TBR"] # names labels for the visualisations
            gv.save_results_analysis_fig(initial_set, df_non_dominated_set, validation_data, df_data_generations, name_input_data, path_results_per_run, labels)
            
            #%% Post analysis 
            pickle.dump(stats, open(path_results_per_run / "stats.pickle", "ab"))
            
            with open(path_results_per_run / "Run_summary_stats.csv", "w") as archive_file:
                w = csv.writer(archive_file)
                for key, val in {**parameters_input, **parameters_constraints, **parameters_GA, **stats}.items():
                    w.writerow([key, val])
                del key, val
            
            print("End of generations: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            
            # Visualise the generations
            if False: # becomes useless when more than 10 generations
                gv.plot_generations_objectives(df_pop_generations[["f_1", "f_2"]])
                
                # Opens and saves the plot
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()
                plt.show()
                plt.savefig(path_results_per_run / "Results_combined.pdf", bbox_inches='tight')
                manager.window.close()
            
            
            
    del HV, i_generation, mutated_variables, offspring_variables, pop_size, survivor_indices
    
    # %% Save results after all runs
    if Decisions["Choice_print_results"]:
        '''Save the summarised results'''
        df_overall_pareto_set = ga.group_pareto_fronts_from_model_runs_2(path_results, parameters_input, "Non_dominated_set.csv").iloc[:,1:]
        df_overall_pareto_set = df_overall_pareto_set[gf.is_pareto_efficient(df_overall_pareto_set[["f_1","f_2"]].values, True)] # reduce the pareto front from the total archive
        df_overall_pareto_set = df_overall_pareto_set.sort_values(by='f_1', ascending=True) # sort
        df_overall_pareto_set.to_csv(path_results / "Overall_Pareto_set.csv")   # save the csv file
        
        '''Save the stats for all the runs'''
        # df_routes_R_initial_set.to_csv(path_results / "Routes_initial_set.csv")
        df_durations = ga.get_stats_from_model_runs(path_results)
        
        stats_overall['execution_end_time'] =  datetime.datetime.now()
        
        stats_overall['total_model_runs'] = run_nr
        stats_overall['average_run_time'] = str(df_durations["Duration"].mean())
        stats_overall['total_duration'] = stats_overall['execution_end_time']-stats_overall['execution_start_time']
        stats_overall['execution_start_time'] = stats_overall['execution_start_time'].strftime("%m/%d/%Y, %H:%M:%S")
        stats_overall['execution_end_time'] = stats_overall['execution_end_time'].strftime("%m/%d/%Y, %H:%M:%S")
        #stats_overall['HV initial set'] = gf.norm_and_calc_2d_hv(df_routes_R_initial_set.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        stats_overall['HV obtained'] = gf.norm_and_calc_2d_hv(df_overall_pareto_set[["f_1","f_2"]], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        #stats_overall['HV Benchmark Mumford 2013'] = gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        stats_overall['HV Benchmark'] = gf.norm_and_calc_2d_hv(validation_data.iloc[:,0:2], UTNDP_problem_1.max_objs, UTNDP_problem_1.min_objs)
        
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
      
        ga.get_sens_tests_stats_from_model_runs(path_results, parameters_GA["number_of_runs"]) # prints the runs summary
        # ga.get_sens_tests_stats_from_UTRP_GA_runs(path_results) 

        del archive_file, path_results_per_run, w           
        
        # %% Plot analysis graph
        '''Plot the analysis graph'''
        gv.save_results_combined_fig(initial_set, df_overall_pareto_set, validation_data, name_input_data, Decisions, path_results, labels)
        
    del run_nr

# %% Sensitivity analysis
''' Sensitivity analysis tests'''

if __name__ == "__main__":
    
    if Decisions["Choice_conduct_sensitivity_analysis"]:
        start = time.perf_counter()
 
         # define empty list
        sensitivity_list = []

        # open file and read the content in a list
        with open(("./Input_Data/"+name_input_data+"/Sensitivity_list.txt"), 'r') as filehandle:
            sensitivity_list = json.load(filehandle)   
 
        sensitivity_list = sensitivity_list[sens_from:sens_to]   
 
        for parameter_index in range(len(sensitivity_list)):
            sensitivity_list[parameter_index].insert(0, parameters_GA)
        
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
                UTNDP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA)
                
                # Run model
                main(UTNDP_problem_1)
                
                # Reset the original parameters
                parameter_dict[dict_entry] = temp_storage
        
        finish = time.perf_counter()
        
        print(f'Finished in {round(finish-start, 6)} second(s)')
        
