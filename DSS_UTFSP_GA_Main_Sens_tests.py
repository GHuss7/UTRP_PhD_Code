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
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import EvaluateRouteSet as ev
import DSS_UTNDP_Classes as gc

# todo def main_dss(): # create a main function to encapsulate the main body
# def main():
    
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

from pymoo.model.selection import Selection
from pymoo.util.misc import random_permuations
    
# %% Load the respective files
name_input_data = "Mandl_Data"      # set the name of the input data
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)
del name_input_data

# %% Set input parameters
Choice_generate_initial_set = True 
Choice_print_results = True 
Choice_conduct_sensitivity_analysis = True    
Choice_print_full_data_for_analysis = True

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
'Problem_name' : "Mandl_UTFSP", # Specify the name of the problem currently being addresses
'walkFactor' : 100, # factor it takes longer to walk than to drive
'boardingTime' : 0.1, # assume boarding and alighting time = 6 seconds
'alightingTime' : 0.1, # problem when alighting time = 0 (good test 0.5)(0.1 also works)
'large_dist' : int(mx_dist.max()), # the large number from the distance matrix
'alpha_const_inter' : 0.5 # constant for interarrival times relationship 0.5 (Spiess 1989)
}

'''State the various GA input parameters for frequency setting''' 
parameters_GA_frequencies={
"method" : "GA",
"population_size" : 2, #should be an even number, John: 200
"generations" : 1, # John: 200
"number_of_runs" : 1, # John: 20
"crossover_probability" : 0.8,  # John: 0.9
"crossover_distribution_index" : 5,
"mutation_probability" : 0.1, #1/parameters_constraints["con_r"], # John: 1/|Route set| -> set later
"mutation_distribution_index" : 10,
"tournament_size" : 2,
"termination_criterion" : "StoppingByEvaluations",
"max_evaluations" : 40000,
"number_of_variables" : "not_set",
"number_of_objectives" : 2 # this could still be automated in the future
}


#%% Input parameter tests

'''Test the inputs for feasibility'''
# Test feasibility if there are enough buses to cover each route once
if parameters_constraints["con_r"] > parameters_constraints["con_fleet_size"]:
    print("Warning: Number of available vehicles are less than the number of routes.\n"\
          "Number of routes allowed set to "+ str(parameters_constraints["con_r"]))
    parameters_constraints["con_r"] = parameters_constraints["con_fleet_size"]

# %% Import the route set to be evaluated
''' Import the route set '''
# R_x = gf.convert_routes_str2list("13-14-10-8-15*4-2-3-6*13-11-12*7-15-9*5-4-6-15*3-2-1*") # unsure of source

# R_x = gf.convert_routes_str2list("10-9-7-5-3-4-1-0*9-13-12-10-11-3-1-0*5-3-11-10-9-6-14-8*6-14-7-5-3-4-1-2*12-10-9-7-5-2-1-0*0-1-2-5-14-6-9-7*)
# John Mumford results longest route set on Pareto front 2016, 
    #f_1 ATT = 10.19203597	and f_2 TRT = 217                              		

R_x = gf.convert_routes_str2list("4-3-1*13-12*8-14*9-10-12*9-6-14-7-5-2-1-0*10-11*") # John Mumford results shortest route set on Pareto front 2016, 
    #f_1 ATT = 13.48041105	and f_2 TRT = 63

if 0 not in set([y for x in R_x for y in x]): # NB: test whether the route is in the correct format containing a 0
    for i in range(len(R_x)): # get routes in the correct format
        R_x[i] = [x - 1 for x in R_x[i]] # subtract 1 from each element in the list
    del i
R_routes = gf2.Routes(R_x)

parameters_GA_frequencies["mutation_probability"] = 1/(len(R_routes.routes))

# %% Initialise the decision variables
'''Initialise the decision variables'''
F_frequencies = gf2.Frequencies(parameters_constraints['con_r']) 

F_frequencies.set_frequencies(np.full(len(R_x), 1/5)) 
    
F_x = F_frequencies.frequencies

parameters_GA_frequencies["number_of_variables"] = len(F_x)

#%% Define the UTFSP Problem      
UTFSP_problem_1 = gf2.UTFSP_problem()
UTFSP_problem_1.problem_data = gf2.Problem_data(mx_dist, mx_demand, mx_coords)
UTFSP_problem_1.problem_constraints = gf2.Problem_constraints(parameters_constraints)
UTFSP_problem_1.problem_inputs = gf2.Problem_inputs(parameters_input)
UTFSP_problem_1.problem_GA_parameters = gf2.Problem_GA_inputs(parameters_GA_frequencies)
UTFSP_problem_1.R_routes = R_routes
UTFSP_problem_1.add_text = "" # define the additional text for the file name

#%% Define the Transit network
TN = gf2.Transit_network(R_x, F_x, mx_dist, mx_demand, parameters_input) 
            

# %% Class: PopulationFreq
class PopulationFreq(gf2.Frequencies):
    """A class for storing the population consisting of arrays"""
    def __init__(self, main_problem):
        super(gf2.Frequencies, self).__init__()
        self.population_size = main_problem.problem_GA_parameters.population_size
        self.variable_args = np.empty([main_problem.problem_GA_parameters.population_size,
                                   main_problem.problem_GA_parameters.number_of_variables]).astype(int)
        self.variables = np.empty([main_problem.problem_GA_parameters.population_size,
                                   main_problem.problem_GA_parameters.number_of_variables])
        self.objectives = np.empty([main_problem.problem_GA_parameters.population_size,
                                   main_problem.problem_GA_parameters.number_of_objectives])
        self.rank = np.empty([main_problem.problem_GA_parameters.population_size,
                                   1])
        self.crowding_dist = np.empty([main_problem.problem_GA_parameters.population_size,
                                   1])
        
    
    def generate_initial_population(self, main_problem, fn_obj):
        for i in range(self.population_size):
            self.variable_args[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_args()
            self.variables[i,] = 1/gf2.Frequencies.theta_set[self.variable_args[i,]]
            self.objectives[i,] = fn_obj(self.variables[i,], main_problem)
            
            # get the objective space values and objects
            # F = pop.get("F").astype(np.float, copy=False)
            F = self.objectives
        
            # do the non-dominated sorting until splitting front
            fronts = NonDominated_Sorting().do(F)

            for k, front in enumerate(fronts):
        
                # calculate the crowding distance of the front
                crowding_of_front = calc_crowding_distance(F[front, :])
        
                # save rank and crowding in the individual class
                for j, i in enumerate(front):
                    self.rank[i] = k
                    self.crowding_dist[i] = crowding_of_front[j]
                    
            
    def get_summary(self):
        df_summary = pd.DataFrame()
        df_summary = df_summary.assign(f_1 = self.objectives[:,0],
                          f_2 = self.objectives[:,1],
                          rank = self.rank[:,0],
                          crowding_dist = self.crowding_dist[:,0])
        
        df_summary = df_summary.sort_values(by='f_1', ascending=True)
        return df_summary
    
    
    def plot_objectives(self, plot_all_black=False):
        # function to visualise the objectives
        
        df_to_plot = pd.DataFrame()
        df_to_plot = df_to_plot.assign(f_1 = self.objectives[:,0],
                          f_2 = self.objectives[:,1],
                          rank = self.rank[:,0])
        
        plt.style.use('seaborn-whitegrid')
        
        if plot_all_black:
            plt.plot(self.objectives[:,0], self.objectives[:,1], 'o', color='black')

        else:
            groups = df_to_plot.groupby("rank")
            for name, group in groups:
                plt.plot(group["f_1"], group["f_2"], marker="o", linestyle="", label=name)
            plt.legend()
    

#%% Class: NonDominated_Sorting
class NonDominated_Sorting:

    def __init__(self, epsilon=0.0, method="fast_non_dominated_sort") -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.method = method

    def do(self, F, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None):
        F = F.astype(np.float)

        # if not set just set it to a very large values because the cython algorithms do not take None
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)

        if self.method == 'fast_non_dominated_sort':
            func = load_function("fast_non_dominated_sort")
        else:
            raise Exception("Unknown non-dominated sorting method: %s" % self.method)

        fronts = func(F, epsilon=self.epsilon)

        # convert to numpy array for each front and filter by n_stop_if_ranked if desired
        _fronts = []
        n_ranked = 0
        for front in fronts:

            _fronts.append(np.array(front, dtype=np.int))

            # increment the n_ranked solution counter
            n_ranked += len(front)

            # stop if more than this solutions are n_ranked
            if n_ranked >= n_stop_if_ranked:
                break

        fronts = _fronts

        if only_non_dominated_front:
            return fronts[0]

        if return_rank:
            rank = rank_from_fronts(fronts, F.shape[0])
            return fronts, rank

        return fronts


def rank_from_fronts(fronts, n):
    # create the rank array and set values
    rank = np.full(n, 1e16, dtype=np.int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank


# Returns all indices of F that are not dominated by the other objective values
def find_non_dominated(F, _F=None):
    M = Dominator.calc_domination_matrix(F, _F)
    I = np.where(np.all(M >= 0, axis=1))[0]
    return I


def get_survivors(pop, n_survive, D=None, **kwargs):

    # get the objective space values and objects
    # F = pop.get("F").astype(np.float, copy=False)
    F = pop.objectives

    # the final indices of surviving individuals
    survivors = []

    # do the non-dominated sorting until splitting front
    fronts = NonDominated_Sorting().do(F, n_stop_if_ranked=n_survive)

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
   
# %% TournamentSelection
# Binary Tournament Selection Function
def binary_tournament_g2(pop, P, tournament_type='comp_by_dom_and_crowding', **kwargs):
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

    return S[:, None].astype(np.int, copy=False)





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
    
    
#%% Functions: Crossover

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


#%% Functions: Mutation
def mutate_pop_args(offspring_variable_args, variable_len, mutation_probability):
    last_arg = gf2.Frequencies.theta_set_len-1
    
    for i in range(len(offspring_variable_args)):
        for j in range(variable_len):
                    
            if random.random() < mutation_probability:
                if offspring_variable_args[i,j] == 0: # case for the first position
                    
                    if random.random() < 0.5:
                        offspring_variable_args[i,j] = last_arg
                    else:
                        offspring_variable_args[i,j] = 1
                    
                elif offspring_variable_args[i,j] == last_arg: # case for the last position
                    
                    if random.random() < 0.5:
                        offspring_variable_args[i,j] = last_arg -1
                    else:
                        offspring_variable_args[i,j] = 0
                
                else:
                    if random.random() < 0.5:
                        offspring_variable_args[i,j] = offspring_variable_args[i,j] -1
                    else:
                        offspring_variable_args[i,j] = offspring_variable_args[i,j] +1
    
    return offspring_variable_args


def keep_individuals(pop, survivor_indices):
    # Function that only keeps to individuals with the specified indices
    pop.variable_args = pop.variable_args[survivor_indices,]
    pop.variables = pop.variables[survivor_indices,]
    pop.objectives = pop.objectives[survivor_indices,]  
    pop.rank = pop.rank[survivor_indices]
    pop.crowding_dist = pop.crowding_dist[survivor_indices]

#%% Funtion: Visualisation of generations
def plot_generations_objectives(pop_generations):
    # function to visualise the different populations
    
    df_to_plot = pd.DataFrame()
    df_to_plot = df_to_plot.assign(f_1 = pop_generations[:,0],
                      f_2 = pop_generations[:,1],
                      generation = pop_generations[:,2])
    
    plt.style.use('seaborn-whitegrid')
    
    groups = df_to_plot.groupby("generation")
    for name, group in groups:
        plt.plot(group["f_1"], group["f_2"], marker="o", linestyle="", label=name)
    plt.legend()
    
def create_df_from_pop(pop_generations):
    df_pop = pd.DataFrame()
    df_pop = df_pop.assign(f_1 = pop_generations[:,0],
                      f_2 = pop_generations[:,1],
                      generation = pop_generations[:,2])
    return df_pop


def generate_data_analysis_labels(num_objectives, num_variables):
    # format 'x' with number
    label_names = []
    for i in range(num_objectives):
        label_names.append("f_"+str(i+1))
    for j in range(num_variables):
        label_names.append("x_"+str(j+1))
    return label_names


def main(UTFSP_problem_1):

    """ Keep track of the stats """
    stats_overall = {
        'execution_start_time' : datetime.datetime.now()} # enter the begin time

    stats = {} # define the stats dictionary
    
    #%% Define the Objective functions
    def fn_obj(frequencies, UTFSP_problem_input):
        return (gf2.f3_ETT(UTFSP_problem_input.R_routes.routes,
                           frequencies, 
                           UTFSP_problem_input.problem_data.mx_dist, 
                           UTFSP_problem_input.problem_data.mx_demand, 
                           UTFSP_problem_input.problem_inputs.__dict__), #f3_ETT
                gf2.f4_TBR(UTFSP_problem_input.R_routes.routes, 
                           frequencies, 
                           UTFSP_problem_input.problem_data.mx_dist)) #f4_TBR
    
    def fn_obj_row(frequencies):
        
        return (gf2.f3_ETT(UTFSP_problem_1.R_routes.routes,
                           frequencies, 
                           UTFSP_problem_1.problem_data.mx_dist, 
                           UTFSP_problem_1.problem_data.mx_demand, 
                           UTFSP_problem_1.problem_inputs.__dict__), #f3_ETT
                gf2.f4_TBR(UTFSP_problem_1.R_routes.routes, 
                           frequencies, 
                           UTFSP_problem_1.problem_data.mx_dist)) #f4_TBR
    
    '''Set the reference point for the Hypervolume calculations'''
    parameters_input['ref_point_min_f1_AETT'], parameters_input['ref_point_max_f2_TBR'] = fn_obj(np.full((1,UTFSP_problem_1.problem_constraints.con_r), 1/5)[0],UTFSP_problem_1)
    parameters_input['ref_point_max_f1_AETT'], parameters_input['ref_point_min_f2_TBR'] = fn_obj(np.full((1,UTFSP_problem_1.problem_constraints.con_r), 1/30)[0],UTFSP_problem_1)
    UTFSP_problem_1.max_objs = np.array([parameters_input['ref_point_max_f1_AETT'],parameters_input['ref_point_max_f2_TBR']])
    UTFSP_problem_1.min_objs = np.array([parameters_input['ref_point_min_f1_AETT'],parameters_input['ref_point_min_f2_TBR']])
    
    # TODO: Test errors
    '''Set the reference point for the Hypervolume calculations'''
    parameters_input['ref_point_min_f1_AETT'], parameters_input['ref_point_max_f2_TBR'] = 12, 25
    parameters_input['ref_point_max_f1_AETT'], parameters_input['ref_point_min_f2_TBR'] = 35, 4.2 
    UTFSP_problem_1.max_objs = np.array([parameters_input['ref_point_max_f1_AETT'],parameters_input['ref_point_max_f2_TBR']])
    UTFSP_problem_1.min_objs = np.array([parameters_input['ref_point_min_f1_AETT'],parameters_input['ref_point_min_f2_TBR']])
    
    
    #%% Function: Add/Delete individuals to/from population
    def combine_offspring_with_pop(pop, offspring_variable_args):
        # Function to combine the offspring with the population
        len_pop = len(pop.objectives)
        pop.variable_args = np.vstack([pop.variable_args, offspring_variable_args])
        
        # Filter out duplicates
        is_unique = np.where(np.logical_not(find_duplicates(pop.variable_args, epsilon=1e-24)))[0]
        pop.variable_args = pop.variable_args[is_unique]
        
        # Only evaluate the offspring
        offspring_variables = 1/gf2.Frequencies.theta_set[pop.variable_args[len_pop:,]]
        offspring_objectives = gf2.calc_fn_obj_for_np_array(fn_obj_row, offspring_variables)
    
        # Add evaluated offspring to population
        pop.variables = np.vstack([pop.variables, offspring_variables])
        pop.objectives = np.vstack([pop.objectives, offspring_objectives])  
        
        # This continues as normal
        pop.rank = np.empty([pop.variable_args.shape[0], 1])
        pop.crowding_dist = np.empty([pop.variable_args.shape[0], 1])
        
        # get the objective space values and objects
        F = pop.objectives
    
        # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F)
    
        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop.rank[i] = k
                pop.crowding_dist[i] = crowding_of_front[j]
    
    def combine_offspring_with_pop_multi(pop, offspring_variable_args):
        # Function to combine the offspring with the population
        len_pop = len(pop.objectives)
        pop.variable_args = np.vstack([pop.variable_args, offspring_variable_args])
        
        # Filter out duplicates
        is_unique = np.where(np.logical_not(find_duplicates(pop.variable_args, epsilon=1e-24)))[0]
        pop.variable_args = pop.variable_args[is_unique]
        
        # Only evaluate the offspring
        offspring_variables = 1/gf2.Frequencies.theta_set[pop.variable_args[len_pop:,]]
        # offspring_objectives = gf2.calc_fn_obj_for_np_array(fn_obj_row, offspring_variables)  
    
        # could put offspring variables in a list:
        #     multi_list = []
        # for offspring_variable in offspring_variables:
        #     multi_list.append(offspring_variable)
    
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     offspring_objectives = executor.map(fn_obj_row, multi_list) # map returns processes in order they were started
                
    
        with concurrent.futures.ProcessPoolExecutor() as executor:
                offspring_objectives = executor.map(fn_obj_row, offspring_variables) # map returns processes in order they were started
            
                for result in offspring_objectives:
                    print(result)
    
    
        # Add evaluated offspring to population
        pop.variables = np.vstack([pop.variables, offspring_variables])
        pop.objectives = np.vstack([pop.objectives, offspring_objectives])  
        
        # This continues as normal
        pop.rank = np.empty([pop.variable_args.shape[0], 1])
        pop.crowding_dist = np.empty([pop.variable_args.shape[0], 1])
        
        # get the objective space values and objects
        F = pop.objectives
    
        # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F)
    
        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop.rank[i] = k
                pop.crowding_dist[i] = crowding_of_front[j]
        
    #%% GA Implementation UTFSP
    
    for run_nr in range(0, parameters_GA_frequencies["number_of_runs"]):
    
        # Create the initial populations   
        stats['begin_time'] = datetime.datetime.now() # enter the begin time
        print("######################### RUN {0} #########################".format(run_nr+1))
        print("Generation 0 initiated" + " ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
        pop_1 = PopulationFreq(UTFSP_problem_1)   
        pop_1.generate_initial_population(UTFSP_problem_1, fn_obj) 
        pop_generations = np.hstack([pop_1.objectives, np.full((len(pop_1.objectives),1),0)])
        
        if Choice_print_full_data_for_analysis:
            data_for_analysis = np.hstack([pop_1.objectives, pop_1.variables]) # create an object to contain all the data for analysis
        df_data_generations = pd.DataFrame(columns = ["Generation","HV"]) # create a df to keep data for SA Analysis
        df_data_generations.loc[0] = [0, gf.norm_and_calc_2d_hv_np(pop_1.objectives, UTFSP_problem_1.max_objs, UTFSP_problem_1.min_objs)]
            
        
        stats['end_time'] = datetime.datetime.now() # enter the begin time
        print("Generation 0 duration: "+ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']))
        
        
        """Run each generation"""
        for i_generation in range(UTFSP_problem_1.problem_GA_parameters.generations):    
            # Some stats
            stats['begin_time_run'] = datetime.datetime.now() # enter the begin time
            stats['generation'] = i_generation + 1
            print("Generation " + str(int(i_generation+1)) + " initiated ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
            
            # Crossover amd Mutation
            offspring_variable_args = crossover_pop_uniform_as_is(pop_1, UTFSP_problem_1)
            
            mutated_variable_args = mutate_pop_args(offspring_variable_args, 
                       UTFSP_problem_1.R_routes.number_of_routes,
                       UTFSP_problem_1.problem_GA_parameters.mutation_probability)
            
            # Combine offspring with population
            combine_offspring_with_pop(pop_1, mutated_variable_args)
            
            pop_size = UTFSP_problem_1.problem_GA_parameters.population_size
            if Choice_print_full_data_for_analysis:
                data_for_analysis = np.vstack([data_for_analysis, np.hstack([pop_1.objectives[pop_size:,], pop_1.variables[pop_size:,]])])
              
            # Get new generation
            survivor_indices = get_survivors(pop_1, pop_size)
            keep_individuals(pop_1, survivor_indices)
        
            # Calculate the HV Quality Measure
            HV = gf.norm_and_calc_2d_hv_np(pop_1.objectives, UTFSP_problem_1.max_objs, UTFSP_problem_1.min_objs)
            df_data_generations.loc[i_generation+1] = [i_generation+1, HV]
        
            pop_generations = np.vstack([pop_generations, np.hstack([pop_1.objectives, np.full((len(pop_1.objectives),1),i_generation+1)])]) # add the population to the generations
            
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
                                                 " "+parameters_GA_frequencies['method']+
                                                 f"_{UTFSP_problem_1.add_text}")
            
            '''Sub folder path'''
            path_results_per_run = path_results / ("Run_"+str(run_nr+1))
            if not path_results_per_run.exists():
                os.makedirs(path_results_per_run)
            
            # Create and save the dataframe 
            df_pop_generations = create_df_from_pop(pop_generations)
            
            df_non_dominated_set = pd.DataFrame(data=np.hstack([pop_1.objectives, pop_1.variables]),
                                                columns=generate_data_analysis_labels(UTFSP_problem_1.problem_GA_parameters.number_of_objectives,
                                                                                      UTFSP_problem_1.problem_constraints.con_r))
            df_non_dominated_set = df_non_dominated_set[gf.is_pareto_efficient(df_non_dominated_set.values, True)]
            df_non_dominated_set = df_non_dominated_set.sort_values(by='f_1', ascending=True) # sort
            
            if Choice_print_full_data_for_analysis:
                df_data_for_analysis = pd.DataFrame(data=data_for_analysis,
                                                    columns=generate_data_analysis_labels(UTFSP_problem_1.problem_GA_parameters.number_of_objectives,
                                                                                          UTFSP_problem_1.problem_constraints.con_r))
                df_data_for_analysis.to_csv(path_results_per_run / "Data_for_analysis.csv")
                del data_for_analysis
            
            df_pop_generations.to_csv(path_results_per_run / "Pop_generations.csv")
            df_non_dominated_set.to_csv(path_results_per_run / "Non_dominated_set.csv")
            
            
            df_data_generations = df_data_generations.assign(mean_f_1=df_pop_generations.groupby('generation', as_index=False)['f_1'].mean().iloc[:,1],
                                   mean_f_2=df_pop_generations.groupby('generation', as_index=False)['f_2'].mean().iloc[:,1])
            df_data_generations.to_csv(path_results_per_run / "Data_generations.csv")
            
            json.dump(parameters_input, open(path_results_per_run / "parameters_input.json", "w")) # saves the parameters in a json file
            json.dump(parameters_constraints, open(path_results_per_run / "parameters_constraints.json", "w"))
            json.dump(parameters_GA_frequencies, open(path_results_per_run / "parameters_GA_frequencies.json", "w"))
            pickle.dump(stats, open(path_results_per_run / "stats.pickle", "ab"))
            
            with open(path_results_per_run / "Run_summary_stats.csv", "w") as archive_file:
                w = csv.writer(archive_file)
                for key, val in {**parameters_input, **parameters_constraints, **parameters_GA_frequencies, **stats}.items():
                    w.writerow([key, val])
                del key, val
            
            print("End of generations: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            
            # Visualise the generations
            if False: # becomes useless when more than 10 generations
                plot_generations_objectives(pop_generations)
                
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
                #John_validation_data = pd.read_csv("./Validation_Data/John_results_on_Mandl_2016/Results_data_headers.csv")
            
                if True:
                    '''Print Objective functions over time, all solutions and pareto set obtained'''
                    fig, axs = plt.subplots(2, 2)
                    fig.set_figheight(15)
                    fig.set_figwidth(20)
                    axs[0, 0].scatter(df_data_generations["Generation"], df_data_generations["mean_f_1"], s=1, c='r', marker="o", label='f1_AETT')
                    axs[0, 0].set_title('Mean AETT over all generations')
                    axs[0, 0].set(xlabel='Generations', ylabel='f1_AETT')
                    axs[0, 0].legend(loc="upper right")
                    
                    axs[1, 0].scatter(df_data_generations["Generation"], df_data_generations["mean_f_2"], s=1, c='b', marker="o", label='f2_TBR')
                    axs[1, 0].set_title('Mean TBR over all generations')
                    axs[1, 0].set(xlabel='Generations', ylabel='f2_TBR')
                    axs[1, 0].legend(loc="upper right") 
                    
                    axs[0, 1].scatter(df_data_generations["Generation"], df_data_generations["HV"], s=1, c='r', marker="o", label='HV obtained')
                    #axs[0, 1].scatter(range(len(df_SA_analysis)), np.ones(len(df_SA_analysis))*gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], UTFSP_problem_1.max_objs, UTFSP_problem_1.min_objs),\
                    #   s=1, c='g', marker="o", label='HV Mumford (2013)')
                    axs[0, 1].set_title('HV over all generations')
                    axs[0, 1].set(xlabel='Generations', ylabel='%')
                    axs[0, 1].legend(loc="upper right")
                    
                    #axs[1, 1].scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=10, c='b', marker="o", label='Initial route sets')
                    axs[1, 1].scatter(df_non_dominated_set["f_2"], df_non_dominated_set["f_1"], s=10, c='r', marker="o", label='Non-dom set obtained')
                    #axs[1, 1].scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="o", label='Mumford results (2013)')
                    #axs[1, 1].scatter(John_validation_data.iloc[:,1], John_validation_data.iloc[:,0], s=10, c='b', marker="o", label='John results (2016)')
                    axs[1, 1].set_title('Non-dominated set obtained vs benchmark results')
                    axs[1, 1].set(xlabel='f2_TBR', ylabel='f1_AETT')
                    axs[1, 1].legend(loc="upper right")
                    
                    manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()
                    plt.show()
                    plt.savefig(path_results_per_run / "Results_summary.pdf", bbox_inches='tight')
                
                    manager.window.close()
                    
    del i_generation, pop_size, survivor_indices

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
        #stats_overall['HV initial set'] = gf.norm_and_calc_2d_hv(df_routes_R_initial_set.iloc[:,0:2], UTFSP_problem_1.max_objs, UTFSP_problem_1.min_objs)
        stats_overall['HV obtained'] = gf.norm_and_calc_2d_hv(df_overall_pareto_set.iloc[:,0:2], UTFSP_problem_1.max_objs, UTFSP_problem_1.min_objs)
        #stats_overall['HV Benchmark'] = gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], UTFSP_problem_1.max_objs, UTFSP_problem_1.min_objs)
        
        df_durations.loc[len(df_durations)] = ["Average", df_durations["Duration"].mean()]
        df_durations.to_csv(path_results / "Run_durations.csv")
        del df_durations
        
        with open(path_results / "Stats_overall.csv", "w") as archive_file:
            w = csv.writer(archive_file)
            for key, val in {**stats_overall,
                             **UTFSP_problem_1.problem_inputs.__dict__, 
                             **UTFSP_problem_1.problem_constraints.__dict__, 
                             **UTFSP_problem_1.problem_GA_parameters.__dict__}.items():
                w.writerow([key, val])
            del key, val
            
        ga.get_sens_tests_stats_from_model_runs(path_results, parameters_GA_frequencies["number_of_runs"]) # prints the runs summary
        # ga.get_sens_tests_stats_from_UTFSP_GA_runs(path_results)            
        
        # %% Plot analysis graph
        '''Plot the analysis graph'''
        if True:
            fig, axs = plt.subplots(1,1)
            fig.set_figheight(15)
            fig.set_figwidth(20)
            
            # axs.scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=20, c='b', marker="o", label='Initial route sets')
            axs.scatter(df_overall_pareto_set["f_2"], df_overall_pareto_set["f_1"], s=10, c='r', marker="o", label='Pareto front obtained from all runs')
            #axs.scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="x", label='Mumford results (2013)')
            #axs.scatter(John_validation_data.iloc[:,1], John_validation_data.iloc[:,0], s=10, c='b', marker="o", label='John results (2016)')
            axs.set_title('Pareto front obtained vs Mumford Results')
            axs.set(xlabel='f2_TBR', ylabel='f1_AETT')
            axs.legend(loc="upper right")
            del axs
            
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()
            plt.savefig(path_results / "Results_combined.pdf", bbox_inches='tight')
            manager.window.close()
            del fig, manager
        del archive_file, df_non_dominated_set, path_parent_folder, path_results, path_results_per_run, pop_generations, w
        
    del run_nr

# %% Sensitivity analysis
''' Sensitivity analysis tests'''

# Single Thread
if __name__ == "__main__":
    
    if Choice_conduct_sensitivity_analysis:
        start = time.perf_counter()
        ''' Create copies of the original input data '''
        #original_UTFSP_problem_1 = copy.deepcopy(UTFSP_problem_1)    
        #original_parameters_constraints = copy.deepcopy(parameters_constraints)  
        #original_parameters_GA_frequencies = copy.deepcopy(parameters_GA_frequencies)  
        
        # Set up the list of parameters to test
        sensitivity_list = [[parameters_GA_frequencies, "population_size", 10, 20, 50, 100, 150, 200, 300],
                            [parameters_GA_frequencies, "generations", 5, 10, 15, 20, 25, 50],
                            [parameters_GA_frequencies, "crossover_probability", 0.7, 0.8, 0.9, 0.95, 1], # bottom two takes WAY longer, subdivide better
                            [parameters_GA_frequencies, "mutation_probability", 0.05, 0.1, 1/parameters_constraints["con_r"], 0.2, 0.3, 0.5]
                            ]
        
        # Set up the list of parameters to test
        sensitivity_list = [#[parameters_GA_frequencies, "population_size", 10, 20, 50, 100, 150],
                            #[parameters_GA_frequencies, "population_size", 200],
                            #[parameters_GA_frequencies, "population_size", 300],
                            #[parameters_GA_frequencies, "generations", 60],
                            #[parameters_GA_frequencies, "crossover_probability", 0.7, 0.8, 0.9],
                            #[parameters_GA_frequencies, "crossover_probability", 0.95, 1],
                            #[parameters_GA_frequencies, "mutation_probability", 0.05, 0.1],
                            #[parameters_GA_frequencies, "mutation_probability", 1/parameters_constraints["con_r"], 0.2, 0.3, 0.5]
                            
                            [parameters_GA_frequencies, "mutation_probability", 0.1],
                            ]
        

        
        for sensitivity_test in sensitivity_list:
            parameter_dict = sensitivity_test[0]
            dict_entry = sensitivity_test[1]
            for test_counter in range(2,len(sensitivity_test)):
                
                print("Test: {0} = {1}".format(sensitivity_test[1], sensitivity_test[test_counter]))
                
                UTFSP_problem_1.add_text = f"{sensitivity_test[1]}_{round(sensitivity_test[test_counter],2)}"
                
                temp_storage = parameter_dict[dict_entry]
                
                # Set new parameters
                parameter_dict[dict_entry] = sensitivity_test[test_counter]
    
                # Update problem instance
                UTFSP_problem_1.problem_constraints = gc.Problem_inputs(parameters_constraints)
                UTFSP_problem_1.problem_GA_parameters = gc.Problem_GA_inputs(parameters_GA_frequencies)
                
                # Run model
                main(UTFSP_problem_1)
                
                # Reset the original parameters
                parameter_dict[dict_entry] = temp_storage
        
        finish = time.perf_counter()
        
        print(f'Finished in {round(finish-start, 6)} second(s)')
        




# %% TESTS: 

if False:  
    
    def fn_obj(frequencies, UTFSP_problem_input):
        return (gf2.f3_ETT(UTFSP_problem_input.R_routes.routes,
                           frequencies, 
                           UTFSP_problem_input.problem_data.mx_dist, 
                           UTFSP_problem_input.problem_data.mx_demand, 
                           UTFSP_problem_input.problem_inputs.__dict__), #f3_ETT
                gf2.f4_TBR(UTFSP_problem_input.R_routes.routes, 
                           frequencies, 
                           UTFSP_problem_input.problem_data.mx_dist)) #f4_TBR
    
    def fn_obj_row(frequencies):
        
        return (gf2.f3_ETT(UTFSP_problem_1.R_routes.routes,
                           frequencies, 
                           UTFSP_problem_1.problem_data.mx_dist, 
                           UTFSP_problem_1.problem_data.mx_demand, 
                           UTFSP_problem_1.problem_inputs.__dict__), #f3_ETT
                gf2.f4_TBR(UTFSP_problem_1.R_routes.routes, 
                           frequencies, 
                           UTFSP_problem_1.problem_data.mx_dist)) #f4_TBR
        
    #pop_1 = PopulationFreq(UTFSP_problem_1)   
    #pop_1.generate_initial_population(UTFSP_problem_1, fn_obj) 
    #pop_1.variables
    
    offspring_variables = np.array([[0.08333333, 0.2       , 0.05555556, 0.16666667, 0.04      ,
    0.08333333],
   [0.08333333, 0.08333333, 0.0625    , 0.08333333, 0.1       ,
    0.2       ],
   [0.1       , 0.16666667, 0.05555556, 0.16666667, 0.04      ,
    0.07142857],
   [0.0625    , 0.05      , 0.04      , 0.1       , 0.03333333,
    0.05      ],
   [0.14285714, 0.16666667, 0.16666667, 0.14285714, 0.16666667,
    0.0625    ],
   [0.05      , 0.07142857, 0.1       , 0.03333333, 0.08333333,
    0.11111111],
   [0.07142857, 0.16666667, 0.2       , 0.14285714, 0.1       ,
    0.16666667],
   [0.14285714, 0.05      , 0.05      , 0.1       , 0.11111111,
    0.14285714],
   [0.08333333, 0.04      , 0.08333333, 0.11111111, 0.16666667,
    0.2       ],
   [0.14285714, 0.05555556, 0.2       , 0.08333333, 0.05      ,
    0.07142857]])
    
    
    start = time.perf_counter()
    offspring_objectives = np.apply_along_axis(fn_obj_row, 1, offspring_variables)
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s): apply_along_axis')

    start = time.perf_counter()
    offspring_objectives = gf2.calc_fn_obj_for_np_array(fn_obj_row, offspring_variables) # takses about 6 seconds for one evaluation   
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s): for loop')

A_link_volumes = TN.mx_volumes_links
A_mx_C_a = TN.mx_C_a
# fn_obj(np.full((1,UTFSP_problem_1.problem_constraints.con_r), 1/10)[0],UTFSP_problem_1)
#TN.R_routes
#gv.plotUTFSPAndSavePDF(mx_dist,routes_R, mx_coords, name)


"""
self.R_routes_named = R_routes_named
self.names_of_transit_routes = names_of_transit_routes
self.names_all_transit_nodes = names_all_transit_nodes
self.n_transit_nodes = n_transit_nodes
self.dict_all_nodes = dict_all_nodes
self.mx_transit_network = mx_transit_network
self.mx_C_a = mx_C_a
self.mx_f_a = mx_f_a
"""

#%% Tests for f_3

R_routes_list = R_routes.routes
#R_routes, F_x, mx_dist, mx_demand, parameters_input
'''Create the transit network'''
R_routes_named = gf2.format_routes_with_letters(R_routes_list)
names_of_transit_routes = gf2.get_names_of_routes_as_list(R_routes_named)
names_all_transit_nodes = list(map(str, range(parameters_input['n'])))+names_of_transit_routes
n_transit_nodes = len(names_all_transit_nodes)

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
                """Test for weird results"""
                mx_transit_network[i,j] = 0
                mx_C_a[i,j] = parameters_input['large_dist']
                mx_f_a[i,j] = 0
            # else:
            #     mx_transit_network[i,j] = 1
            #     mx_C_a[i,j] = parameters_input['walkFactor']*mx_dist[i,j]
            #     mx_f_a[i,j] = inf


''''Fill in the boarding links characteristics'''
counter = 0

for i in range(len(R_routes_list)): 
    for j in range(len(R_routes_list[i])):       
        i_index =  int(re.findall(r'\d+', names_of_transit_routes[counter])[0]) # number of the transit node
        j_index = names_all_transit_nodes.index(names_of_transit_routes[counter]) # position of the transit node in network graph

        mx_transit_network[i_index, j_index] = 1   
        mx_C_a[i_index, j_index] = parameters_input['boardingTime'] # sets the boarding
        mx_f_a[i_index, j_index] = F_x[i] # set the frequencies per transit line
  
        counter = counter + 1  

'''Fill in the alighting links characteristics'''
counter = 0

for i in range(len(R_routes_list)): 
    for j in range(len(R_routes_list[i])):       
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
                (gf2.f_i_u_i_test(f_i,u_i, parameters_input['alpha_const_inter']) + f_a*(min_u_j_and_c_a))/(f_i+f_a)
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
    
# end the overall destination change for loop spanning from 6.)

  
'''Add the volume per arc details to the list_transit_links object'''
df_transit_links.insert(4, "v_a", np.zeros(len(df_transit_links)))

for i in range(len(df_transit_links)):
    df_transit_links.iloc[i,4] = mx_volumes_links[int(dict_all_nodes[df_transit_links.iloc[i,0]]),\
                         int(dict_all_nodes[df_transit_links.iloc[i,1]])]

# F3 Expected Travel Time
print(sum(sum(mx_volumes_links*mx_C_a))/(parameters_input['total_demand']*2))


# %% Transit network tests
B_df_opt_strat_alg = TN.df_opt_strat_alg
B_df_opt_strat_alg_named = TN.df_opt_strat_alg_named
B_mx_A_bar_strategy_lines = TN.mx_A_bar_strategy_lines
B_df_A_bar_strategy_lines = TN.df_A_bar_strategy_lines

B_df_opt_strat_alg[0].iloc[:,0]

u_i_times = np.zeros(shape=(15,15))

for destination_i_strategy, destination_i in zip(B_df_opt_strat_alg, range(15)):
    for origin_j in range(15):

        if origin_j != destination_i:
            #print(destination_i_strategy[(destination_i_strategy.iloc[:,0] == origin_j) & destination_i_strategy.iloc[:,4]].iloc[0,3])
            u_i_times[origin_j,destination_i] = destination_i_strategy[(destination_i_strategy.iloc[:,0] == origin_j) & destination_i_strategy.iloc[:,4]].iloc[0,3]
        #else:
            #print(0)
            
print(sum(sum(mx_demand*u_i_times))/(parameters_input['total_demand']*2))
