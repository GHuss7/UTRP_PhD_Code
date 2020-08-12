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

# %% Import personal functions
import DSS_Admin as ga
import DSS_UTNDP_Functions as gf
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import EvaluateRouteSet as ev

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


    
# %% Load the respective files
name_input_data = "Mandl_Data"      # set the name of the input data
mx_dist, mx_demand, mx_coords = gf.read_problem_data_to_matrices(name_input_data)
del name_input_data
# %% Set input parameters
    
'''State the various parameter constraints''' 
parameters_constraints = {
'con_r' : 6,               # number of allowed routes (aim for > [numNodes N ]/[maxNodes in route])
'con_minNodes' : 3,                        # minimum nodes in a route
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
'ref_point_max_f1_ATT' : 15.1304, # max f1_ATT for the Hypervolume calculations
'ref_point_min_f1_ATT' : 10.3301, # min f1_ATT for the Hypervolume calculations
'ref_point_max_f2_TRT' : 224, # max f2_TRT for the Hypervolume calculations
'ref_point_min_f2_TRT' : 63, # min f2_TRT for the Hypervolume calculations
'walkFactor' : 3, # factor it takes longer to walk than to drive
'boardingTime' : 0.1, # assume boarding and alighting time = 6 seconds
'alightingTime' : 0.1, # problem when alighting time = 0 (good test 0.5)(0.1 also works)
'large_dist' : int(mx_dist.max()), # the large number from the distance matrix
'alpha_const_inter' : 0.5 # constant for interarrival times relationship (Spiess 1989)
}

'''State the various GA input parameters for frequency setting''' 
parameters_GA_frequencies={
"population_size" : 200, #should be an even number
"offspring_population_size" : 200,
"generations" : 20,
"crossover_probability" : 0.9,  # John: 0.9
"crossover_distribution_index" : 5,
"mutation_probability" : 1/6, # John: 1/|Route set| -> set later
"mutation_distribution_index" : 10,
"tournament_size" : 2,
"termination_criterion" : "StoppingByEvaluations",
"max_evaluations" : 25000,
"number_of_variables" : "not_set",
"number_of_objectives" : 2 # this could still be automated in the future
}

stats_overall = {
'execution_start_time' : datetime.datetime.now()} # enter the begin time

stats = {} # define the stats dictionary

'''Set the reference point for the Hypervolume calculations'''
#max_objs = np.array([parameters_input['ref_point_max_f1_ATT'],parameters_input['ref_point_max_f2_TRT']])
#min_objs = np.array([parameters_input['ref_point_min_f1_ATT'],parameters_input['ref_point_min_f2_TRT']])

#%% Input parameter tests

'''Test the inputs for feasibility'''
# Test feasibility if there are enough buses to cover each route once
if parameters_constraints["con_r"] > parameters_constraints["con_fleet_size"]:
    print("Warning: Number of available vehicles are less than the number of routes.\n"\
          "Number of routes allowed set to "+ str(parameters_constraints["con_r"]))
    parameters_constraints["con_r"] = parameters_constraints["con_fleet_size"]

# %% Import the route set to be evaluated
''' Import the route set '''
R_x = gf.convert_routes_str2list("13-14-10-8-15*4-2-3-6*13-11-12*7-15-9*5-4-6-15*3-2-1*")
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

#%% Define the Transit network
TN = gf2.Transit_network(R_x, F_x, mx_dist, mx_demand, parameters_input) 

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

# %% Class: IndividualFreq
class IndividualFreq:
    """A class for storing an individual, variables and fitness"""
    def __init__(self, X=None, F=None, CV=None, G=None, feasible=None, **kwargs) -> None:
        self.X = X # variables
        self.F = F # objective function values
        self.CV = CV #
        self.G = G #
        self.feasible = feasible # feasibility (True / False)
        self.data = kwargs # any other relevant data?
        self.attr = set(self.__dict__.keys())
        
    def has(self, key):
        return key in self.attr or key in self.data

    def set(self, key, value):
        if key in self.attr:
            self.__dict__[key] = value
        else:
            self.data[key] = value

    def copy(self):
        ind = copy.copy(self)
        ind.data = self.data.copy()
        return ind

    def get(self, keys):
        if keys in self.data:
            return self.data[keys]
        elif keys in self.attr:
            return self.__dict__[keys]
        else:
            return None
            

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


#%% Class: PopFreq_2
class PopulationFreq_2(list):
    """A class for storing the population consisting of individuals"""
    pass

    
    def generate_initial_population(self, main_problem, fn_obj):
        for i in range(self.population_size):
            self.variables[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_frequencies()
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

#%% Class: InitialisationFreq        
class InitialisationFreq(gf2.Frequencies):      
    """A class used to initialise the population with individuals"""
    def __init__(self, main_problem, pop_size) -> None:
        super(gf2.Frequencies, self).__init__()       
    
    def generate_random_individual(self, main_problem, fn_obj):
        self.X = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_frequencies()
        self.F = fn_obj(self.X, main_problem)
    

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


#%% Class: RankAndCrowdingSurvival

class RankAndCrowdingSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=True)

    def _do(self, problem, pop, n_survive, D=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(np.float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


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


# %% Class:TournamentSelection
    
from pymoo.model.selection import Selection
from pymoo.util.misc import random_permuations

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


class TournamentSelection_g2(Selection):
    """
      The Tournament selection is used to simulated a tournament between individuals. The pressure balances
      greedy the genetic algorithm will be.
    """

    def __init__(self, func_comp=None, pressure=2):
        """

        Parameters
        ----------
        func_comp: func
            The function to compare two individuals. It has the shape: comp(pop, indices) and returns the winner.
            If the function is None it is assumed the population is sorted by a criterium and only indices are compared.

        pressure: int
            The selection pressure to bie applied. Default it is a binary tournament.
        """

        # selection pressure to be applied
        self.pressure = pressure

        self.f_comp = func_comp
        if self.f_comp is None:
            raise Exception("Please provide the comparing function for the tournament selection!")

    def _do(self, pop, n_select, n_parents=1, **kwargs):
        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = random_permuations(n_perms, len(pop))[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        # compare using tournament function
        S = self.f_comp(pop, P, **kwargs)

        return np.reshape(S, (n_select, n_parents))


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


#%% Create in PYMOO form
def convert_pop_g2_to_pop_pymoo(pop_g2):
    x = pop_from_array_or_individual(pop_g2.variables)
    
    # Create object in the standardised form
    for i in range(len(x)):
        x[i].set("F", pop_g2.variables)
        
        # Examples:
        # x.get("F")
        # x.set("Rank", pop_1.rank)
        # x.get("Rank")
        # x.F = x.get("F")
        
        return x
    
    
#%% Functions: Crossover

def crossover_uniform_as_is(parent_A, parent_B, parent_length):
    x_index = random.randint(1,parent_length-1)
    child_A = np.hstack((parent_A[0:x_index], parent_B[x_index:]))
    child_B = np.hstack((parent_B[0:x_index], parent_A[x_index:]))
    
    return child_A, child_B
    
    
def crossover_pop_uniform_as_is(pop, main_problem):
    selection = tournament_selection_g2(pop, n_select=int(main_problem.problem_GA_parameters.offspring_population_size/2))
    
    offspring_variable_args = np.empty([main_problem.problem_GA_parameters.offspring_population_size,
                                   main_problem.R_routes.number_of_routes]).astype(int)
    
    for i in range(0,int(main_problem.problem_GA_parameters.offspring_population_size/2)):
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


#%% Function: Add/Delete individuals to/from population
#TODO: dont recompute old solutions' obj values -- once per solution
def combine_offspring_with_pop(pop, offspring_variable_args):
    pop.variable_args = np.vstack([pop.variable_args, offspring_variable_args])
    # Filter out duplicates
    is_unique = np.where(np.logical_not(find_duplicates(pop.variable_args, epsilon=1e-24)))[0]
    pop.variable_args = pop.variable_args[is_unique]
    # Update the rest accordingly
    pop.variables = 1/gf2.Frequencies.theta_set[pop.variable_args]
    pop.objectives = np.apply_along_axis(fn_obj_row, 1, pop.variables)   
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
        

def combine_offspring_with_pop_2(pop, offspring_variable_args):
    # Function to combine the offspring with the population
    len_pop = len(pop.objectives)
    pop.variable_args = np.vstack([pop.variable_args, offspring_variable_args])
    
    # Filter out duplicates
    is_unique = np.where(np.logical_not(find_duplicates(pop.variable_args, epsilon=1e-24)))[0]
    pop.variable_args = pop.variable_args[is_unique]
    
    # Only evaluate the offspring
    offspring_variables = 1/gf2.Frequencies.theta_set[pop.variable_args[len_pop:,]]
    offspring_objectives = np.apply_along_axis(fn_obj_row, 1, offspring_variables)   

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
        label_names.append("f"+str(i))
    for j in range(num_variables):
        label_names.append("x"+str(j))
    return label_names

#%% GA Implementation


# Create the initial populations   
stats['begin_time'] = datetime.datetime.now() # enter the begin time
run_nr = 0
print("Generation 0 initiated" + " ("+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+")")
pop_1 = PopulationFreq(UTFSP_problem_1)   
pop_1.generate_initial_population(UTFSP_problem_1, fn_obj) 
pop_generations = np.hstack([pop_1.objectives, np.full((len(pop_1.objectives),1),0)])

data_for_analysis = np.hstack([pop_1.objectives, pop_1.variables]) # create an object to contain all the data for analysis

stats['end_time'] = datetime.datetime.now() # enter the begin time
print("Generation 0 duration: "+ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']))


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
    combine_offspring_with_pop_2(pop_1, mutated_variable_args)
    
    pop_size = UTFSP_problem_1.problem_GA_parameters.population_size
    data_for_analysis = np.vstack([data_for_analysis, np.hstack([pop_1.objectives[pop_size:,], pop_1.variables[pop_size:,]])])
      
    # Get new generation
    survivor_indices = get_survivors(pop_1, pop_size)
    keep_individuals(pop_1, survivor_indices)

    pop_generations = np.vstack([pop_generations, np.hstack([pop_1.objectives, np.full((len(pop_1.objectives),1),i_generation+1)])]) # add the population to the generations
    
    stats['end_time_run'] = datetime.datetime.now() # save the end time of the run
    print("Generation " + str(int(i_generation+1)) +" duration: "+ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']))
    
    
#%% Stats updates
stats['end_time'] = datetime.datetime.now() # save the end time of the run
stats['full_duration'] = stats['end_time'] - stats['begin_time'] # calculate and save the duration of the run
stats['begin_time'] = stats['begin_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
stats['end_time'] =  stats['end_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format


#%% Save the results
'''Write all results and parameters to files'''
'''Main folder path'''
path_parent_folder = Path(os.path.dirname(os.getcwd()))
path_results = path_parent_folder / ("Results/Results_"+parameters_input['Problem_name']+"/"+parameters_input['Problem_name']+"_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S"))

'''Sub folder path'''
path_results_per_run = path_results / ("Run_"+str(run_nr+1))
if not path_results_per_run.exists():
    os.makedirs(path_results_per_run)

# Create and save the dataframe 
df_pop_generations = create_df_from_pop(pop_generations)
df_non_dominated_set = df_pop_generations.iloc[-UTFSP_problem_1.problem_GA_parameters.population_size:,0:2]
df_non_dominated_set = df_non_dominated_set.sort_values(by='f_1', ascending=True) # sort

df_data_for_analysis = pd.DataFrame(data=data_for_analysis,columns=generate_data_analysis_labels(2,6)) #TODO: automate the 2,6

df_pop_generations.to_csv(path_results_per_run / "Pop_generations.csv")
df_non_dominated_set.to_csv(path_results_per_run / "Non_dominated_set.csv")
df_data_for_analysis.to_csv(path_results_per_run / "Data_for_analysis.csv")

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
plot_generations_objectives(pop_generations)

# TODO: add Hypervolume measurement and print that for each generation as the code executes