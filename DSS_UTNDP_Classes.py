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
from datetime import timedelta, datetime
#import pygmo as pg
from math import inf
import matplotlib.pyplot as plt

import DSS_UTNDP_Functions as gf
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import DSS_Admin as ga

from pymoo.util.function_loader import load_function
from pymoo.model.survival import Survival
from pymoo.model.selection import Selection
from pymoo.util.misc import random_permuations

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
    
    def replace_route_with_random_feasible_route(self, UTNDP_problem_input):
        """Generate feasible route based on appending random shortest paths"""
        self.routes = gf.generate_initial_feasible_route_set(UTNDP_problem_input.problem_data.mx_dist, 
                                                          UTNDP_problem_input.problem_constraints.__dict__)


    def return_feasible_route(UTNDP_problem_input):
        """Generate feasible route based on appending random shortest paths"""
        return gf.generate_initial_feasible_route_set(UTNDP_problem_input.problem_data.mx_dist, 
                                                      UTNDP_problem_input.problem_constraints.__dict__)
    
    def return_feasible_route_robust(UTNDP_problem_input):
        """Generate feasible route based on appending random shortest paths"""
        for try_number in range(1000):
            '''Create the transit network graph'''
            g_tn = gf.create_igraph_from_dist_mx(UTNDP_problem_input.problem_data.mx_dist)
    
            paths_shortest_all = gf.get_all_shortest_paths(g_tn) # Generate all the shortest paths
    
            """Remove duplicate lists in reverse order""" #can be added for efficiency, but not that neccessary
            paths_shortest_all = gf.remove_half_duplicate_routes(paths_shortest_all)
            
            # Shorten the candidate routes according to the constraints
            for i in range(len(paths_shortest_all)-1, -1, -1):
                if len(paths_shortest_all[i]) < UTNDP_problem_input.problem_constraints.con_minNodes or len(paths_shortest_all[i]) > UTNDP_problem_input.problem_constraints.con_maxNodes:  
                    del paths_shortest_all[i]
            
            initial_route_set = gf.routes_generation_unseen_prob(paths_shortest_all, paths_shortest_all, UTNDP_problem_input.problem_constraints.con_r)
            
            if gf.test_all_four_constraints(initial_route_set, UTNDP_problem_input):
                return initial_route_set
                
            else:
                routes_R = gf.repair_add_missing_from_terminal_multiple(initial_route_set, UTNDP_problem_input)
                if gf.test_all_four_constraints(routes_R, UTNDP_problem_input):
                    return routes_R
                
        return False

                
    def return_feasible_route_robust_k_shortest(UTNDP_problem_input):
        """Generate feasible route based on appending random shortest paths"""
        for try_number in range(1000):   
            
            k_short_paths = copy.deepcopy(UTNDP_problem_input.k_short_paths.paths)
            
            # Shorten the candidate routes according to the constraints
            for i in range(len(k_short_paths)-1, -1, -1):
                if len(k_short_paths[i]) < UTNDP_problem_input.problem_constraints.con_minNodes or len(k_short_paths[i]) > UTNDP_problem_input.problem_constraints.con_maxNodes:  
                    del k_short_paths[i]
            
            initial_route_set = gf.routes_generation_unseen_prob(k_short_paths, k_short_paths, UTNDP_problem_input.problem_constraints.con_r)
            
            if gf.test_all_four_constraints(initial_route_set, UTNDP_problem_input):
                return initial_route_set
                
            else:
                routes_R = gf.repair_add_missing_from_terminal_multiple(initial_route_set, UTNDP_problem_input)
                if gf.test_all_four_constraints(routes_R, UTNDP_problem_input):
                    return routes_R
        
        return False
    
    def return_feasible_route_robust_k_shortest_probabilistic(UTNDP_problem_input):
        """Generate feasible route based on appending random shortest paths"""
        for try_number in range(1000):   
            
            k_short_paths = copy.deepcopy(UTNDP_problem_input.k_short_paths.paths)
            
            # Shorten the candidate routes according to the constraints
            for i in range(len(k_short_paths)-1, -1, -1):
                if len(k_short_paths[i]) < UTNDP_problem_input.problem_constraints.con_minNodes or len(k_short_paths[i]) > UTNDP_problem_input.problem_constraints.con_maxNodes:  
                    del k_short_paths[i]
            
            initial_route_set = gf.routes_generation_unseen_probabilistic(k_short_paths, k_short_paths, UTNDP_problem_input.problem_constraints.con_r)
            
            if gf.test_all_four_constraints(initial_route_set, UTNDP_problem_input):
                return initial_route_set
                
            else:
                routes_R = gf.repair_add_missing_from_terminal_multiple(initial_route_set, UTNDP_problem_input)
                if gf.test_all_four_constraints(routes_R, UTNDP_problem_input):
                    return routes_R
        
        return False
         
    def return_feasible_route_set_greedy_demand(UTNDP_problem_input):
        """Generate feasible route based on appending random shortest paths"""
        return gf.generate_feasible_route_set_greedy_demand(UTNDP_problem_input)
    
    def plot_routes(self, main_problem):
        """A function that plots the routes of a problem based on the problem defined"""
        gv.plotRouteSet2(main_problem.problem_data.mx_dist, self.routes, main_problem.problem_data.mx_coords) # using iGraph
     
    def plot_routes_no_coords(self, main_problem,layout_style="kk"):
        """A function that plots the routes of a problem based on the problem defined, where no coords are defined"""
        gv.plotRouteSet(main_problem.problem_data.mx_dist, self.routes,layout_style) # using iGraph
        
    def to_str(self):
        """A function that returns the string representation of a route set"""
        return gf.convert_routes_list2str(self.routes)
     

# %% Class: PopulationRoutes
class PopulationRoutes(Routes):
    """A class for storing the population consisting of arrays and lists"""
    def __init__(self, main_problem):
        super(Routes, self).__init__()
        self.population_size = main_problem.problem_GA_parameters.population_size
        #self.variable_args = np.empty([main_problem.problem_GA_parameters.population_size,
        #                           main_problem.problem_GA_parameters.number_of_variables]).astype(int)
        self.variables = [None] * main_problem.problem_GA_parameters.population_size
        self.variables_str = [None] * main_problem.problem_GA_parameters.population_size
        self.objectives = np.empty([main_problem.problem_GA_parameters.population_size,
                                   main_problem.problem_GA_parameters.number_of_objectives])
        self.rank = np.empty([main_problem.problem_GA_parameters.population_size,
                                   1])
        self.crowding_dist = np.empty([main_problem.problem_GA_parameters.population_size,
                                   1])
        
    
    def generate_initial_population(self, main_problem, fn_obj):
        t_now = datetime.now() # TIMING FUNCTION
        average_at = 5 # TIMING FUNCTION
        
        for i in range(self.population_size):
            #self.variable_args[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_args()
            self.variables[i] = Routes.return_feasible_route(main_problem)
            self.variables_str[i] = gf.convert_routes_list2str(self.variables[i])
            self.objectives[i,] = fn_obj(self.variables[i], main_problem)
 
            if i == average_at-1 or i == 9 or i == self.population_size-1: # TIMING FUNCTION
                tot_iter = ga.determine_total_iterations(main_problem, 1)
                sec_per_iter_time_delta = datetime.now() - t_now
                ga.time_projection((sec_per_iter_time_delta.seconds)/(i+1), tot_iter, t_now=t_now, print_iter_info=True) # prints the time projection of the algorithm
   
            # get the objective space values and objects
            # F = pop.get("F").astype(np.float, copy=False)
        F = self.objectives
        
            # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F)

        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = gf.calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.rank[i] = k
                self.crowding_dist[i] = crowding_of_front[j]
                
    def generate_initial_population_robust(self, main_problem, fn_obj):
        t_now = datetime.now() # TIMING FUNCTION
        average_at = 5 # TIMING FUNCTION
        
        for i in range(self.population_size):
            #self.variable_args[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_args()
            self.variables[i] = Routes.return_feasible_route_robust(main_problem)
            self.variables_str[i] = gf.convert_routes_list2str(self.variables[i])
            self.objectives[i,] = fn_obj(self.variables[i], main_problem)
 
            if i == average_at-1 or i == 9 or i == self.population_size-1: # TIMING FUNCTION
                tot_iter = ga.determine_total_iterations(main_problem, 1)
                sec_per_iter_time_delta = datetime.now() - t_now
                ga.time_projection((sec_per_iter_time_delta.seconds)/(i+1), tot_iter, t_now=t_now, print_iter_info=True) # prints the time projection of the algorithm

            # get the objective space values and objects
            # F = pop.get("F").astype(np.float, copy=False)
        F = self.objectives
        
            # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F)

        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = gf.calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.rank[i] = k
                self.crowding_dist[i] = crowding_of_front[j]
                
    def generate_initial_population_robust_ksp(self, main_problem, fn_obj):
        t_now = datetime.now() # TIMING FUNCTION
        st = [] # List for starting times
        ft = [] # List for finishing times
        average_at = 5 # TIMING FUNCTION
        
        for i in range(self.population_size):
            #self.variable_args[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_args()
            
            # Create a feasible route set
            self.variables[i] = Routes.return_feasible_route_robust_k_shortest(main_problem)
            self.variables_str[i] = gf.convert_routes_list2str(self.variables[i])
            
            # Determine the objective function values
            start_time = datetime.now() # TIMING FUNCTION
            self.objectives[i,] = fn_obj(self.variables[i], main_problem)
            end_time = datetime.now() # TIMING FUNCTION
            st.append(start_time) # TIMING FUNCTION
            ft.append(end_time) # TIMING FUNCTION
    
            # Determine and print projections
            if i == average_at-1 or i == self.population_size-1: # TIMING FUNCTION
                diffs = [x-y for x,y in zip(ft,st)]
                diffs_sec = [float(str(x.seconds)+"."+str(x.microseconds)) for x in diffs]
                avg_time = np.average(np.asarray(diffs_sec))
                tot_time = np.sum(np.asarray(diffs_sec))
                tot_iter = ga.determine_total_iterations(main_problem, 1)
                ga.time_projection(avg_time, tot_iter, t_now=t_now, print_iter_info=True) # prints the time projection of the algorithm

            # get the objective space values and objects
            # F = pop.get("F").astype(np.float, copy=False)
        F = self.objectives
        
            # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F)

        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = gf.calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.rank[i] = k
                self.crowding_dist[i] = crowding_of_front[j]
                
    def generate_initial_population_greedy_demand(self, main_problem, fn_obj):
        t_now = datetime.now() # TIMING FUNCTION
        st = [] # List for starting times
        ft = [] # List for finishing times
        average_at = 5 # TIMING FUNCTION
        
        for i in range(self.population_size):
            #self.variable_args[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_args()
            
            # Create a feasible route set
            self.variables[i] = Routes.return_feasible_route_set_greedy_demand(main_problem)
            self.variables_str[i] = gf.convert_routes_list2str(self.variables[i])
            
            # Determine the objective function values
            start_time = datetime.now() # TIMING FUNCTION
            self.objectives[i,] = fn_obj(self.variables[i], main_problem)
            end_time = datetime.now() # TIMING FUNCTION
            st.append(start_time) # TIMING FUNCTION
            ft.append(end_time) # TIMING FUNCTION
    
            # Determine and print projections
            if i == average_at-1 or i == self.population_size-1: # TIMING FUNCTION
                diffs = [x-y for x,y in zip(ft,st)]
                diffs_sec = [float(str(x.seconds)+"."+str(x.microseconds)) for x in diffs]
                avg_time = np.average(np.asarray(diffs_sec))
                tot_time = np.sum(np.asarray(diffs_sec))
                tot_iter = ga.determine_total_iterations(main_problem, 1)
                ga.time_projection(avg_time, tot_iter, t_now=t_now, print_iter_info=True) # prints the time projection of the algorithm

            # get the objective space values and objects
            # F = pop.get("F").astype(np.float, copy=False)
        F = self.objectives
        
            # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F)

        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = gf.calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.rank[i] = k
                self.crowding_dist[i] = crowding_of_front[j]
      
    def generate_initial_population_hybrid(self, main_problem, fn_obj):
        t_now = datetime.now() # TIMING FUNCTION
        st = [] # List for starting times
        ft = [] # List for finishing times
        average_at = 5 # TIMING FUNCTION
        
        sol_gen_funcs = [Routes.return_feasible_route_robust_k_shortest,
                        Routes.return_feasible_route_set_greedy_demand]
        
        div = self.population_size // len(sol_gen_funcs)
             
        for i in range(self.population_size):
            #self.variable_args[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_args()
            
            # Create a feasible route set
            sol_gen_func = sol_gen_funcs[i//div]
            self.variables[i] = sol_gen_func(main_problem)
            self.variables_str[i] = gf.convert_routes_list2str(self.variables[i])
            
            # Determine the objective function values
            start_time = datetime.now() # TIMING FUNCTION
            self.objectives[i,] = fn_obj(self.variables[i], main_problem)
            end_time = datetime.now() # TIMING FUNCTION
            st.append(start_time) # TIMING FUNCTION
            ft.append(end_time) # TIMING FUNCTION
    
            # Determine and print projections
            if i == average_at-1 or i == self.population_size-1: # TIMING FUNCTION
                diffs = [x-y for x,y in zip(ft,st)]
                diffs_sec = [float(str(x.seconds)+"."+str(x.microseconds)) for x in diffs]
                avg_time = np.average(np.asarray(diffs_sec))
                tot_time = np.sum(np.asarray(diffs_sec))
                tot_iter = ga.determine_total_iterations(main_problem, 1)
                ga.time_projection(avg_time, tot_iter, t_now=t_now, print_iter_info=True) # prints the time projection of the algorithm

            # get the objective space values and objects
            # F = pop.get("F").astype(np.float, copy=False)
        F = self.objectives
        
            # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F)

        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = gf.calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.rank[i] = k
                self.crowding_dist[i] = crowding_of_front[j]
    
    def generate_initial_population_smart(self, main_problem, fn_obj, route_gen_func=Routes.return_feasible_route_robust_k_shortest, new_pop_size=False):
        t_now = datetime.now() # TIMING FUNCTION
        st = [] # List for starting times
        ft = [] # List for finishing times
        average_at = 5 # TIMING FUNCTION
        
        if new_pop_size: 
            # Reinitiate the population to make provision for larger population
            pop_size = new_pop_size
            self.population_size = pop_size
            self.variables = [None] * pop_size
            self.variables_str = [None] * pop_size
            self.objectives = np.empty([pop_size,
                                       main_problem.problem_GA_parameters.number_of_objectives])
            self.rank = np.empty([pop_size,
                                       1])
            self.crowding_dist = np.empty([pop_size,
                                       1])
            
        else:
            pop_size = self.population_size
        
        for i in range(pop_size):
            #self.variable_args[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_args()
            
            # Create a feasible route set
            self.variables[i] = route_gen_func(main_problem)
            self.variables_str[i] = gf.convert_routes_list2str(self.variables[i])
            
            # Determine the objective function values
            start_time = datetime.now() # TIMING FUNCTION
            self.objectives[i,] = fn_obj(self.variables[i], main_problem)
            end_time = datetime.now() # TIMING FUNCTION
            st.append(start_time) # TIMING FUNCTION
            ft.append(end_time) # TIMING FUNCTION
    
            # Determine and print projections
            if i == average_at-1 or i == self.population_size-1: # TIMING FUNCTION
                diffs = [x-y for x,y in zip(ft,st)]
                diffs_sec = [float(str(x.seconds)+"."+str(x.microseconds)) for x in diffs]
                avg_time = np.average(np.asarray(diffs_sec))
                tot_time = np.sum(np.asarray(diffs_sec))
                tot_iter = pop_size
                ga.time_projection(avg_time, tot_iter, t_now=t_now, print_iter_info=True) # prints the time projection of the algorithm

            # get the objective space values and objects
            # F = pop.get("F").astype(np.float, copy=False)
        F = self.objectives
        
            # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F)

        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = gf.calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.rank[i] = k
                self.crowding_dist[i] = crowding_of_front[j]
    
    def generate_or_load_initial_population(self, main_problem, fn_obj, route_gen_func=Routes.return_feasible_route_robust_k_shortest, pop_choices=False):
        t_now = datetime.now() # TIMING FUNCTION
        st = [] # List for starting times
        ft = [] # List for finishing times
        average_at = 5 # TIMING FUNCTION
        
        # Loads the population if it exists
        if pop_choices:
            choice_indices = random.sample(range(pop_choices.population_size), self.population_size)
            
            for i, j in enumerate(choice_indices):
                self.variables[i] = pop_choices.variables[j]
                self.variables_str[i] = pop_choices.variables_str[j]
                self.objectives[i,] = pop_choices.objectives[j,]

            
        else:
            for i in range(self.population_size):
                #self.variable_args[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_args()
                
                # Create a feasible route set
                self.variables[i] = route_gen_func(main_problem)
                self.variables_str[i] = gf.convert_routes_list2str(self.variables[i])
                
                # Determine the objective function values
                start_time = datetime.now() # TIMING FUNCTION
                self.objectives[i,] = fn_obj(self.variables[i], main_problem)
                end_time = datetime.now() # TIMING FUNCTION
                st.append(start_time) # TIMING FUNCTION
                ft.append(end_time) # TIMING FUNCTION
        
                # Determine and print projections
                if i == average_at-1 or i == self.population_size-1: # TIMING FUNCTION
                    diffs = [x-y for x,y in zip(ft,st)]
                    diffs_sec = [float(str(x.seconds)+"."+str(x.microseconds)) for x in diffs]
                    avg_time = np.average(np.asarray(diffs_sec))
                    tot_time = np.sum(np.asarray(diffs_sec))
                    tot_iter = ga.determine_total_iterations(main_problem, 1)
                    ga.time_projection(avg_time, tot_iter, t_now=t_now, print_iter_info=True) # prints the time projection of the algorithm
    
        # get the objective space values and objects
        # F = pop.get("F").astype(np.float, copy=False)
        F = self.objectives
        
            # do the non-dominated sorting until splitting front
        fronts = NonDominated_Sorting().do(F)

        for k, front in enumerate(fronts):
    
            # calculate the crowding distance of the front
            crowding_of_front = gf.calc_crowding_distance(F[front, :])
    
            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.rank[i] = k
                self.crowding_dist[i] = crowding_of_front[j]     

               
    def generate_good_initial_population(self, main_problem, fn_obj):
        """Generate initial population based the best n_trial solutions"""
        # See SA Section for initial pop generation
        
        for i in range(self.population_size):
            #self.variable_args[i,] = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_args()
            self.variables[i] = Routes.return_feasible_route(main_problem)
            self.objectives[i,] = fn_obj(self.variables[i], main_problem)
            
            # get the objective space values and objects
            # F = pop.get("F").astype(np.float, copy=False)
            F = self.objectives
        
            # do the non-dominated sorting until splitting front
            fronts = NonDominated_Sorting().do(F)

            for k, front in enumerate(fronts):
        
                # calculate the crowding distance of the front
                crowding_of_front = gf.calc_crowding_distance(F[front, :])
        
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

    def insert_solution_into_pop(self, solutions, main_problem, fn_obj=False, obj_values=False):
        # Create a feasible route set
        if obj_values:
            assert len(solutions) == len(obj_values())
        
        for i in range(len(solutions)):
            self.variables[i] = solutions[i]
            self.variables_str[i] = gf.convert_routes_list2str(self.variables[i])
            
            # Determine the objective function values
            if obj_values:
                self.objectives[i,] = (obj_values[i,0], obj_values[i,1])
            
            else:
                if callable(fn_obj):
                    self.objectives[i,] = fn_obj(self.variables[i], main_problem)
                else:
                    print("Provide a function to evaluate variables.")

        

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


    
#%% Class: UTFSP_problem and main components

class Problem_data():
    """A class for storing the data of a generic problem"""
    def __init__(self, mx_dist, mx_demand, mx_coords: list):
        self.mx_dist = mx_dist
        self.mx_demand = mx_demand
        self.mx_coords = mx_coords
        
class K_shortest_paths():
    """A class for storing the k_shortest path specific data"""
    def __init__(self, df_k_shortest_paths: pd.DataFrame):
        
        k_shortest_paths = []
        for index_i in range(len(df_k_shortest_paths)):
            k_shortest_paths.append(gf.convert_path_str2list(df_k_shortest_paths["Routes"].iloc[index_i]))
        
        self.df = df_k_shortest_paths
        self.paths = k_shortest_paths
        self.lengths = np.float64(df_k_shortest_paths["Travel_time"].values)
        self.demand = np.float64(df_k_shortest_paths["Demand"].values)
        self.demand_per_length = df_k_shortest_paths["Demand_per_minute"].values
        
        loc_v_0 = df_k_shortest_paths.columns.get_loc("v_0")
        self.vertices_bin = df_k_shortest_paths.iloc[: , loc_v_0:].to_numpy()


    def create_paths_bool(self, tot_num_vertices):
        """A function that creates a boolean matrix indicating the vertices 
        each path contains"""
        paths_bool = np.zeros((len(self.paths), tot_num_vertices))
        for index_i in range(len(self.paths)):
            for vertex in self.paths[index_i]:
                paths_bool[index_i,vertex] = 1
              
        self.paths_bool = paths_bool
        
        return paths_bool
        
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
            
class Problem_metaheuristic_inputs():
    """A class for storing the input parameters of the Genetic Algorithm"""
    def __init__(self, parameters_input: dict, **kwargs):
        for k, v in parameters_input.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

class UTNDP_problem():
    """A class for storing all the information pertaining to a problem"""
    pass


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
            rank = gf.rank_from_fronts(fronts, F.shape[0])
            return fronts, rank

        return fronts
    
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
    
#%% Class: InitialisationFreq        
class InitialisationFreq(gf2.Frequencies):      
    """A class used to initialise the population with individuals"""
    def __init__(self, main_problem, pop_size) -> None:
        super(gf2.Frequencies, self).__init__()       
    
    def generate_random_individual(self, main_problem, fn_obj):
        self.X = gf2.Frequencies(main_problem.R_routes.number_of_routes).return_random_theta_frequencies()
        self.F = fn_obj(self.X, main_problem)
        
# %% Class: TournamentSelection_g2
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
    
#%% Create in PYMOO form
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
    
