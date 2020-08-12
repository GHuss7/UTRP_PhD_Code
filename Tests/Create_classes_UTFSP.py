# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:07:37 2020

@author: 17832020
"""

class Constraints():
    
    def __init__(self, mx_dist):
            self.con_r = 6               # number of allowed routes (aim for > [numNodes N ]/[maxNodes in route])
            self.con_minNodes = 3                       # minimum nodes in a route
            self.con_maxNodes = 10                       # maximum nodes in a route
            self.con_N_nodes = len(mx_dist)             # number of nodes in the network
            self.con_fleet_size = 40                     # number of vehicles that are allowed
            self.con_vehicle_capacity = 20                 # the max carrying capacity of a vehicle
        
#c2 = Constraints(mx_dist)

class Problem_UTFSP(Constraints):
    
    def __init__(self, mx_dist):
        super().__init__(mx_dist)
        


#prob_1 = Problem_UTFSP(mx_dist)

#prob_1.con_N_nodes
        
#problem = Problem_UTFSP(mx_dist)   
     
#%% Test 1

def convert_routes_str2list(routes_R_str):
    # converts a string standarised version of routes list into a routes list
    routes_R_list = list()
    temp_list = list()
    flag_end_node = True
    for i in range(len(routes_R_str)):
        if routes_R_str[i] != "-" and routes_R_str[i] != "*":
            if flag_end_node:
                temp_list.append(int(routes_R_str[i]))
                flag_end_node = False
            else:
                temp_list[len(temp_list)-1] = int(str(temp_list[len(temp_list)-1]) + routes_R_str[i])
        else:   
            if routes_R_str[i] == "*":          # indicates the end of the route
                routes_R_list.append(temp_list)
                temp_list = list()
                flag_end_node = True
            else:
                if routes_R_str[i] == "-":
                    flag_end_node = True
    return routes_R_list

R_routes = convert_routes_str2list("13-14-10-8-15*4-2-3-6*13-11-12*7-15-9*5-4-6-15*3-2-1*")
for i in range(len(R_routes)): # get routes in the correct format
    R_routes[i] = [x - 1 for x in R_routes[i]] # subtract 1 from each element in the list

# %% Initialise the decision variables

class Routes():
    # A class for the general routes used as an array
    def __init__(self, R_routes: list):
        self.number_of_routes = len(R_routes)
        self.routes = R_routes
        
    def __str__(self) -> str:
        return gf.convert_routes_list2str(self.routes)
        
    
r_1 = Routes(R_routes)
r_1.number_of_routes    
r_1.routes
r_1.__str__()


r_2 = Routes([[1,3,4,5], [5,6,7]])
r_2.number_of_routes

r_3 = Routes("hello")
r_3.number_of_routes

r_1.__dict__
r_1.__str__()
r_3.__dict__

r_4 = Routes(dict(gun = 1, len = 7))
r_4.__dict__


