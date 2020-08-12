# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:24:00 2019

@author: 17832020
"""
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
from numpy import array,loadtxt,isfinite,set_printoptions,zeros,ones,int,empty,inf,asarray,minimum,newaxis
import numpy as np
from EvaluateRouteSet import *


def main():

    '''Read files'''
    travelTimes, DemandMat, routes,smallest = readfiles()
    n = n = len(DemandMat)
    print('n =',n)
    total_demand = sum(sum(DemandMat))/2
    '''Get user input'''
    tp = getusrinput()
    
    '''Convert route sets to start at node 0 '''
    routes, smallest = standardize(routes)
    
    '''Check validity of route set'''
    connected, allPresent, duplicates = checkValidity(routes,n)
    print('validity checked')
    if (not connected) or (not allPresent):
        print('invalid route set')
        if not connected:
            print('not connected')
        if not allPresent:
            print('one or more nodes is missing')
    
        return()
    if duplicates:
            print('one or more node is duplicated in a single route making a cycle')
    '''waiting time, wt, is different from transfer penalty,tp'''
    wt = 0 #waiting time, can be changed if desired
    ATT,d0,d1,d2,drest,noRoutes,longest,shortest = fullPassengerEvaluation(routes, \
                        travelTimes,DemandMat, total_demand,n,tp,wt)
    RL = evaluateTotalRouteLength(routes,travelTimes)
    if RL > 0:
        print('ATT =',ATT)
        print('d0 =',d0)
        print('d1 =',d1)
        print('d2 =',d2)
        print('dun =',drest)
        print('RL =',RL)
        print('No of routes = ',noRoutes)
        print('Shortest route is',shortest,'nodes long')
        print('Longest route is',longest,'nodes long')
    else:
        print('illegal edge present...')
    input('Press any key to continue ')


def fullPassengerEvaluation(routes, travelTimes,DemandMat, total_demand,n,tp,wt):
    routeadj,changes,inv_map,t,shortest,longest = expandTravelMatrixChanges(routes, travelTimes,n,tp,wt)
    D, changes = FastFloydChanges(routeadj,changes)
    SPMatrix,ChMatrix = shortest_paths_matrixChanges(D, inv_map, t, n, changes)
    print(ChMatrix)
    a = np.array(ChMatrix)
    np.savetxt('ChangesMat.txt', a.astype(int), fmt='%i', delimiter=' ')   # X is an array
    d0, d1, d2, drest, ATT, total_ATT, total_TT = EvaluateChanges(n, SPMatrix, ChMatrix, DemandMat, total_demand, tp, wt)
    noRoutes = len(routes)
    return(ATT,d0,d1,d2,drest,noRoutes,longest,shortest)
    
     
    
def expandTravelMatrixChanges(routes, travelTimes,n,tp,wt):
    set_printoptions(threshold= 200)
    t = int(0); # t give the sum of the number of nodes in all the routes
    r = len(routes) # Number of routes
    routelength = zeros((r,), dtype=int)
    shortest = inf
    longest = 0
    for i in range(r):
        length = len(routes[i])
        routelength[i] = length
        t = t + length
        if length < shortest:
            shortest = length
        if length > longest:
            longest = length
    #print('routelength array',routelength)
    routeadj = empty((t,t))
    routeadj[:] = inf
    changes = zeros((t,t),dtype=int)
    displacement = 0    # displacement keeps track of where in the adj matrix you are busy
    mapping = [[]]*n    # mapping shows where all the corresponding nodes are found on the route adj matrix
    inv_map = zeros((t,), dtype=int)
    for i in range(r):
        for j in range(routelength[i]-1):
            p1 = routes[i][j]   # gets the first node index of the examined edge
            p2 = routes[i][j+1] # gets the second node index of the examined edge
            q1 = j + displacement       # sets the one index for the expanded route adj matrix
            q2 = j + 1 + displacement   # sets the other index for the expanded route adj matrix
            routeadj[q1][q2] = travelTimes[p1][p2]  # populate the route adj matrix both ways
            routeadj[q2][q1] = travelTimes[p2][p1]  # populate the route adj matrix both ways
            mapping[p1]= mapping[p1] + [q1]         # adds the routes adj matrix index to the correct mapping position of the node involved
            mapping[p2]= mapping[p2] + [q2]         # adds the routes adj matrix index to the correct mapping position of the node involved
            inv_map[q1] = p1                        # adds the inverse of the map to keep track of what node is represented in the route adj matrix
            inv_map[q2] = p2                        # adds the inverse of the map to keep track of what node is represented in the route adj matrix
        displacement = displacement + routelength[i] # increment the displacement according
                                                        # to the current route's length
    # add the 5 minute delays for vehicle changes
    
    # note to self: the mapping contains doubles and leads to redundant additions, but is probably because of
    # the node i to i transfers and it's a cost to pay 
    
    # mapping contains all of the same nodes but spread over the different routes in the adj matrix
    
    for i in range(n):
        for j in range(int(len(mapping[i]))-1):
            for k in range((j+1),int(len(mapping[i]))):
                q1 = mapping[i][j]                  # not sure about how mapping works ???
                q2 = mapping[i][k]                  # not sure about how mapping works ???
                routeadj[q1][q2] = tp + wt          # adds the penalties to the adj matrix where transfers would occur
                routeadj[q2][q1] = tp + wt          # adds the penalties to the adj matrix where transfers would occur
                changes[q1][q2] = 1                 # indicates where transfers would occur
                changes[q2][q1] = 1                 # indicates where transfers would occur

    routeadj = array(routeadj)
    changes = array(changes)
    return(routeadj,changes,inv_map,t,shortest,longest)
    
    
def FastFloydChanges(m,changes):
    m, changes,t = check_and_convert_adjacency_matrix(m,changes)
    for k in range(t):
        new_m = m[newaxis,k,:] + m[:,k,newaxis]             # adds the two vectors to each other over the nxn matrix
        new_c = changes[newaxis,k,:] + changes[:,k,newaxis] # adds the two vectors to each other over the nxn matrix
        bool1 = (m > new_m) + zeros((t,t),dtype=int)        # boolean for where the adj mx is > than the new m mx
        bool2 = (m<= new_m) + zeros((t,t),dtype=int)        # boolean for where the adj mx is <= than the new m mx
        bool3 = (m == new_m) + zeros((t,t),dtype=int)       # boolean for where the adj mx is = to the new m mx
        bool4 = (changes > new_c) + zeros((t,t),dtype=int)  # boolean for where the changes mx is > than the new c mx
        bool5 = bool3*bool4
        bool1 = bool1 + bool5
        bool2 = bool2 - bool5
        temp1 = bool1*new_c
        temp2 = bool2*changes
        changes = temp1 + temp2
        m = minimum(m, new_m)
    return(m, changes)
    
    
def shortest_paths_matrixChanges(D, inv_map, t, n, changes):

    SPMatrix = inf*ones((n,n), dtype=float)
    ChMatrix = zeros((n,n), dtype=int)
    #count = 0
    for i in range(t):
        p1 = inv_map[i]
        for j in range(t):
            p2 = inv_map[j]
            if (D[i][j]<SPMatrix[p1][p2]) or ((D[i][j]==SPMatrix[p1][p2]) and (changes[i][j]<ChMatrix[p1][p2])):
                SPMatrix[p1][p2] = D[i][j]
                ChMatrix[p1][p2] = changes[i][j]
                #count = count + 1
    return(SPMatrix,ChMatrix)