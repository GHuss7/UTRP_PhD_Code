"""
An efficient Cython implementation of the Floyd-Warshall algorithm for finding the shortest path distances between all nodes of a weighted graph.
See http://en.wikipedia.org/wiki/Floyd-Warshall_algorithm

Amit Moscovich Eiger, 2014
"""
import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy


#%% Shortest Path Matrix Cython

def shortest_paths_matrix(double[:,:] D, int[:] inv_map, int t, int n):

    cdef int i, j, p1, p2

    cdef double[:,:] SPMatrix = numpy.inf*numpy.ones((n,n))

    for i in range(t):
        p1 = inv_map[i]
        for j in range(t):
            p2 = inv_map[j]
            if (D[i][j]<SPMatrix[p1][p2]):
                SPMatrix[p1][p2] = D[i][j]

    return numpy.asarray(SPMatrix)

#%% ExpandTravelMatrix

def expandTravelMatrix(routes, travelTimes,n,tp,wt):
    #numpy.set_printoptions(threshold= 200)
    cdef int t = 0 # t give the sum of the number of nodes in all the routes
    cdef int r = len(routes) # Number of routes
    cdef int i, j, length
    routelength = numpy.zeros((r,), dtype=int)
    cdef float shortest = numpy.inf
    cdef int longest = 0
    
    for i in range(r):
        length = len(routes[i])
        routelength[i] = length
        t = t + length
        if length < shortest:
            shortest = length
        if length > longest:
            longest = length
    #print('routelength array',routelength)
    routeadj = numpy.empty((t,t))
    routeadj[:] = numpy.inf
    cdef int displacement = 0
    mapping = [[]]*n
    inv_map = numpy.zeros((t,), dtype=int)
    for i in range(r):
        for j in range(routelength[i]-1):
            p1 = routes[i][j]
            p2 = routes[i][j+1]
            q1 = j + displacement
            q2 = j + 1 + displacement
            routeadj[q1][q2] = travelTimes[p1][p2]
            routeadj[q2][q1] = travelTimes[p2][p1]
            mapping[p1]= mapping[p1] + [q1]
            mapping[p2]= mapping[p2] + [q2]
            inv_map[q1] = p1
            inv_map[q2] = p2
        displacement = displacement + routelength[i]
    # add the 5 minute delays for vehicle changes
    for i in range(n):
        for j in range(int(len(mapping[i]))-1):
            for k in range((j+1),int(len(mapping[i]))):
                q1 = mapping[i][j]
                q2 = mapping[i][k]
                routeadj[q1][q2] = tp + wt
                routeadj[q2][q1] = tp + wt
                
    routeadj = numpy.array(routeadj)
    return(routeadj,inv_map,t,shortest,longest)  