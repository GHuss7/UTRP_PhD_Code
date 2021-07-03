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

def shortest_paths_matrix(int[:,:] D, int[:] inv_map, int t, int n):

    cdef unsigned int i, j, p1, p2

    cdef float[:,:] SPMatrix = numpy.inf*numpy.ones((n,n), dtype=float)

    for i in range(t):
        p1 = inv_map[i]
        for j in range(t):
            p2 = inv_map[j]
            if (D[i][j]<SPMatrix[p1][p2]):
                SPMatrix[p1][p2] = D[i][j]

    return(SPMatrix)