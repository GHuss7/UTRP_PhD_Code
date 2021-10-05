"""
To run these tests, assuming you have py.test installed, just run:
    py.test -vs

('-v' is verbose and '-s' switch does not suppress prints)
"""

import time
from numpy import array, inf, asarray, diagonal, minimum, newaxis, fill_diagonal
from numpy.random import random

import pyximport; #pyximport.install()
pyximport.install(reload_support=True)
import floyd_warshall

def check_and_convert_adjacency_matrix(adjacency_matrix):
    mat = asarray(adjacency_matrix)

    (nrows, ncols) = mat.shape
    assert nrows == ncols
    n = nrows

    assert (diagonal(mat) == 0.0).all()

    return (mat, n)

def floyd_warshall_naive(adjacency_matrix):
    (mat, n) = check_and_convert_adjacency_matrix(adjacency_matrix)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                mat[i,j] = min(mat[i,j], mat[i,k] + mat[k,j])

    return mat

def floyd_warshall_numpy(adjacency_matrix):
    '''floyd_warshall_numpy(adjacency_matrix) -> shortest_path_distance_matrix

    A vectorized NumPy implementation of the Floyd-Warshall algorithm.

    Input
        An NxN NumPy array describing the directed distances between N nodes.

        adjacency_matrix[i,j] = distance to travel directly from node i to node j (without passing through other nodes)

        Notes:
        * If there is no edge connecting i->j then adjacency_matrix[i,j] should be equal to numpy.inf.
        * The diagonal of adjacency_matrix should be zero.

    Output
        An NxN NumPy array such that result[i,j] is the shortest distance to travel between node i and node j. If no such path exists then result[i,j] == numpy.inf
    '''
    (mat, n) = check_and_convert_adjacency_matrix(adjacency_matrix)

    for k in range(n):
        mat = minimum(mat, mat[newaxis,k,:] + mat[:,k,newaxis]) 

    return mat

def test_floyd_warshall_algorithms_on_small_matrix():
    INPUT = array([
        [  0.,  inf,  -2.,  inf],
        [  4.,   0.,   3.,  inf],
        [ inf,  inf,   0.,   2.],
        [ inf,  -1.,  inf,   0.]
    ])

    OUTPUT = array([
        [ 0., -1., -2.,  0.],
        [ 4.,  0.,  2.,  4.],
        [ 5.,  1.,  0.,  2.],
        [ 3., -1.,  1.,  0.]])

    assert (floyd_warshall_naive(INPUT) == OUTPUT).all()
    assert (floyd_warshall_numpy(INPUT) == OUTPUT).all()
    assert (floyd_warshall.floyd_warshall_single_core(INPUT) == OUTPUT).all()
    assert (floyd_warshall.floyd_warshall_parallelized(INPUT) == OUTPUT).all()

class Timer(object):
    def __init__(self, text = None):
        self.start_clock = time.process_time() # time.process_time() #alternative
        self.start_time = time.time()
        if text != None:
            print ('%s:' % text), 
    def stop(self):
        wall_time = time.time() - self.start_time
        cpu_time = time.process_time() - self.start_clock
        print(f'Wall time: {wall_time:.3f} seconds.  CPU time: {cpu_time:.3f} seconds.')
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.stop()
        
class Timer_2(object):
    def __init__(self, text = None):
        self.start_clock = time.process_time() # time.process_time() #alternative
        self.start_time = time.time()
        if text != None:
            print ('%s:' % text), 
    def stop(self):
        wall_time = time.time() - self.start_time
        cpu_time = time.process_time() - self.start_clock
        print(f'Wall time: {wall_time:.3f} seconds.  CPU time: {cpu_time:.3f} seconds.')
        return self
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.stop()

def test_floyd_warshall_speed():
    M = random((1500, 1500))
    fill_diagonal(M, 0)

    prev_res = None
    print('')
    for (name, func) in [#('Naive', floyd_warshall_naive),
                         ('Python+NumPy', floyd_warshall_numpy), 
                         ('Cython', floyd_warshall.floyd_warshall_single_core), 
                         ('Cython multicore', floyd_warshall.floyd_warshall_parallelized),
                         ('Cython mutlicore no bounds checking', floyd_warshall.floyd_warshall_parallelized_nbc)]:
        print (f"{name}"),
        with Timer():
            result = func(M)
        if not (prev_res == result).all():
        #if prev_res == None:
            prev_res = result
        else:
            assert (prev_res == result).all()

# test_floyd_warshall_speed()

import pandas as pd
def test_floyd_warshall_speed_various():
    results_df = pd.DataFrame(columns=["Vertices","Clock Time","CPU Time"])
    
    test_sizes = [#100,200,300,400,
                  500,1000,2000,3000,
                  #4000,5000
                  ]
    
    for i in range(len(test_sizes)):
        print(f"Testing: {test_sizes[i]} Vertices #######################################################")
        for j in range(20):
            M = random((test_sizes[i], test_sizes[i]))
            fill_diagonal(M, 0)
        
            prev_res = None
            print('')
            for (name, func) in [('Cython mutlicore no bounds checking', floyd_warshall.floyd_warshall_parallelized_nbc)]:
                #print (f"{name}"),
                with Timer_2():
                    result = func(M)
                if not (prev_res == result).all():
                #if prev_res == None:
                    prev_res = result
                else:
                    assert (prev_res == result).all()
                
    return results_df

test_floyd_warshall_speed_various()
