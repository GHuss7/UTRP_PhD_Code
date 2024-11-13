floyd-warshall-cython
=====================

An efficient Cython implementation of the Floyd-Warshall algorithm for finding the shortest path distances between all pairs of vertices in a weighted directed graph originally developed by Amit Moscovich Eiger (moscovich@gmail.com). See http://en.wikipedia.org/wiki/Floyd-Warshall_algorithm

Günther Hüsselmann (ghussel94@gmail.com) built on Amit's code to make the Floyd-Warshall algorithm even more efficient for Cython by introducing more constraints so that even less checks were required, as well as using all of the computer's cores.
