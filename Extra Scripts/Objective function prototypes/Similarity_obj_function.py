# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:50:00 2021

@author: 17832020
"""

from collections import Counter

inventory = Counter()

loot = {'sword': 1, 'bread': 3}
inventory.update(loot)
inventory
Counter({'bread': 3, 'sword': 1})

more_loot = {'sword': 1, 'apple': 1}
inventory.update(more_loot)
inventory
Counter({'bread': 3, 'sword': 2, 'apple': 1})

len(inventory)
sum(inventory.values())

route_1 = Counter()
route_2 = Counter()


route_1_list = [(1, 2), (2, 3), (3, 4), (5, 2), (2, 6)]
route_2_list = [(1, 2), (2, 3), (3, 4), (5, 2), (2, 3), (3, 6)]

route_1.update(route_1_list)
route_2.update(route_2_list)

import multiset

def return_all_route_set_edges(R_x):
    R_x_edges = [(P_x[i], P_x[i+1]) for P_x in R_x for i in range(len(P_x)-1)]
    return R_x_edges

def calc_route_set_similarity(R_1, R_2):
    '''Takes as input two route sets, each being a list of lists,
    for example:
        R_1 = [[1,2,3,4], [5,2,6]]
        R_2 = [[1,2,3,4], [5,2,3,6]]
    and returns the percentage similarity in terms of edges'''
    
    R_1_edges = return_all_route_set_edges(R_1)
    R_2_edges = return_all_route_set_edges(R_2)

    # Working with the multisets
    R_1_ms = multiset.Multiset(R_1_edges)
    R_2_ms = multiset.Multiset(R_2_edges)

    len(R_1_ms.intersection(R_2_ms))
    
    similarity = 1 - 2*(len(R_1_ms.intersection(R_2_ms)))/(len(R_1_ms) + len(R_2_ms))
    return similarity

