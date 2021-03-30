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

# Creating the edge lists
R_x = [1,2,3,4]
[(R_x[i], R_x[i+1]) for i in range(len(R_x)-1)]

# Working with the multisets
R_1 = multiset.Multiset([(1, 2), (2, 3), (3, 4), (5, 2), (2, 6)])
R_2 = multiset.Multiset([(1, 2), (2, 3), (3, 4), (5, 2), (2, 3), (3, 6)])

len(R_1.intersection(R_2))

similarity = 1 - 2*(len(R_1.intersection(R_2)))/(len(R_1) + len(R_2))
