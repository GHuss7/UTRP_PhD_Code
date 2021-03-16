# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:57:21 2021

@author: 17832020
"""
# %% Import Libraries
import pandas as pd 
import numpy as np

# %% Import personal functions
import DSS_Admin as ga
import DSS_UTNDP_Functions as gf
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import EvaluateRouteSet as ev
import DSS_UTNDP_Classes as gc

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
        Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

csv_combined = pd.read_csv("Validation_data.csv")

df_combined = pd.DataFrame()
df_combined = df_combined.assign(F_3=csv_combined[:,0])
df_combined = df_combined.assign(F_4=csv_combined[:,1])
df_combined = df_combined[is_pareto_efficient(df_combined.iloc[:,0:2].values, True)]
df_combined.to_csv("mandl_solution_UTFSP_by_john_overall")