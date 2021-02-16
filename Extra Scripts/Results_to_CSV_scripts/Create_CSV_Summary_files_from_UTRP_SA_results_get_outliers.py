# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:48:53 2020

@author: 17832020
"""
import pandas as pd
import os
import re
import string
from pathlib import Path


prefix_for_each_csv_file = "UTRP_SA_Outliers_Summary"
spl_word = 'SA_' # initializing split word 

def count_Run_folders(path_to_folder):
    # NB only all the folders of Runs should me in the main path, otherwise errors will occur
    result_entries = os.listdir(path_to_folder) # gets the names of all the results entries
    counter = 0
    for i in range(len(result_entries)):
        if re.match("^Run_[0-9]+$", result_entries[i]):
            counter = counter + 1
    return counter


path_to_main_folder = Path(os.getcwd())
nr_of_runs = count_Run_folders(path_to_main_folder)
result_entries = os.listdir(path_to_main_folder)
         
df_list_of_ST_results_HV = []
df_list_of_ST_results_Iterations = []
parameters_list = []

""" Captures and saves all the data """
for results_folder_name in result_entries:  
    if os.path.isdir(path_to_main_folder / results_folder_name):
  
        """Get the substrings"""
        test_string = results_folder_name  # initializing string 
        res = test_string.partition(spl_word)[2] # partitions the string in three

        value = re.split("_", res)[-1] # gets the last number
        parameter_name = re.split("_[0-9]", res)[0] # gets the parameter name


        """Capture all the results into dataframes"""
        if parameter_name in parameters_list:
            df_results_read = pd.read_csv(f"{path_to_main_folder / results_folder_name}/Results_description_HV_with_outliers.csv")
            df_results_read["parameter"] = parameter_name
            df_results_read["value"] = value
            #df_results_read.rename(columns={0 :'Measurement'}, inplace=True )

            df_index = parameters_list.index(parameter_name)
            df_list_of_ST_results_HV[df_index] = df_list_of_ST_results_HV[df_index].append(df_results_read)
        
        else:
            # Creates the new dataframe
            parameters_list.append(parameter_name)
            df_list_of_ST_results_HV.append(pd.DataFrame())
            
            df_results_read = pd.read_csv(f"{path_to_main_folder / results_folder_name}/Results_description_HV_with_outliers.csv")
            df_results_read["parameter"] = parameter_name
            df_results_read["value"] = value
            #df_results_read.rename(columns={0 :'Measurement'}, inplace=True )
            
            df_index = parameters_list.index(parameter_name)
            df_list_of_ST_results_HV[df_index] = df_results_read


"""Print dataframes as .csv files"""
if True:
    for parameter, results_dataframe in zip(parameters_list, df_list_of_ST_results_HV):
        results_dataframe.to_csv(path_to_main_folder / f"{prefix_for_each_csv_file}_{parameter}.csv")




