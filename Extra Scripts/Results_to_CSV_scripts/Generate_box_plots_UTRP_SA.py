# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:30:32 2020

@author: 17832020
"""

import pandas as pd
import os
import re
import string
from pathlib import Path
import math

'''State the dictionaries''' 
title = {
'max_iterations_t' : 'Max iterations per epoch',               
'min_accepts' : 'Minimum accepts',                        
'max_attempts' : 'Maximum attempts',                       
'max_reheating_times' : 'Maximum reheating times',              
'max_poor_epochs' : 'Maximum poor epochs',
'Cooling_rate' : 'Cooling rate',                  
'Temp' : 'Starting temperature',               
'Reheating_rate' : 'Reheating rate'                            
}

file_suffix = {
'max_iterations_t' : 'max_iterations_per_epoch',               
'min_accepts' : 'min_accepts',                        
'max_attempts' : 'max_attempts',                       
'max_reheating_times' : 'max_reheating_times',              
'max_poor_epochs' : 'max_poor_epochs',
'Cooling_rate' : 'cooling_rate',                  
'Temp' : 'initial_temp',               
'Reheating_rate' : 'reheating_rate'                            
}

def isNaN(num):
    return num != num

def outlier_logic(outlier_entry, row_i):
    if isNaN(outlier_entry): 
        return "" 
    else: 
        split_str = str(outlier_entry).split(", ")
        temp_str = str()
        for _ in split_str:
            temp_str = "\n".join([temp_str, f'({row_i+1},{round(float(_),5)})']) # f'({row_i+1},{_})'
        return temp_str

prefix_for_each_csv_file = "UTRP_SA_Outliers_Summary"
spl_word = 'SA_' # initializing split word 


path_to_main_folder = Path(os.getcwd())
result_entries = os.listdir(path_to_main_folder)
         
df_list_of_ST_results_HV = []
df_list_of_ST_results_Iterations = []
parameters_list = []
nl = '\n'

box_plot_str = str()

""" Captures and saves all the data """
for file_name in result_entries:  
    if file_name.endswith('.csv'):
        df_results_read = pd.read_csv(f"{path_to_main_folder}/{file_name}")
        
        """Generate xticks"""
        x_tick_str = ",".join([str(x+1) for x in range(len(df_results_read["value"]))])
        x_tick_labels_str = ",".join([str(x) for x in df_results_read["value"]])
        
        """Add the prerequisites""" 
        temp_str = (f'% ######################## UTRP SA {title[df_results_read["parameter"].iloc[0]]} ######################## {nl}' \
            f'\\begin{{figure}} {nl}' \
         	f'\\centering {nl}' \
         	f'\\tikzsetnextfilename{{UTRP_DBMOSA_BP_{file_suffix[df_results_read["parameter"].iloc[0]]}}} {nl}' \
         	f'\\begin{{tikzpicture}} {nl}' \
          	f'\\begin{{axis}}[ {nl}' \
          	f'title={{{title[df_results_read["parameter"].iloc[0]]}}}, {nl}' \
          	f'boxplot/draw direction=y, {nl}' \
          	f'xtick={{{x_tick_str}}}, {nl}' \
          	f'xticklabels={{{x_tick_labels_str}}}, {nl}' \
          	f'x tick label style={{rotate=0, align=center}}, {nl}' \
          	f'xlabel={{{title[df_results_read["parameter"].iloc[0]]}}}, {nl}' \
          	f'ylabel={{HV obtained [\\%]}}, {nl}' \
        	f'] {nl}') 
            
        box_plot_str = "\n".join([box_plot_str, temp_str])

        """Add each plot"""
        for row_i in range(len(df_results_read)):
            temp_str = (f'% ############## {df_results_read["parameter"].iloc[row_i]}={df_results_read["value"].iloc[row_i]} ################## {nl}' \
            f'\\addplot[boxplot, mark=*, {nl}' \
            f'boxplot prepared={"{"} {nl}' \
			f'lower whisker={round(df_results_read["min"].iloc[row_i],5)}, {nl}' \
			f'upper whisker={round(df_results_read["max"].iloc[row_i],5)}, {nl}' \
			f'lower quartile={round(df_results_read["lq"].iloc[row_i],5)}, {nl}' \
			f'upper quartile={round(df_results_read["uq"].iloc[row_i],5)}, {nl}' \
			f'median={round(df_results_read["med"].iloc[row_i],5)}, {nl}' \
			f'average={round(df_results_read["mean"].iloc[row_i],5)}{"}"}, {nl}' \
    		f'color = blue, solid, area legend] {nl}' \
    		f'coordinates {{{outlier_logic(df_results_read["outliers"].iloc[row_i], row_i)}}}; {nl}')
                
            box_plot_str = "\n".join([box_plot_str, temp_str])
                
        """Add ending"""
       	temp_str = (f'\\end{{axis}}{nl}' \
       	f'\\end{{tikzpicture}}{nl}' \
        f'\\end{{figure}} {nl}') 
               
        box_plot_str = "\n".join([box_plot_str, temp_str])
        
with open("Tikz_boxplot_UTRP_SA.txt", "w") as text_file:
    print(box_plot_str, file=text_file)