# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:30:32 2020

@author: 17832020
"""

import pandas as pd
import os
import subprocess
import re
import string
from pathlib import Path
import math

# Ensure the directory is set to the file location
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

'''State the dictionaries''' 
title = {
'con_minNodes' : 'Minimum vertices',               
'con_r' : 'Number of routes',                        
'crossover_probability' : 'Crossover probability',                       
'generations' : 'Number of generations',              
'mutation_probability' : 'Mutation probability',
'population_size' : 'Population size',

'Initial_solutions' : 'Use supplemented initial solution set'                           
}

file_suffix = {
'con_minNodes' : 'min_nodes',               
'con_r' : 'num_routes',                        
'crossover_probability' : 'crossover_probability',                       
'generations' : 'num_generations',              
'mutation_probability' : 'mutation_probability',
'population_size' : 'population_size',

'Initial_solutions' : 'initial_solutions'                           
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

prefix_for_each_csv_file = "UTRP_GA_Outliers_Summary"
spl_word = 'GA_' # initializing split word 


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
        temp_str = (f'% ######################## UTRP GA {title[df_results_read["parameter"].iloc[0]]} ######################## {nl}' \
            f'\\begin{{figure}} {nl}' \
         	f'\\centering {nl}' \
         	f'\\tikzsetnextfilename{{UTRP_NSGAII_BP_{file_suffix[df_results_read["parameter"].iloc[0]]}}} {nl}' \
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
        
with open("Tikz_boxplots_UTRP_GA.txt", "w") as text_file:
    print(box_plot_str, file=text_file)


preamble = r'''\documentclass[crop=false]{standalone}
%\documentclass{standalone}
\usepackage{tikz} % To generate the plot from csv
\usepackage{pgfplots}
\usepackage{graphicx}
\usepackage{subfig}
\usepackage{float}
\usepackage[section]{placeins} % getting figures below sections
\usepackage{blindtext}
\usepackage{siunitx}
\usepgfplotslibrary{units} % Allows to enter the units nicely
\usetikzlibrary{external} %https://tex.stackexchange.com/questions/1460/script-to-automate-externalizing-tikz-graphics
\tikzexternalize[prefix=savedfigures/]

\pgfplotsset{compat=newest} % Allows to place the legend below plot
\usepackage{pgfplotstable}
\usepgfplotslibrary{statistics}

% #################### Function definition for box plots read table ##################\
\makeatletter
\pgfplotsset{
	boxplot prepared from table/.code={
		\def\tikz@plot@handler{\pgfplotsplothandlerboxplotprepared}%
		\pgfplotsset{
			/pgfplots/boxplot prepared from table/.cd,
			#1,
		}
	},
	/pgfplots/boxplot prepared from table/.cd,
	table/.code={\pgfplotstablecopy{#1}\to\boxplot@datatable},
	row/.initial=0,
	make style readable from table/.style={
		#1/.code={
			\pgfplotstablegetelem{\pgfkeysvalueof{/pgfplots/boxplot prepared from table/row}}{##1}\of\boxplot@datatable
			\pgfplotsset{boxplot/#1/.expand once={\pgfplotsretval}}
		}
	},
	make style readable from table=lower whisker,
	make style readable from table=upper whisker,
	make style readable from table=lower quartile,
	make style readable from table=upper quartile,
	make style readable from table=median,
	make style readable from table=average,
	make style readable from table=lower notch,
	make style readable from table=upper notch
}
\makeatother'''

#print(preamble)

#%% Write the tex file and execute
with open('box_plots.tex','w') as file:
    file.write(preamble)
    file.write('\n\\begin{document}\n')
    file.write(box_plot_str)
    file.write('\n\\end{document}\n')

x = subprocess.call('pdflatex box_plots.tex -interaction nonstopmode -shell-escape') # -interaction nonstopmode -shel-escape
if x != 0:
	print('Exit-code not 0, check result!')
else:
	os.system('start box_plots.pdf')
