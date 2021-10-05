#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:48:53 2020

@author: 17832020
"""
import pandas as pd
import os
import re
import string
import json
from pathlib import Path
import subprocess

import argparse

# Arguments from command line
parser = argparse.ArgumentParser()

#-dir DIRECTORY
parser.add_argument("-dir", "--directory", dest = "dir", default = os.path.dirname(os.path.realpath(__file__)), help="Directory", type=str)
parser.add_argument("-mt", "--meta_type", dest = "meta_type", default = 'GA', help="Metaheuristic type", type=str)

args = parser.parse_args()
arg_dir = args.dir
print(arg_dir)



# Ensure the directory is set to the file location
dir_path = arg_dir
os.chdir(dir_path)

meta_type = args.meta_type
#meta_type = 'SA'
if meta_type == 'GA': meta_full = 'NSGAII'
else: meta_full = 'DBMOSA'
if meta_type == 'GA': spl_word = 'GA_' # initializing split word 
else: spl_word = 'SA_' 
if meta_type == 'GA': counter_type = 'Generations'
else: counter_type = 'Total_iterations'
if meta_type == 'GA': counter_xlabel = 'Generations'
else: counter_xlabel = 'Total iterations'
if meta_type == 'GA': counter_ylabel = 'Mutation ratio'
else: counter_ylabel = 'Perturbation ratio'

prefix_for_each_csv_file = f"UTRP_{spl_word}Summary"

'''State the dictionaries for boxplots''' 
title = {
'con_minNodes' : 'Minimum vertices',               
'con_r' : 'Number of routes',                        
'crossover_probability' : 'Crossover probability', 
'Crossover_prob' : 'Crossover probability', 
'generations' : 'Number of generations',              
'mutation_probability' : 'Mutation probability',
'Mut_prob' : 'Mutation probability',
'population_size' : 'Population size',
'Pop_size' : 'Population size',

'Initial_solutions' : 'Use supplemented initial solution set',
'Crossover' : 'Crossover operators applied',
'Mutations' : 'Mutation operators applied',
'Update_mut_ratio' : 'Updating the mutation ratio',
'Mut_threshold' : 'Mutation threshold',
'Repairs' : 'Repair strategy used',
'repair_func' : 'Repair strategy used',   

# SA Parameters
'Mut_update' : 'Updating the mutation ratio',
'Choice_import_saved_set' : 'Import saved solutions',
'B_Mutations_AMALGAM' : 'Mutation operators applied',
'Mutations_AMALGAM' : 'Mutation operators applied',
'Mutations_AMALGAM_every_n' : 'Mutation operators applied',
'Mutations_Counts_normal' : 'Mutation operators applied',
'Mutations_more' : 'Mutation operators applied'


}

file_suffix = { #NB! Suffix may not have any spaces!
'con_minNodes' : 'min_nodes',               
'con_r' : 'num_routes',                        
'crossover_probability' : 'crossover_probability', 
'Crossover_prob' : 'crossover_probability',                      
'generations' : 'num_generations',              
'mutation_probability' : 'mutation_probability',
'Mut_prob' : 'mutation_probability',
'population_size' : 'population_size',
'Pop_size' : 'population_size',

'Initial_solutions' : 'initial_solutions',
'Crossover' : 'crossover_funcs' ,
'Mutations' : 'mutation_funcs',
'Update_mut_ratio' : 'update_mut_ratio' ,
'Mut_threshold' : 'mut_threshold',
'Repairs' : 'repairs',
'repair_func' : 'repairs',   

# SA Parameters

'Mut_update' : 'update_mut_ratio',
'Choice_import_saved_set' : 'initial_solutions',
'B_Mutations_AMALGAM' : 'mutation_funcs_Best_comp',
'Mutations_AMALGAM' : 'mutation_funcs_AMAL',
'Mutations_AMALGAM_every_n' : 'mutation_funcs_AMAL_every_n',
'Mutations_Counts_normal' : 'mutation_funcs_Counts_normal',
'Mutations_more' : 'mutation_funcs_more'
}

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
nl = '\n' # define new line character

#%%
def get_stacked_area_str(df_smooth_mut_ratios):
    if len(df_smooth_mut_ratios) < 400:
        n_th = 1
    elif len(df_smooth_mut_ratios) < 4000:
        n_th = 10
    else:
        n_th = 100
    
    preamble_stacked_area = r"""\documentclass[crop=false]{standalone}
%https://tex.stackexchange.com/questions/288373/how-to-draw-stacked-area-chart
\usepackage{pgfplots}
\usepgfplotslibrary{dateplot}
\pgfplotsset{compat=1.12}
\usetikzlibrary{fillbetween}
\usetikzlibrary{external}
\tikzexternalize[prefix=savedfigures/]""" + nl + nl

    doc_start = r"\begin{document}"+ nl +\
	f"\\tikzsetnextfilename{{{'Stacked_area_mut'}}}"+ nl +\
    r"""\begin{tikzpicture}[]
    \begin{axis}[
    legend style={at={(1.1,1)},anchor=north west}, %(<x>,<y>)
    %date coordinates in=x,
    table/col sep=comma,
    xticklabel style={rotate=90, anchor=near xticklabel},
    ymin=0,
    ymax=1,
    %max space between ticks=20,
    stack plots=y,%
    area style,"""+\
    f"""each nth point={n_th}, filter discard warning=false, unbounded coords=discard,
    xmin={0},
    xmax={len(df_smooth_mut_ratios)},
    xlabel = {counter_xlabel},
    ylabel = {counter_ylabel}
    ]""" + nl + nl

    doc_end = r"""    \end{axis}
    \end{tikzpicture}
\end{document}"""

    # Each entry part
    colours_stacked_area = ['blue', 'red', 'yellow', 'darkgray', 'green', 'magenta', 'cyan', 'orange', 'violet', 'lightgray', 'teal', 'pink', 'lime', 'black']
    doc_data_entries =""
    col_names = df_smooth_mut_ratios.columns
    for data_entries_i in range(2, len(col_names)):
        colour_entry = colours_stacked_area[data_entries_i-2]
        entry_name = col_names[data_entries_i]
        entry_str = f'''\\addplot [draw={colour_entry}, fill={colour_entry}!30!white] table [mark=none, x={counter_type},y={entry_name}
] {{Smoothed_Mutation_Ratios_for_Tikz.csv}}
    \closedcycle;
    \\addlegendentry{{{entry_name.replace("_", " ")}}};'''
        doc_data_entries = doc_data_entries + entry_str + nl + nl

    full_doc = preamble_stacked_area + doc_start + doc_data_entries + doc_end

    return full_doc

# %% Stacked mutation ratios
""" Creates the mutation ratio TikZ stacked area chart"""
for results_folder_name in result_entries:  
    working_folder = f"{path_to_main_folder / results_folder_name}/Mutations"
    if os.path.exists(f"{working_folder}/Smoothed_avg_mut_ratios.csv"):
        df_smooth_mut_ratios = pd.read_csv(f"{working_folder}/Smoothed_avg_mut_ratios.csv")
        df_smooth_mut_ratios = df_smooth_mut_ratios.drop(columns=['Total_iterations'])
        df_smooth_mut_ratios.rename(columns={'Unnamed: 0': counter_type}, inplace=True)
        df_smooth_mut_ratios.to_csv(f'{working_folder}/Smoothed_Mutation_Ratios_for_Tikz.csv', index=False)
        stacked_area_doc = get_stacked_area_str(df_smooth_mut_ratios)

        os.chdir(working_folder)
        with open('stacked_area_plot.tex','w') as file:
            file.write(stacked_area_doc)
        x = subprocess.call('pdflatex stacked_area_plot.tex -interaction nonstopmode -shell-escape') # -interaction nonstopmode -shell-escape
        if x != 0:
            print(f'Exit-code not 0, check result! [{results_folder_name}]')
        else:
            print(f"Success! Stacked Area [{results_folder_name}]")
        os.chdir(path_to_main_folder)


#%% Capture and save all data
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
            try:
                df_results_read = pd.read_csv(f"{path_to_main_folder / results_folder_name}/Results_description_HV.csv")
                df_results_read["parameter"] = parameter_name
                df_results_read["value"] = value
                #df_results_read.rename(columns={0 :'Measurement'}, inplace=True )
                # Get HV overall
                stats_dataframe = pd.read_csv(f"{path_to_main_folder / results_folder_name}/Stats_overall.csv")
                HV_overall = stats_dataframe[stats_dataframe.iloc[:,0] == "HV obtained"].iloc[0,1]
                df_results_read["HV_overall"] = HV_overall

                df_index = parameters_list.index(parameter_name)
                df_list_of_ST_results_HV[df_index] = df_list_of_ST_results_HV[df_index].append(df_results_read)
            except:
                print(f"Error in FOLDER: {results_folder_name} with PARAMETER: {parameter_name}")
        
        else:
            # Creates the new dataframe
            try:
                df_results_read = pd.read_csv(f"{path_to_main_folder / results_folder_name}/Results_description_HV.csv")
                df_results_read["parameter"] = parameter_name
                df_results_read["value"] = value
                #df_results_read.rename(columns={0 :'Measurement'}, inplace=True )
                # Get HV overall
                stats_dataframe = pd.read_csv(f"{path_to_main_folder / results_folder_name}/Stats_overall.csv")
                HV_overall = stats_dataframe[stats_dataframe.iloc[:,0] == "HV obtained"].iloc[0,1]
                df_results_read["HV_overall"] = HV_overall
                
                parameters_list.append(parameter_name)
                df_list_of_ST_results_HV.append(pd.DataFrame())
                
                df_index = parameters_list.index(parameter_name)
                df_list_of_ST_results_HV[df_index] = df_results_read
            except:
                print(f"Folder invalid: {results_folder_name}")


"""Print dataframes as .csv files"""
named_cols = ["test", "count", "mean", "std", "min", "lq", "med", "uq", "max", "outliers", "parameter", "value", "HV_overall"]
all_in_one_df = pd.DataFrame(columns = named_cols)
# assert (df_list_of_ST_results_HV[0].columns == named_cols).all()

if True:
    for parameter, results_dataframe in zip(parameters_list, df_list_of_ST_results_HV):
        results_dataframe.columns = named_cols
        
        if parameter in ['Mut_threshold','Mut_prob','Crossover_prob']:
            results_dataframe.value = results_dataframe.value.astype(float)
        elif parameter in ['Repairs', 'repair_func']:
            pass
        else:
            print(f"\n\nParameter not in list: {parameter}\n\n")
            results_dataframe.value = results_dataframe.value.astype(int)
        results_dataframe = results_dataframe.sort_values(by='value', ascending=True)
        results_dataframe.to_csv(path_to_main_folder / f"{prefix_for_each_csv_file}_{parameter}.csv")
        
        all_in_one_df = all_in_one_df.append(results_dataframe)

    all_in_one_df.to_csv(path_to_main_folder / f"{prefix_for_each_csv_file}_All_in_one.csv") #  all results in folder in one

#%% Box plots functions
import pandas as pd
import os
import subprocess
import re
import string
from pathlib import Path
import math

# Ensure the directory is set to the file location
dir_path = arg_dir
os.chdir(dir_path)

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

#spl_word = 'SA_' # initializing split word 
prefix_for_each_csv_file = f"UTRP_{spl_word}Outliers_Summary"


path_to_main_folder = Path(os.getcwd())
result_entries = os.listdir(path_to_main_folder)
         
parameters_list = []


box_plot_str = str()

""" Captures and saves all the data """
for file_name in result_entries:  
    if file_name.endswith('.csv') and not file_name.endswith('_All_in_one.csv'):
        df_results_read = pd.read_csv(f"{path_to_main_folder}/{file_name}")
        
        """Generate xticks"""
        x_tick_str = ",".join([str(x+1) for x in range(len(df_results_read["value"]))])
        x_tick_labels_str = ",".join([str(x) for x in df_results_read["value"]])
        
        """Add the prerequisites""" 
        temp_str = (f'% ######################## UTRP {meta_type} {title[df_results_read["parameter"].iloc[0]]} ######################## {nl}' \
            f'\\begin{{figure}} {nl}' \
         	f'\\centering {nl}' \
         	f'\\tikzsetnextfilename{{UTRP_{meta_full}_BP_{file_suffix[df_results_read["parameter"].iloc[0]]}}} {nl}' \
         	f'\\begin{{tikzpicture}} {nl}' \
          	f'\\begin{{axis}}[ {nl}' \
          	f'title={{{title[df_results_read["parameter"].iloc[0]]}}}, {nl}' \
          	f'boxplot/draw direction=y, {nl}' \
          	f'xtick={{{x_tick_str}}}, {nl}' \
          	f'xticklabels={{{x_tick_labels_str}}}, {nl}' \
          	f'x tick label style={{rotate=0, align=center}}, {nl}' \
          	f'xlabel={{{title[df_results_read["parameter"].iloc[0]]}}}, {nl}' \
            f'y tick label style={{/pgf/number format/.cd,fixed,precision=3, zerofill}}, {nl}' \
          	f'ylabel={{HV obtained [\\%]}}, {nl}' \
        	f'] {nl}') 
            
        box_plot_str = "\n".join([box_plot_str, temp_str])

        """Add each plot"""
        for row_i in range(len(df_results_read)):
            temp_str = (f'% ############## {df_results_read["parameter"].iloc[row_i]}={df_results_read["value"].iloc[row_i]} ################## {nl}' \
            f'\\addplot[boxplot, mark=asterisk, {nl}' \
            f'boxplot prepared={"{"} {nl}' \
			f'lower whisker={round(df_results_read["min"].iloc[row_i],5)}, {nl}' \
			f'upper whisker={round(df_results_read["max"].iloc[row_i],5)}, {nl}' \
			f'lower quartile={round(df_results_read["lq"].iloc[row_i],5)}, {nl}' \
			f'upper quartile={round(df_results_read["uq"].iloc[row_i],5)}, {nl}' \
			f'median={round(df_results_read["med"].iloc[row_i],5)}, {nl}' \
			f'average={round(df_results_read["mean"].iloc[row_i],5)}{"}"}, {nl}' \
    		f'color = blue, solid, area legend] {nl}' \
    		f'coordinates {{{outlier_logic(df_results_read["outliers"].iloc[row_i], row_i)}}}; {nl}'\
            f'\\addplot[only marks,mark=*,color = blue]coordinates{{({row_i+1},{round(float(df_results_read["HV_overall"].iloc[row_i]),5)})}}; {nl}')
                
            box_plot_str = "\n".join([box_plot_str, temp_str])
                
        """Add ending"""
       	temp_str = (f'\\end{{axis}}{nl}' \
       	f'\\end{{tikzpicture}}{nl}' \
        f'\\end{{figure}} {nl}') 
               
        box_plot_str = "\n".join([box_plot_str, temp_str])

if False: # prints text of boxplots   
    with open("Tikz_boxplots_UTRP_GA.txt", "w") as text_file:
        print(box_plot_str, file=text_file)


preamble = r'''\documentclass[crop=false]{standalone}
%\documentclass{standalone}
\usepackage{tikz} % To generate the plot from csv
\usepackage{pgfplots}
\usepackage{graphicx}
\usepackage{booktabs}
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
try:
    with open(('./'+result_entries[0]+'/Parameters/Sensitivity_list.txt'), 'r') as filehandle:
        sensitivity_list = json.load(filehandle)  
        print("LOADED: Sensitivity list\n")
        if (sensitivity_list[0][2] == "crossover_func"):
            sens_list_values = [x.replace('_',' ') for x in sensitivity_list[0][2:]]
        elif (sensitivity_list[0][2] == "mutation_funcs"):
            sens_list_values = [x[0].replace('_',' ') for x in sensitivity_list[0][2:]]
        else:
            sens_list_values = [x for x in sensitivity_list[0][2:]]
        df_sens_list = pd.DataFrame()
        df_sens_list = df_sens_list.assign(Index=list(range(0,len(sensitivity_list[0])-2)), 
                                           Name=sens_list_values)
        legend_table_str = df_sens_list.to_latex(index=False, column_format='l'*2, caption='Legend for the boxplot.')
        
except:
    print(f'Could not load sensitivity list: ./{result_entries[0]}/Parameters/Sensitivity_list.txt')

with open('box_plots.tex','w') as file:
    file.write(preamble)
    file.write('\n\\begin{document}\n')
    title_str = result_entries[0].split(' ')[0].replace('_',' ')
    file.write(f'\n\\section{{{title_str}}}\n')
    file.write(box_plot_str)
    #file.write('\n\section{Legend}\n')
    if df_results_read["parameter"].iloc[0] not in ['population_size', 'crossover_probability', 'mutation_probability']:
        try:
            file.write(legend_table_str)
        except NameError as err:
            print(f"{err}: name 'legend_table_str' is not defined")
    if False:
        for i in range(2,len(sensitivity_list[0])):
            legend_str = f'{i-2}\t{sensitivity_list[0][i][0]}'.replace('_',' ')
            file.write(legend_str+'\n\n')
            print(legend_str)
    #else:
        #print('Could not write sensitivity list to .tex file')
    file.write('\n\\end{document}\n')

x = subprocess.call('pdflatex box_plots.tex -interaction nonstopmode -shell-escape') # -interaction nonstopmode -shell-escape
if x != 0:
	print('Exit-code not 0, check result!')
else:
    if False:
        os.system('start box_plots.pdf') # opens up the file if successfull
