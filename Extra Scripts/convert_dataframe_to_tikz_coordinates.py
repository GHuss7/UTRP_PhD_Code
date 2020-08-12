# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:36:30 2020

@author: 17832020
"""

data = mumford_results_acsv[1:,] # change the input data here
text = str()

switch_columns = True

for i in range(len(data)):
    if switch_columns:
        text = text + "("+str(data[i,1])+","+str(data[i,0])+")\n"
        
    else:
        text = text + "("+str(data[i,0])+","+str(data[i,1])+")\n"
    
print(text) # copy the output after this