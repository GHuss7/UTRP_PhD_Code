# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 08:49:16 2020

@author: 17832020
"""

import pandas as pd
import os
import re
import string
from pathlib import Path


path_to_main_folder = Path(os.getcwd())
file_names = os.listdir(path_to_main_folder)
         




""" Captures and saves all the data """
for file_name in file_names:  
    if file_name.lower().endswith(".txt"):
        print(f"{file_name}")
        file = open(file_name, "r")
        
        with open(file_name) as fp:
            for line in fp:

                numbers.extend( #Append the list of numbers to the result array
                               [int(item) #Convert each number to an integer
                                for item in line.split() #Split each line of whitespace
                                ])
            
print(numbers)
