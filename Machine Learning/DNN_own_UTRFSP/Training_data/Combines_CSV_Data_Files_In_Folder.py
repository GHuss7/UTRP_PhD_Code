# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:48:53 2020

@author: 17832020
"""
import os
import glob
import pandas as pd
from pathlib import Path


path_to_folder = Path(os.getcwd())
os.chdir(path_to_folder)

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "Combined_Data.csv", index=False, encoding='utf-8-sig')