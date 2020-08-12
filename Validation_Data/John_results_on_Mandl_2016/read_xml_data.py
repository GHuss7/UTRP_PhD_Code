# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:07:02 2020

@author: 17832020
"""

import xml.etree.ElementTree as ET
import pandas as pd
import DSS_UTNDP_Functions as gf

tree = ET.parse('Mandl.xml') # read in XML file 
root = tree.getroot()

# root.tag
# root.attrib

    
            
if False: # tests   
    for child_1 in root:
        for child_2 in child_1:
            for child_3 in child_2:
                print(child_3.text)
                #print(child_3.objective.text)
     
    root[0][0][0].text # get first objective
    root[0][0][1].text # get second objective
    root[0][2][0][0].text # get route 1 nodes
    root[0][2][1][0].text # get route 2 nodes
    
    root[1][0][0].text # get nexr objective

"""FINAL"""

df_results = pd.DataFrame(columns = ["f_1_ATT","f_2_TRT","Routes"])

for i, solution in zip(range(len(root)), root):
    routes_list = []
    for route in solution[2]:
        route_list = []
        for node in route:
            route_list.append(node.text)
        routes_list.append(route_list)
    
    #print(solution[0][0].text, solution[0][1].text)
    df_results.loc[i] = [solution[0][0].text, solution[0][1].text, gf.convert_routes_list2str(routes_list)]
    
df_results.to_csv("Results_data.csv")   # save to csv
