# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:42:36 2020

@author: 17832020
"""

import DSS_Admin as ga
import DSS_UTNDP_Functions as gf
import DSS_UTNDP_Classes as gc
import DSS_UTFSP_Functions as gf2
import DSS_Visualisation as gv
import EvaluateRouteSet as ev

pop_1.variables = pop_1.variables + pop_1.variables
pop_1.variables = [pop_1.variables[i] for i in survivor_indices]

[i for i in survivor_indices]




class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        
        return bin(int(a,2) + int(b,2))[2:]
    
a = "11" 
b = "1"  

# Output: "100"
obj = Solution()
obj.addBinary(a, b)

a = "1010" 
b = "1011"

# Output: "10101"
obj = Solution()
obj.addBinary(a, b)