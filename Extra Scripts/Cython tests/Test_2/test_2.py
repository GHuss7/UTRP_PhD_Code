# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:23:20 2021

@author: gunth

from: https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
"""

# with Python console: 
# > cd directory     
# > python setup.py build_ext --inplace

import cython
import pyximport; pyximport.install()

import helloworld
