# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:42:58 2021

@author: gunth
"""


from distutils.core import setup
from cython.build import cythonize

setup(ext_modules = cythonize('run_cython.pyx'))