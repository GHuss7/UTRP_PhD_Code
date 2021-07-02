# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:25:14 2021

@author: gunth
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("helloworld.pyx")
)