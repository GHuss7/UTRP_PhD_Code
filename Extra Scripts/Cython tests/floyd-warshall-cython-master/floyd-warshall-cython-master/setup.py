# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:25:14 2021

@author: gunth
"""

# Old setup:
# from setuptools import setup
# from Cython.Build import cythonize

# setup(
#     ext_modules = cythonize("floyd_warshall.pyx")
# )


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("floyd_warshall", ["floyd_warshall.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize("floyd_warshall.pyx"),
    include_dirs=[numpy.get_include()]
)  