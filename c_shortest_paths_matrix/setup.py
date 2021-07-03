# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:25:14 2021

@author: gunth
"""

# Old setup:
# from setuptools import setup
# from Cython.Build import cythonize


# setup(
#     ext_modules = cythonize("shortest_paths_matrix.pyx")
# )

# Run > cd directory
#     >python setup.py build_ext --inplace

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules=[
#         Extension("shortest_paths_matrix", ["shortest_paths_matrix.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

# setup(
#     ext_modules=cythonize("shortest_paths_matrix.pyx"),
#     include_dirs=[numpy.get_include()]
# ) 


#%%
# from: https://stackoverflow.com/questions/11368486/openmp-and-python
# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# setup(
#       cmdclass = {'build_ext': build_ext},
#       ext_modules = [Extension("calculate",
#                                 ["shortest_paths_matrix.pyx"],
#                                 extra_compile_args=['-fopenmp'],
#                                 extra_link_args=['-fopenmp'],
#                                 include_dirs=[numpy.get_include()])]
      
#       ) 


#%%
# from: https://blog.paperspace.com/boosting-python-scripts-cython/
import distutils.core
import Cython.Build
import numpy 
distutils.core.setup(
    ext_modules = Cython.Build.cythonize("shortest_paths_matrix_master.pyx"),
    include_dirs=[numpy.get_include()]
    )