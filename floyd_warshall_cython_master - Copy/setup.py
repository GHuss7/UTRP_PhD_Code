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

# Run > cd directory
#     >python setup.py build_ext --inplace

# from distutils.core import setup, Extension
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules=[
#         Extension("floyd_warshall", ["floyd_warshall.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

# setup(
#     ext_modules=cythonize("floyd_warshall.pyx"),
#     include_dirs=[numpy.get_include()]
# ) 

#%%
# # from: https://stackoverflow.com/questions/11368486/openmp-and-python
# from Cython.Distutils import build_ext
# setup(
#       cmdclass = {'build_ext': build_ext},
#       ext_modules = [Extension("calculate",
#                                 ["floyd_warshall.pyx"],
#                                 extra_compile_args=['-fopenmp'],
#                                 extra_link_args=['-fopenmp'])]
      
#       ) 

#%% Cython docs
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "floyd_warshall",
        ["floyd_warshall.pyx"],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp'],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name='floyd_warshall',
    ext_modules=cythonize(ext_modules),
)

