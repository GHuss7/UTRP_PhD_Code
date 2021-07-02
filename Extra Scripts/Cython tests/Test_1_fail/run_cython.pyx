# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:38:06 2021

@author: gunth
"""

cpdef int test(int x):
    cdef int y = 1
    cdef int i
    for i in range(1, x+1):
        y *= i
    return y