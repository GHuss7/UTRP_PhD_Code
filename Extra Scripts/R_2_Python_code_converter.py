# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:38:19 2019

@author: 17832020
"""

from pyensae.languages import r2python

rscript = """
nb=function(y=1930){
debut=1816
MatDFemale=matrix(D$Female,nrow=111)
colnames(MatDFemale)=(debut+0):198
cly=(y-debut+1):111
deces=diag(MatDFemale[:,cly[cly%in%1:199]])
return(c(B$Female[B$Year==y],deces))}
"""


print(r2python(rscript, pep8=True))

