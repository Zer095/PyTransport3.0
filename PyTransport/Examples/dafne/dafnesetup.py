############################## Set up file for the Axion Monodromy Quadratic ###########################
import sympy as sym
from gravipy import *

#################################################################
from PyTransport.PyTransPy import PyTransSetup, PyTransScripts

PyTransSetup.pathSet()
#############################################################
nF=1
nP=3
f = sym.symarray('f', nF) 
p = sym.symarray('p', nP)

V = p[0]*(1 - sym.exp(p[1]*f[0]))**p[2]
#############################################################
PyTransSetup.potential(V,nF,nP,True)
PyTransSetup.compileName("Dafne") 