############################## Set up file for the Axion Monodromy Quadratic ###########################
import sympy as sym
from gravipy import *

#################################################################
from PyTransport.PyTransPy import PyTransSetup, PyTransScripts

PyTransSetup.pathSet()
#############################################################
nF = 2
nP = 6
f = sym.symarray('f', nF) 
p = sym.symarray('p', nP)

V = 0.5*p[1]*f[0]**2 + p[2]*sym.cos(f[0]/p[3]) + 0.5*p[4]*(f[1] - p[5])**2
G = sym.Matrix( [ [sym.exp(f[1]/p[0]), 0], [0, 1] ] )
#############################################################
PyTransSetup.potential(V,nF,nP,False,G)
PyTransSetup.compileName("AMQ", True)