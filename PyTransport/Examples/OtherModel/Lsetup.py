############################## Set up file for the Laura's model ###########################
import sympy as sym
import math
from gravipy import *
from pathlib import Path

#################################################################
from PyTransPy import PyTransSetup, PyTransScripts

PyTransSetup.pathSet()
#############################################################

nF = 2
nP = 4
f = sym.symarray('f', nF) 
p = sym.symarray('p', nP)

V = p[0]*( f[0]**2 / (p[1]**2+f[0]**2) ) + (1./2.)*p[2]*f[1]**2
G = sym.Matrix( [[1, 0], [0, sym.exp(2.*p[3]*f[0]) ]] )

PyTransSetup.potential(V,nF,nP,False,G)
PyTransSetup.compileName("LNC", True)
