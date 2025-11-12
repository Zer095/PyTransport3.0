####################################### Setup file for the double quadratic potential evolved with SIGMA Matrices ######################################################

import sympy as sym # we import the sympy package
import math         # we import the math package (not used here, but has useful constants such math.pi which might be needed in other cases)
import sys          # we import the sys module used below 
import os
from pathlib import Path
############################################################################################################################################

from PyTransport.PyTransPy import PyTransSetup 

PyTransSetup.pathSet()
############################################################################################################################################

nF = 2  # number of fields needed to define the double quadratic potential
nP = 2  # number of parameters needed to define the double quadratic potential
f = sym.symarray('f', nF)   # an array representing the nF fields present for this model
p = sym.symarray('p', nP)   # an array representing the nP parameters needed to define this model (that we might wish to change)
                            # if we don't wish to change them they could be typed explicitly in the potential below

V = (1./2.)*p[0]**2.0*f[0]**2.0 + (1./2.) * p[1]**2.0 * f[1]**2.0   # this is the potential written in sympy notation

# The last argument is for wheter the sympy's simplify is used to on derivatives of the potential and field geometric quantities.
# Caution is recommended as simpy's simplify is known to have bugs. Simplification can increase the speed of numerically evolutions, but at the cost of compiling more slowly.

PyTransSetup.potential(V,nF,nP,True)    # writes this potential and its derivatives into C++ file potential.h when run 
PyTransSetup.compileName("DQ")  # this compiles a python module using the C++ code, including the edited potential potential.h file, called PyTransDQuadSigma
                                        # and places it in the location folder ready for use

############################################################################################################################################